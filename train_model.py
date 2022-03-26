#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 18:16:12 CST 2021
@author: lab-chen.weidong
"""

import os
from tqdm import tqdm
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import json
import argparse

import utils
from config import Cfg, create_workshop, modify_config

class Engine():
    def __init__(self, cfg, local_rank: int, world_size: int):
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = world_size
        self.ckpt_save_path = self.cfg.ckpt_save_path
        self.device = self.cfg.train.device
        self.EPOCH = self.cfg.train.EPOCH
        self.current_epoch = 0
        self.iteration = 0
        
        if self.cfg.train.find_init_lr:
            self.cfg.train.lr = 0.000001
            self.cfg.train.step_size = 1
            self.cfg.train.gamma = 1.05
            if self.local_rank == 0:
                self.writer = SummaryWriter(self.cfg.workshop)

        ### prepare model and train tools
        with open('./config/model_config.json', 'r') as f1, open(f'./config/{self.cfg.dataset.database}_feature_config.json', 'r') as f2:
            model_json = json.load(f1)[self.cfg.model.type]
            feas_json = json.load(f2)
            data_json = feas_json[self.cfg.dataset.feature]
            data_json['meta_csv_file'] = feas_json['meta_csv_file']
            model_json['num_classes'] = feas_json['num_classes']
            model_json['input_dim'] = (data_json['feature_dim'] // model_json['num_heads']) * model_json['num_heads']
            model_json['ffn_embed_dim'] = model_json['input_dim'] // 2
            model_json['hop'] = data_json['hop']
        
        self.model = utils.model.load_model(self.cfg.model.type, self.device, **model_json)
        self.optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, self.model.parameters()), lr=self.cfg.train.lr, momentum=0.9)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.calculate_score = utils.toolbox.calculate_score_classification
        self.data_loader_feactory = utils.dataset.DataloaderFactory(self.cfg)
        self.train_dataloader = self.data_loader_feactory.build(state='train', **data_json)
        self.test_dataloader = self.data_loader_feactory.build(state='test', **data_json)
        if self.cfg.train.find_init_lr:
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.train.step_size, gamma=self.cfg.train.gamma)
        else:
            self.scheduler = lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.EPOCH, eta_min=self.cfg.train.lr / 100)
        
        ### prepare logger
        self.logger_train = utils.logger.create_logger(self.cfg.workshop, name='train') if self.local_rank == 0 else None
        self.logger_test = utils.logger.create_logger(self.cfg.workshop, name='test') if self.local_rank == 0 else None
        if self.logger_train is not None:
            self.logger_train.info(f'workshop: {self.cfg.workshop}')
            self.logger_train.info(f'seed: {self.cfg.train.seed}')
            self.logger_train.info(f'pid: {os.getpid()}')

        ### prepare meters
        data_type = torch.int64
        self.loss_meter = utils.avgmeter.AverageMeter(device='cpu')
        self.score_meter = utils.avgmeter.AverageMeter(device='cpu')
        self.predict_recoder = utils.recoder.TensorRecorder(device='cuda', dtype=data_type)
        self.label_recoder = utils.recoder.TensorRecorder(device='cuda', dtype=data_type)
        self.tag_recoder = utils.recoder.StrRecorder(device='cpu', dtype=str)
        self.train_score_1, self.train_score_2, self.train_score_3, self.train_loss = [], [], [], []
        self.test_score_1, self.test_score_2, self.test_score_3, self.test_loss = [], [], [], []
        
    def reset_meters(self):
        self.loss_meter.reset()
        self.score_meter.reset()
    
    def reset_recoders(self):
        self.predict_recoder.reset()
        self.label_recoder.reset()
        self.tag_recoder.reset()

    def gather_distributed_data(self, gather_data):
        if isinstance(gather_data, torch.Tensor):
            _output = [torch.zeros_like(gather_data) for _ in range(self.world_size)]
            dist.all_gather(_output, gather_data, async_op=False)
            output = torch.cat(_output)
        else:
            if gather_data[0] is not None:
                _output = [None for _ in range(self.world_size)]
                if hasattr(dist, 'all_gather_object'):
                    dist.all_gather_object(_output, gather_data)
                else:
                    utils.distributed.all_gather_object(_output, gather_data, self.world_size)
                output = []
                for lst in _output:
                    output.extend(lst)
            else:
                output = None
        return output

    def train_epoch(self):
        self.train_dataloader.set_epoch(self.current_epoch)
        if self.local_rank == 0:
            print(f'-------- {self.cfg.workshop} --------')
        discrip_str = f'Epoch-{self.current_epoch}/{self.EPOCH}'
        pbar_train = tqdm(self.train_dataloader, disable=self.local_rank != 0)
        pbar_train.set_description('Train' + discrip_str)

        self.reset_meters()
        self.reset_recoders()

        self.model.train()
        for data in pbar_train:
            self.iteration += 1

            x = torch.stack(data[0], dim=0).to(self.device)
            y = torch.tensor(data[1]).to(self.device)
            vote_tag = data[2]
            batch_size = y.shape[0]

            self.optimizer.zero_grad()

            out = self.model(x)
            loss = self.loss_func(out, y)

            loss.backward()
            self.optimizer.step()

            y_pred = torch.argmax(out, dim=1)

            self.predict_recoder.record(y_pred)
            self.label_recoder.record(y)
            self.tag_recoder.record(vote_tag)

            score = utils.toolbox.calculate_basic_score(y_pred.cpu(), y.cpu())
            self.loss_meter.update(loss.item())
            self.score_meter.update(score, batch_size)

            pbar_train_dic = OrderedDict()
            pbar_train_dic['iter'] = self.iteration
            pbar_train_dic['lr'] = self.optimizer.param_groups[0]['lr']
            pbar_train_dic['score'] = f'{self.score_meter.avg:.5f}'  # acc / MSE in local_rank: 0
            pbar_train_dic['loss'] = f'{self.loss_meter.avg:.5f}'    # loss in local_rank: 0
            pbar_train.set_postfix(pbar_train_dic)

            if self.cfg.train.find_init_lr:
                if loss.item() > 20:
                    raise ValueError(f'Loss: {loss.item()} started to expand. Please use tensorboard to find the appropriate lr.')
                if self.local_rank == 0:
                    self.writer.add_scalar('Step Loss', loss.item(), self.iteration)
                    self.writer.add_scalar('Total Loss', self.loss_meter.avg, self.iteration)
                    self.writer.add_scalar('Step LR', self.optimizer.param_groups[0]['lr'], self.iteration)
                self.scheduler.step()

        epoch_preds = self.gather_distributed_data(self.predict_recoder.data).cpu()
        epoch_labels = self.gather_distributed_data(self.label_recoder.data).cpu()
        epoch_tag = self.gather_distributed_data(self.tag_recoder.data)
        
        if self.local_rank == 0:
            average_f1 = 'weighted' if self.cfg.dataset.database == 'meld' else 'macro'
            score_1, score_2, score_3, score_4, score_5 = self.calculate_score(epoch_preds, epoch_labels, average_f1)

            self.train_score_1.append(score_1)
            self.train_score_2.append(score_2)
            self.train_score_3.append(score_3)
            self.train_loss.append(self.loss_meter.avg)

        if self.logger_train is not None:
            self.logger_train.info(
                f'Training epoch: {self.current_epoch}, accuracy: {score_1:.5f}, precision: {score_4:.5f}, recall: {score_2:.5f}, F1: {score_3:.5f}, loss: {self.loss_meter.avg:.5f}'
            )           
    
    def test(self):
        assert self.test_dataloader is not None, print('Test dataloader is not defined')
        discrip_str = f'Epoch-{self.current_epoch}'
        pbar_test = tqdm(self.test_dataloader, disable=self.local_rank != 0)
        pbar_test.set_description('Test' + discrip_str)

        self.reset_meters()
        self.reset_recoders()

        self.model.eval()
        with torch.no_grad():
            for data in pbar_test:
                x = torch.stack(data[0], dim=0).to(self.device)
                y = torch.tensor(data[1]).to(self.device)
                vote_tag = data[2]

                batch_size = y.shape[0]

                out = self.model(x)
                loss = self.loss_func(out, y)

                y_pred = torch.argmax(out, dim=1)

                self.predict_recoder.record(y_pred)
                self.label_recoder.record(y)
                self.tag_recoder.record(vote_tag)

                score = utils.toolbox.calculate_basic_score(y_pred.cpu(), y.cpu())

                self.loss_meter.update(loss.item())
                self.score_meter.update(score, batch_size)

                pbar_test_dic = OrderedDict()
                pbar_test_dic['score'] = f'{self.score_meter.avg:.5f}'
                pbar_test_dic['loss'] = f'{self.loss_meter.avg:.5f}'
                pbar_test.set_postfix(pbar_test_dic)

            epoch_preds = self.gather_distributed_data(self.predict_recoder.data).cpu()
            epoch_labels = self.gather_distributed_data(self.label_recoder.data).cpu()
            epoch_tag = self.gather_distributed_data(self.tag_recoder.data)
        
            if self.local_rank == 0:
                if hasattr(self.cfg.train, 'vote'):
                    if self.cfg.dataset.database == 'pitt':
                        modify_tag_func = utils.toolbox._majority_target_Pitt
                    elif self.cfg.dataset.database == 'daic_woz':
                        modify_tag_func = utils.toolbox._majority_target_DAIC_WOZ
                    else:
                        raise KeyError(f'Database: {self.cfg.dataset.database} do not need voting!')
                    _, epoch_preds, epoch_labels = utils.toolbox.majority_vote(epoch_tag, epoch_preds, epoch_labels, modify_tag_func)
                average_f1 = 'weighted' if self.cfg.dataset.database == 'meld' else 'macro'
                score_1, score_2, score_3, score_4, score_5 = self.calculate_score(epoch_preds, epoch_labels, average_f1)

                self.test_score_1.append(score_1)
                self.test_score_2.append(score_2)
                self.test_score_3.append(score_3)
                self.test_loss.append(self.loss_meter.avg)

            if self.logger_test is not None:
                self.logger_test.info(
                    f'Testing epoch: {self.current_epoch}, accuracy: {score_1:.5f}, precision: {score_4:.5f}, recall: {score_2:.5f}, F1: {score_3:.5f}, loss: {self.loss_meter.avg:.5f}, confuse_matrix: \n{score_5}'
                )

    def model_save(self, is_best=False):  
        ckpt_save_file = os.path.join(self.ckpt_save_path, 'best.pt') if is_best \
            else os.path.join(self.ckpt_save_path, f'epoch{self.current_epoch}.pt')
        save_dict = {
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
            }
        torch.save(save_dict, ckpt_save_file)

    def run(self):
        if self.cfg.train.find_init_lr:
            while self.current_epoch < self.EPOCH:
                self.train_epoch()
                self.current_epoch += 1
        else:
            plot_sub_titles = ['WA-train', 'UA-train', 'F1-train', 'Loss-train', 'WA-test', 'UA-test', 'F1-test', 'Loss-test']
            plot_data_name = ['train_score_1', 'train_score_2', 'train_score_3', 'train_loss', 'test_score_1', 'test_score_2', 'test_score_3', 'test_loss']

            while self.current_epoch < self.EPOCH:
                self.train_epoch()
                self.scheduler.step()
                self.test()

                self.current_epoch += 1

                if self.local_rank == 0:
                    plot_data = [getattr(self, data_name) for data_name in plot_data_name]
                    utils.write_result.plot_process(plot_data, plot_sub_titles, self.cfg.workshop)

        utils.logger.close_logger(self.logger_train)
        utils.logger.close_logger(self.logger_test)

def main_worker(local_rank, cfg, world_size, dist_url):
    utils.environment.set_seed(cfg.train.seed + local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=dist_url,
        world_size=world_size,
        rank=local_rank,
    )
    
    if cfg.dataset.database == 'iemocap':
        cfg.train.strategy = '5cv'
        folds = [1, 2, 3, 4, 5]
    elif cfg.dataset.database == 'meld':
        folds = [1]
    elif cfg.dataset.database == 'pitt':
        cfg.train.vote = True
        folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif cfg.dataset.database == 'daic_woz':
        cfg.train.vote = True
        folds = [1]
    else:
        raise KeyError(f'Unknown database: {cfg.dataset.database}')

    for f in folds:
        cfg_clone = cfg.clone()
        cfg_clone.train.current_fold = f
        create_workshop(cfg_clone, local_rank)
        engine = Engine(cfg_clone, local_rank, world_size)
        engine.run()
        torch.cuda.empty_cache()

    if local_rank == 0:
        criterion = ['accuracy', 'precision', 'recall', 'F1']
        evaluate = cfg.train.evaluate
        outfile = f'result/result_{cfg.model.type}.csv'
        utils.write_result.path_to_csv(os.path.dirname(cfg_clone.workshop), criterion, evaluate, csvfile=outfile)

def main(cfg):
    utils.environment.visible_gpus(cfg.train.device_id)
    utils.environment.set_seed(cfg.train.seed)

    free_port = utils.distributed.find_free_port()
    dist_url = f'tcp://127.0.0.1:{free_port}'   
    world_size = torch.cuda.device_count()    # num_gpus
    print(f'world_size={world_size} Using dist_url={dist_url}')

    mp.spawn(fn=main_worker, args=(cfg, world_size, dist_url), nprocs=world_size)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mo", "--model.type", help="modify cfg.train.model.type", type=str)
    parser.add_argument("-d", "--dataset.database", help="modify cfg.dataset.database", type=str)
    parser.add_argument("-f", "--dataset.feature", help="modify cfg.dataset.feature", type=str)
    parser.add_argument("-g", "--train.device_id", help="modify cfg.train.device_id", type=str)
    parser.add_argument("-m", "--mark", help="modify cfg.mark", type=str)
    parser.add_argument("-s", "--train.seed", help="modify cfg.train.seed", type=int)
    args = parser.parse_args()

    modify_config(Cfg, args)
    main(Cfg)
    
