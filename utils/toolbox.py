
import torch
import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def _majority_target_Pitt(source_tag: list):
    return [re.match('.*-.*-', mark).group() for mark in source_tag]

def _majority_target_DAIC_WOZ(source_tag: list):
    return [mark.split('_')[0] for mark in source_tag]

def majority_vote(source_tag: list, source_value: torch.Tensor, source_label: torch.Tensor, modify_tag, task='classification'):
    '''
    Args:
    source_tag: Guideline for voting, e.g. sample same.
    source_value: value before voting.
    source_label: label before voting.
    task: classification / regression

    Return:
    target: voting object.
    vote_value: value after voting.
    vote_label: label after voting.
    '''
    source_tag = modify_tag(source_tag)
    target = set(source_tag)
    vote_value_dict = {t:[] for t in target}
    vote_label_dict = {t:[] for t in target}

    if task == 'regression':
        logit_vote = True
    else:
        if source_value.dim() != 1:
            logit_vote = True
        else:
            logit_vote = False

    for i, (mark) in enumerate(source_tag):
        value = source_value[i]
        label = source_label[i]
        vote_value_dict[mark].append(value)
        vote_label_dict[mark].append(label)
    for key, value in vote_value_dict.items():
        if logit_vote:
            logit = torch.mean(torch.stack(value, dim=0), dim=0)
            if task == 'regression':
                vote_value_dict[key] = logit
            else:
                vote_value_dict[key] = torch.argmax(logit)
        else:
            vote_value_dict[key] = max(value, key=value.count)

    vote_value, vote_label = [], []
    for t in target:
        vote_value.append(vote_value_dict[t])
        vote_label.append(vote_label_dict[t][0])

    vote_value = torch.tensor(vote_value)
    vote_label = torch.tensor(vote_label)
    
    return target, vote_value, vote_label

def calculate_score_classification(preds, labels, average_f1='weighted'):  # weighted, macro
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average_f1, zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    ua = recall_score(labels, preds, average='macro', zero_division=0)
    confuse_matrix = confusion_matrix(labels, preds)
    return accuracy, ua, f1, precision, confuse_matrix

def calculate_basic_score(preds, labels):
    return accuracy_score(labels, preds)

def tidy_csvfile(csvfile, colname, ascending=True):
    '''
    tidy csv file base on a particular column.
    '''
    print(f'tidy file: {csvfile}, base on column: {colname}')
    df = pd.read_csv(csvfile)
    df = df.sort_values(by=[colname], ascending=ascending, na_position='last')
    df = df.round(3)
    df.to_csv(csvfile, index=False, sep=',')

    
