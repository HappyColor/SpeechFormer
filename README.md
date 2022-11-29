# SpeechFormer
![SpeechFormer](./figures/framework.png)
Paper：[SpeechFormer: A Hierarchical Efficient Framework Incorporating the Characteristics of Speech](https://arxiv.org/abs/2203.03812)  
This paper was submitted to INTERSPEECH 2022.

# Getting started
## Install dependencies
All dependencies can be installed using pip:
```
python -m pip install -r requirements.txt
```
Our experiments run on Python 3.6 and PyTorch 1.5. Other versions should work but are not tested.

## Prepare data
### Download datasets
* Speech emotion recognition: [IEMOCAP](https://sail.usc.edu/iemocap/index.html), [MELD](https://affective-meld.github.io/)  
* Alzheimer’s disease detection: [Pitt](https://dementia.talkbank.org/)  
* Depression classification: [DAIC-WOZ](https://dcapswoz.ict.usc.edu/)  

Note that you should create a metadata file (`.csv` format) for each dataset to record the `name` and `label` (and `state`, e.g. `train` or `dev` or `test`) of the samples. Then modify the argument: `meta_csv_file` in `./config/xxx_feature_config.json` according to the absolute path of the corresponding `.csv` file. The example `.csv` files are in the `./metadata` directory.

### Extract acoustic feature
* Three acoustic features are extracted from each audio sample, including `spectrogram (Spec)`, `Log-Mel spectrogram (Logmel)` and `pre-trained Wav2vec`.  
* Each extracted feature is saved in `.mat` format using `scipy`.  
* The pre-trained wav2vec model is publicly available at [here.](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec)
```
python ./extract_feature/extract_spec.py
python ./extract_feature/extract_logmel.py
python ./extract_feature/extract_wav2vec.py
```
Modify the argument: `matdir` in `./config/xxx_feature_config.json` to the folder path of your extracted feature.

## Train model
Set the hyper-parameters on `./config/config.py` and `./config/model_config.json`.  
Note: the value of `expand` in `./config/model_config.json` for SpeechFormer-S is `[1, 1, 1, -1]`, while that of SpeechFormer-B is `[1, 1, 2, -1]`.  
Next, run:
```
python train_model.py
```
You can also pass the hyper-parameters from the command line for convenience, more details can be found in `train_model.py`.
  
