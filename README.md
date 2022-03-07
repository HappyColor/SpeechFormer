# Framework: SpeechFormer
![SpeechFormer](./figures/framework.png)
Paper：SpeechFormer: A Hierarchical Efficient Framework Incorporating the Characteristics of Speech

# Prepare data
## Download datasets
* Speech emotion recognition: [IEMOCAP](https://sail.usc.edu/iemocap/index.html), [MELD](https://affective-meld.github.io/)  
* Alzheimer’s disease detection: [Pitt](https://dementia.talkbank.org/)  
* Depression classification: [DAIC-WOZ](https://dcapswoz.ict.usc.edu/)  

## Extract acoustic feature
```
python ./extract_feature/extract_spec.py
python ./extract_feature/extract_logmel.py
python ./extract_feature/extract_wav2vec.py
```

# Train model
