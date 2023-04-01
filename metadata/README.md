### Generate `metadata_iemocap.csv` file for IEMOCAP
* `metadata_iemocap.csv` records the sample `name` and `label`.
* For example, the names and lables of session 1 are in `PATH_to_IEMOCAP/Session1/dialog/EmoEvaluation/*.txt`.
* You need to record the sample names and labels from all the `.txt` files in the `EmoEvaluation` directory of five sessions into `metadata_iemocap.csv`.

### Generate `metadata_meld.csv` file for MELD
* `metadata_meld.csv` records the sample `name`, `label`, and `state`.
* After downloading the MELD, you can see three `.csv` files provided by the official, named `train_sent_emo.csv`, `dev_sent_emo.csv`, and `test_sent_emo.csv`. 
* In the official `.csv` files, the `Emotion` column is the `label` we need. The `Dialogue_ID` and `Utterance_ID` together make up the `name` in `metadata_meld.csv`.
* `state` column records whether the sample is from the training set, development set, or test set.

### Generate `metadata_pitt_crop.csv` file for Pitt
* `metadata_pitt_crop.csv` records the sample `name` and `label`.
* The official `.csv` files are in `PATH_to_PITT/label/Control/cookie.csv` and `PATH_to_PITT/label/Dementia/cookie.csv`.
* Pitt provides speech samples at the dialogue level. We need to extract each utterance spoken by the subjects and save each utterance as a separate audio file.
* For example, there are 18 utterances in the `002-0.mp3` dialogue from the subject. We save these utterances to `002-0-0.wav`, `002-0-1.wav`, ..., `002-0-17.wav`.
* The `name` in `metadata_pitt_crop.csv` is the filename of the extracted audio file and the `label` is `Control` or `Dementia`.

### Generate `metadata_daicwoz_crop_resample.csv` file for DAIC-WOZ
* `metadata_daicwoz_crop_resample.csv` records the sample `name`, `label`, and `state`.
* Similar to the Pitt, the DAIC-WOZ corpus provides speech samples at the dialogue level.
* The official `.csv` files are in `PATH_to_DAICWOZ/train_split_Depression_AVEC2017.csv` and `PATH_to_DAICWOZ/dev_split_Depression_AVEC2017.csv`.  Since the labels of test data are not provided, we use only the train set and the development set. The `PHQ8_Binary` value is used as the `label` in our experiment.
* `XXX_P/` folder (`XXX` is the Participant_ID) includes `XXX_AUDIO.wav` and `XXX_TRANSCRIPT.csv`. The transcript file contains the start time and stop time of each utterance. Based on these timestamps, we extract the participant's utterances and save them as separate files. For example, `XXX_sY_AUDIO.wav` denotes the `Y-th` utterance of the participant `XXX`. Also, `XXX_sY_AUDIO.wav` is the `name` in `metadata_daicwoz_crop_resample.csv`.
* Since the DAIC-WOZ suffers from sample imbalance and some participants' utterances are quite short, we resample the corpus in our experiment. In detail, when the participant belongs to label `0`, we select his/her longest 18 utterances as the training or testing sample. The longest 46 utterances are selected when the participant belongs to label `1`.
* `state` column records whether the sample is from the training set or the development set.
