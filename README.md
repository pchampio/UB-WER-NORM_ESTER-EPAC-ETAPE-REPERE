UB-WER and WER scoring with text normalization for ESTER-EPAC-ETAPE-REPERE
==============================================

### Install
Install speechbrain plus the folowing:
```sh
pip3 install spacy
python3 -m spacy download fr_core_news_md
pip3 install unidecode
```

### Run
Modify the variable of ./run.sh according to what you whish to evaluate + the
paths
```sh
./run.sh
```

### Results
```sh
# EPAC UB-WER & WER with text norm
tail ./scoring_files/EPAC_char_k2.wer
# Original WER from speechbrain without text norm
tail ./scoring_files/EPAC_char_k2_SB_OUT.txt
```
The scoring_files example results are zipped with a password since the data isn't free.  
The zip file was created with:
```
zip --password $(cat $DATA_PATH/ESTER1/ester1_train.lst $DATA_PATH/ESTER2/train.lst  $DATA_PATH/EPAC/epac.lst  $DATA_PATH/ETAPE/train/BFMTV_BFMStory_2010-09-03_175900.stm $DATA_PATH/REPERE/train/BFMTV_BFMStory_2011-05-11_175900.stm | md5sum | awk '{ print $1 }') scoring_files.zip -r scoring_files
```
