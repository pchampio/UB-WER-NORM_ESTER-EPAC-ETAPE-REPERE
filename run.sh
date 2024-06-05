#!/usr/bin/env bash

## Install additional deps ##
# pip3 install spacy
# python3 -m spacy download fr_core_news_md
# pip3 install unidecode

set -e

mkdir -p ./scoring_files || true

tsv_h=false

# =-----=

# slug=whisper_largev3
# tsv_h=true
# cp -f ./examples/whisper_largev3_h.txt ./scoring_files/${slug}_h.txt

# dset=EPAC
dset=ESTER1
# dset=ESTER2
# dset=ETAPE
# dset=REPERE2014        # Modify stm var & Modify wrong stm ref pers=Mathieu_LADEVEZE>Barcella&lt
model=phone_k2/11
# model=char_k2/10
sb_out=../results/train_ESTER2+train_ESTER1+train_EPAC+train_ETAPE+train_REPERE_wav2vec2_$model/metric_test_$dset/wer_HLG_1best.txt
# sb_out=../results/train_ESTER2+train_ESTER1+train_EPAC+train_ETAPE+train_REPERE_wav2vec2_char_k2/10/metric_test_ETAPE/wer_HLG_1best.txt
slug=${dset}_$(dirname $model)
cp $sb_out ./scoring_files/${slug}_SB_OUT.txt

# RICH ANNOTATION (not sutable for POS tagging, but nice for ASR eval)
stm=/data/pchampio/$dset/test/

# stm=/data/pchampio/REPERE/test2014/

# ASR eval on POS tag:
ub_wer_pos="XFAMIL,PROPN"
# ub_wer_pos="MILITARY"


slug_pos=.$dset
slug_asr=.$dset

# =-----=

if ! $tsv_h; then
  refs_only=false

  if [[ -f ./scoring_files/${slug_pos}_pos-wer.txt ]]; then
    refs_only=true
  fi

  echo "Parse sb summary wer output file.."
  python3 parse-sb-sclite-pos.py \
    --sclite "$sb_out" \
    --refs ./scoring_files/${slug_pos}_pos-wer.txt \
    --hyps ./scoring_files/${slug}_h.txt \
    --hyps-only $refs_only

  # refs_only=false
  if ! $refs_only; then
    echo "normalize.."
    python3 normalize.py \
      --stm "$stm/" \
      --stm-out "./scoring_files/${slug_asr}_asr-wer.txt.norm" \
      --sb-pos-refs "./scoring_files/${slug_pos}_pos-wer.txt"
  fi
fi

sort -k1 -o "./scoring_files/${slug_pos}_pos-wer.txt.norm"{,}
sort -k1 -o "./scoring_files/${slug_asr}_asr-wer.txt.norm"{,}

echo "UB-WER scoring.."
python3 UB-WER.py \
  --refs-pos "./scoring_files/${slug_pos}_pos-wer.txt.norm" \
  --refs-asr "./scoring_files/${slug_asr}_asr-wer.txt.norm" \
  --list-words "./militaire_word_list.txt|MILITARY" \
  --hyps "./scoring_files/${slug}_h.txt" \
  --pos-type $ub_wer_pos | tee ./scoring_files/${slug}.wer

echo "----"
head -n 1 $sb_out

\rm "./scoring_files/${slug}_h.txt"
\rm "./scoring_files/${slug}_h.txt.norm"
\rm "./scoring_files/${slug}_h.txt.tmp"
\rm "./scoring_files/${slug}_h.txt.ctm"
\rm ./scoring_files/.*
