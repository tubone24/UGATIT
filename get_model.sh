#!/bin/bash
FILE_ID=19xQK2onIy-3S5W5K-XIh85pAg_RNvBVf
FILE_NAME=100_epoch_selfie2anime_checkpoint.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}

tar xvf $FILE_NAME
# unzip $FILE_NAME
