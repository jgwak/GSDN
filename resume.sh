#!/bin/bash
# Usage
#
# ./resume.sh GPU_ID LOG_FILE "--argument 1 --argument 2"

export PYTHONUNBUFFERED="True"

export CUDA_VISIBLE_DEVICES=$1
export LOG=$2
# $3 is reserved for the arguments
export OUTPATH=$(dirname "${LOG}")

set -e

echo "" >> $LOG
echo "Resume training" >> $LOG
echo "" >> $LOG
nvidia-smi | tee -a $LOG

time python main.py \
    --log_dir $OUTPATH \
    --resume $OUTPATH/weights.pth \
    $3 2>&1 | tee -a "$LOG"

time python main.py \
    --is_train False \
    --log_dir $OUTPATH \
    --weights $OUTPATH/weights.pth \
    $3 2>&1 | tee -a "$LOG"
