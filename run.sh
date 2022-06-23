#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
export EXPERIMENT=$2
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export OUTPATH=./outputs/$EXPERIMENT/$TIME
export VERSION=$(git rev-parse HEAD)

# Save the experiment detail and dir to the common log file
mkdir -p $OUTPATH

LOG="$OUTPATH/$TIME.txt"
echo Logging output to "$LOG"
# put the arguments on the first line for easy resume
echo -e "$3" >> $LOG
echo $(pwd) >> $LOG
echo "Version: " $VERSION >> $LOG
echo "Git diff" >> $LOG
echo "" >> $LOG
git diff | tee -a $LOG
echo "" >> $LOG
nvidia-smi | tee -a $LOG
echo -e "python main.py --log_dir $OUTPATH $3" >> $LOG

time python -W ignore main.py \
    --log_dir $OUTPATH \
    $3 2>&1 | tee -a "$LOG"

time python -W ignore main.py \
    --is_train False \
    --log_dir $OUTPATH \
    --weights $OUTPATH/weights.pth \
    $3 2>&1 | tee -a "$LOG"
