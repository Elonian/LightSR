#!/usr/bin/env bash
# set -euo pipefail
set -x  # Echo commands for debugging

### ─── MODIFY THESE ─────────────────────────────────────────
TRAIN_HR="/mntdata/main/light_sr/sr/datasets/df2kdata/versions/1/DF2K_train_HR"
TRAIN_LR="/mntdata/main/light_sr/sr/datasets/df2kdata/versions/1/DF2K_train_LR_bicubic"
VAL_HR="/mntdata/main/light_sr/sr/datasets/df2kdata/versions/1/DF2K_valid_HR"
VAL_LR="/mntdata/main/light_sr/sr/datasets/df2kdata/versions/1/DF2K_valid_LR_bicubic/X2"
SAVE_DIR="/mntdata/main/light_sr/sr/results/DF2K/2x/results"
GPUS=1
BATCH_SIZE=2           # Set per-GPU batch size (global = BATCH_SIZE × GPUS)
EPOCHS=200
PATCH_SIZE=96
LEARNING_RATE=1e-4
TEST_EVERY=5
LOG_EVERY=50
SCALE=2
CHANNELS=3
FP=32  # Use 16 for mixed-precision, or 32 for full precision
LOSS="L1Loss"
DEVICES="0"  # GPU IDs
### ─────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="$DEVICES"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Optional: set fork start method on Linux
# if [[ "$(uname)" == "Linux" ]]; then
#   python - <<EOF
# import torch.multiprocessing as mp
# mp.set_start_method('fork', force=True)
# EOF
# fi

# Launch training via torchrun
torchrun --nproc_per_node=$GPUS \
  /mntdata/main/light_sr/sr/trainer/train.py \
  --train_HR_folder "$TRAIN_HR" \
  --train_LR_folder "$TRAIN_LR" \
  --val_HR_folder "$VAL_HR" \
  --val_LR_folder "$VAL_LR" \
  --save_dir "$SAVE_DIR" \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --patch_size $PATCH_SIZE \
  --lr $LEARNING_RATE \
  --scale $SCALE \
  --channels $CHANNELS \
  --fp $FP \
  --loss $LOSS \
  --test_every $TEST_EVERY \
  --log_every $LOG_EVERY \
  --gpu_ids $DEVICES \
