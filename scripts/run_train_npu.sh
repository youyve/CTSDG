if [ $# != 7 ]
then
    echo "Usage: bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [CFG_PATH] [SAVE_PATH] [VGG_PRETRAIN] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}


DEVICE_ID=$1
CFG_PATH=$(get_real_path $2)
SAVE_PATH=$(get_real_path $3)
VGG_PRETRAIN=$(get_real_path $4)
IMAGES_PATH=$(get_real_path $5)
MASKS_PATH=$(get_real_path $6)
ANNO_PATH=$(get_real_path $7)

export DEVICE_ID="$DEVICE_ID"

if [ -d "$SAVE_PATH" ];
then
    rm -rf "$SAVE_PATH"
fi
mkdir -p "$SAVE_PATH"

cp "$CFG_PATH" "$SAVE_PATH"

python train.py \
  --is_distributed=0 \
  --device_target=GPU \
  --gen_lr_train=0.0002 \
  --gen_lr_finetune=0.00005 \
  --train_iter=350000 \
  --finetune_iter=150000 \
  --save_checkpoint_steps=10000 \
  --log_frequency_step=1000 \
  --config_path="$CFG_PATH" \
  --pretrained_vgg="$VGG_PRETRAIN" \
  --data_root="$IMAGES_PATH" \
  --train_masks_root="$MASKS_PATH" \
  --anno_path="$ANNO_PATH" \
  --save_path="$SAVE_PATH" > "$SAVE_PATH"/log.txt 2>&1 &