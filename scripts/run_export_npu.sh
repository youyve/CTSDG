if [ $# != 3 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_export_npu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CFG_PATH=$(get_real_path $2)
CKPT_PATH=$(get_real_path $3)

export DEVICE_NUM=1
export DEVICE_ID=$1

if [ ! -d "./logs" ]
then
  mkdir "./logs"
fi

echo "Start export for device $DEVICE_ID"

python export.py \
  --checkpoint_path=$CKPT_PATH \
  --device_target='Ascend' \
  --device_num=$DEVICE_NUM \
  --config_path=$CFG_PATH \
  --device_id=$DEVICE_ID > ./logs/export_log.txt 2>&1 &