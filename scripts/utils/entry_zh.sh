#!/bin/bash
. ~/script_utils/utils.sh

###===> Install dependency
python_path=`which python`
echo 'Python at '$python_path `$python_path --version`
pip install loguru GitPython accelerate transformers datasets einops omegaconf opencv-python matplotlib pandas scipy tensorboard wandb torchvision xformers fvcore

export TORCH_DISTRIBUTED_DEBUG=DETAIL
echo "pythonpath="$PYTHONPATH
export ACCELERATE_MIXED_PRECISION=${ACCELERATE_MIXED_PRECISION:-fp16}


inner_entry() {
#  accelerate_args=(--num_processes $((NGPUS*WORLD_SIZE)) --num_machines $WORLD_SIZE --machine_rank $RANK --mixed_precision fp16 --main_process_port $MASTER_PORT --main_process_ip $MASTER_ENDPOINT)
#
#  echo accelerate launch "${accelerate_args[@]}" "$ENTRY_FILE" "${@}"
#  accelerate launch "${accelerate_args[@]}" "$ENTRY_FILE" "${@}"
  img_folder=$HOME/assets
  ln -s ${img_folder} .
  if [ ${IDC} == klara-2-pek02 ]; then
    RUNNER="torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${NGPUS} --node_rank=${RANK} --master_addr=${MASTER_ENDPOINT} --master_port=${MASTER_PORT}"
  else
    RUNNER="torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${NGPUS} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
  fi
  echo mixed_precision is $ACCELERATE_MIXED_PRECISION
  echo $RUNNER "$ENTRY_FILE" "${@}"
  $RUNNER "$ENTRY_FILE" "${@}"
}

inner_run "${@}"
