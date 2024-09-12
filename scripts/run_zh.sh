. ~/script_utils/utils.sh

export ENTRY_FILE=train.py
export ACCELERATE_MIXED_PRECISION=fp16
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export USE_XFORMERS=t

export ENTRY_FILE=train.py
export PROJ_NAME="UViT"
export ODIR_NAME="output_dir"

export GIT=f
export NGPUS=8

run \
--eval_freq 4 \
--trajectories_per_upd 4096 \
--lr=1e-5 --K_epochs=5 --D_epochs=5 \
--gen_steps=8 --action_std=0.6 --heu=0 \
--data_root assets/imagenet-tsv

