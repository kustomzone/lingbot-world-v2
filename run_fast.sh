#!/bin/bash

set -x

WEIGHT_DIR=${1:-lingbot-world-v2-14b-causal-fast}
FRAME=${2:-361}

# causal_fast (default) — distilled few-step model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; torchrun --nproc_per_node=8 generate.py \
                           --task i2v-A14B \
                           --size 480*832 \
                           --ckpt_dir ${WEIGHT_DIR} \
                           --image examples/03/image.jpg \
                           --action_path examples/03 \
                           --dit_fsdp \
                           --t5_fsdp \
                           --ulysses_size 8 \
                           --frame_num ${FRAME} \
                           --prompt "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples reflect the tree and sky, creating a tranquil, meditative atmosphere." \
                           --local_attn_size 18 \
                           --sink_size 6
