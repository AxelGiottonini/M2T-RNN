#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:lmol
#SBATCH --time=20:00:00

CUDA_VISIBLE_DEVICES=0 python ./train.py \
    --model_name "M2T-BVR" \
    --model_version "0.1" \
    --from_pretrained_bert "Rostlab/prot_bert_bfd" \
    --hidden_size 1024 \
    --num_layers 5 \
    --dropout 0.0 \
    --latent_size 256 \
    --training_set "./data/_train.csv" \
    --validation_set "./data/_val.csv" \
    --max_length 512 \
    --mask \
    --mask_rate 0.15 \
    --split \
    --n_epochs 1 \
    --global_batch_size 512 \
    --local_batch_size 64 \
    --mode "vae" \
    --f_kl 1 
