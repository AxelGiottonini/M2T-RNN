#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:lmol
#SBATCH --time=05:00:00

lr=$1   #Learning rate
bs=$2   #Global batch size
mr=$3   #Mask rate
hs=$4   #Hidden size
ls=$5   #Latent size
nl=$6   #Num layers

CUDA_VISIBLE_DEVICES=0 python ./train.py \
    --model_name "M2T-BVR" \
    --model_version "${lr}_${bs}_${mr}_${hs}_${ls}_${nl}" \
    --from_pretrained_bert "Rostlab/prot_bert_bfd" \
    --hidden_size $hs \
    --num_layers $nl \
    --dropout 0.0 \
    --latent_size $ls \
    --training_set "./data/_train.csv" \
    --validation_set "./data/_val.csv" \
    --max_length 512 \
    --mask \
    --mask_rate $mr \
    --split \
    --n_epochs 20 \
    --global_batch_size $bs \
    --local_batch_size 64 \
    --mode "vae" \
    --f_kl 0.2 \
    --learning_rate $lr
