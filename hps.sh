#!/bin/bash

for lr in "0.0001" "0.0004" "0.001" "0.004"; do 
for bs in 64 128 256 512; do
for mr in 0.05 0.10 0.15; do
for hs in 1024 512 256; do
for ls in 256 128 64; do
for nl in 2 4; do

sbatch ./train.sh $lr $bs $mr $hs $ls $nl

done
done
done
done
done
done