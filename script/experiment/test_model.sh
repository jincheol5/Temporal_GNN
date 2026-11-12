#!/bin/bash

DATASETS=(wikipedia)
SEEDS=(1 2 3)
LRS=(0.0001 0.0005)
MODELS=(TGAT TGN CAW-N GraphMixer DyGFormer TPNet)

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for lr in "${LRS[@]}"; do
            for model in "${MODELS[@]}"; do
                echo "Test: dataset=$dataset, seed=$seed, lr=$lr, model=$model"

                if [ "$model" == "GraphMixer" ]; then
                    python -m app.experiment.test_model --model_name "$model" --seed "$seed" --lr "$lr" --one_hot 1 --dataset_name "$dataset" --experiment_name Temporal_GNN --save_metric 1
                else
                    python -m app.experiment.test_model --model_name "$model" --seed "$seed" --lr "$lr" --dataset_name "$dataset" --experiment_name Temporal_GNN --save_metric 1
                fi
            done
        done
    done
done