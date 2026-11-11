#!/bin/bash

python -m app.experiment.train_model --model_name TGAT --seed 1 --lr 0.0001 --dataset_name wikipedia

python -m app.experiment.train_model --model_name TGN --seed 1 --lr 0.0001 --dataset_name wikipedia

python -m app.experiment.train_model --model_name CAW-N --seed 1 --lr 0.0001 --dataset_name wikipedia

python -m app.experiment.train_model --model_name GraphMixer --seed 1 --lr 0.0001 --dataset_name wikipedia --one_hot 1

python -m app.experiment.train_model --model_name DyGFormer --seed 1 --lr 0.0001 --dataset_name wikipedia

python -m app.experiment.train_model --model_name TPNet --seed 1 --lr 0.0001 --dataset_name wikipedia