from model import TGAT,TGN,CAW_N,GraphMixer,DyGFormer,TPNet
from model_train import Model_Trainer,TGN_Trainer,TPNet_Trainer

"""
<< 공통 하이퍼 파라미터 >>
node_dim: 172
edge_dim: 172
time_dim: 100
latent_dim: 172
embed_dim: 172

<< 공통 모델 관련 파라미터 >>
n_layer: 2
n_head: 2
n_neighbor: 10

<< 공통 모델 학습 관련 파라미터 >>
batch_size: 200
epoch (max_epoch): 100
learning_rate: 0.0001
drop_out: 0.1

<< 모델별 하이퍼 파라미터 >>
TGAT 
- neighbor_sampling: last

TGN
- mem_dim: 172
- msg_dim: 100
- msg_fn: mlp
- aggr_fn: last
- memory_updater: GRU

CAW-N
- n_walk: 64
- walk_len: 2
- alpha: 0.000001 (1e^-6)

GraphMixer 
- n_last_event: 50
- node_ft는 one-hot vec로 처리한다.

DyGFormer
- max_history_len: 64
- patch_size: 2
- co_dim: 50
- common_dim: 50

TPNet
- walk_len: 3
- time_decay_coef: 0.0000001 (1e^-7)
- n_last_event: 20
- mlp_mixer_n_layer: 2
- random_ft_dim: 128
"""

DATASET_REGISTRY=[
    "reddit","wikipedia","mooc",
    "lastfm","enron","SocialEvo",
    "uci","CollegeMsg","TaobaoSmall",
    "CanParl","Contacts","Flights",
    "UNtrade","USLegis","UNvote",
    "DGraphFin","TaobaoLarge","YoutubeReddit",
    "YoutubeRedditLarge"
]

MODEL_REGISTRY={
    "TGAT":TGAT,
    "TGN":TGN,
    "CAW-N":CAW_N,
    "GraphMixer":GraphMixer,
    "DyGFormer":DyGFormer,
    "TPNet":TPNet
}

TRAINER_REGISTRY={
    "TGAT":Model_Trainer,
    "TGN":TGN_Trainer,
    "CAW-N":Model_Trainer,
    "GraphMixer":Model_Trainer,
    "DyGFormer":Model_Trainer,
    "TPNet":TPNet_Trainer
}

COMMON_MODEL_CONFIG={
    "time_dim":100,
    "latent_dim":172,
    "embed_dim":172,
    "n_layer":2,
    "n_neighbor":10,
    "n_head":2
}

SPECIFIC_MODEL_CONFIG={
    "TGAT":{},
    "TGN":{
        "msg_dim":100,
        "mem_dim":172,
        "aggr_fn":"last",
        "msg_fn":"mlp",
    },
    "CAW-N":{
        "n_walk":64,
        "walk_len":2,
        "alpha":1e-6,
    },
    "GraphMixer":{
        "n_last_event": 50,
    },
    "DyGFormer":{
        "co_dim":50,
        "common_dim":50,
        "max_history_len":64,
        "patch_size":2,
    },
    "TPNet":{
        "random_ft_dim":128,
        "walk_len":3,
        "time_decay_coef":1e-7,
        "n_last_event":20,
        "mlp_mixer_n_layer":2,
    }
}

# 모델 생성자별 허용 파라미터
MODEL_PARAM={
    "TGAT":(
        "node_dim","edge_dim","time_dim",
        "latent_dim","embed_dim",
        "n_layer","n_neighbor","n_head"
    ),
    "TGN":(
        "node_dim","edge_dim","time_dim",
        "latent_dim","embed_dim",
        "n_layer","n_neighbor","n_head",
        "msg_dim","mem_dim",
        "aggr_fn","msg_fn"
    ),
    "CAW-N":(
        "node_dim","edge_dim","time_dim",
        "latent_dim","embed_dim",
        "n_head",
        "n_walk","walk_len","alpha",
    ),
    "GraphMixer":(
        "node_dim","edge_dim","time_dim",
        "latent_dim","embed_dim", 
        "n_last_event"
    ),
    "DyGFormer":(
        "node_dim","edge_dim","time_dim",
        "latent_dim","embed_dim", 
        "n_layer","n_head",
        "co_dim","common_dim",
        "max_history_len","patch_size"
    ),
    "TPNet":(
        "node_dim","edge_dim","time_dim",
        "latent_dim","embed_dim", 
        "random_ft_dim",
        "walk_len","time_decay_coef","n_last_event","mlp_mixer_n_layer"
    ),
}

TRAIN_CONFIG={
    "optimizer":"adam",
    "epoch":100,
    "early_stop":True,
    "patience":10,
    "batch_size":200
}