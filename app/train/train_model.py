import argparse
from torch.utils.data import DataLoader
from model import TGAT,TGN
from model_train import Model_Trainer,TGN_Trainer
from utils import DataUtils,TrainUtils,TemporalGraphDataset

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
def main(**kwargs):
    ### set seed
    seed=kwargs["seed"]
    TrainUtils.set_seed(seed=seed)

    """
    1. Setting Data
    """
    ### load data
    dataset_name=kwargs["dataset_name"]
    directed=kwargs["directed"]
    one_hot=kwargs["one_hot"] 
    data=DataUtils.preprocess_dataset(
        dataset_name=dataset_name,
        directed=False
    )
    graph_df=data["graph_df"]
    n_node=int(max(graph_df["u"].max(),graph_df["i"].max()))
    node_ft=data["node_ft"]
    edge_ft=data["edge_ft"]
    bipartite=data["bipartite"]
    
    if node_ft is not None:
        node_dim=node_ft.shape[1]
    else:
        node_dim=n_node if one_hot else 172 
    if edge_ft is not None:
        edge_dim=edge_ft.shape[1]
    else:
        edge_dim=172

    ### set train_loader and val_loader
    batch_size=200
    train_df,val_df,_=TrainUtils.split_graph_df(graph_df=graph_df)
    train_dataset=TemporalGraphDataset(graph_df=train_df)
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
    val_dataset=TemporalGraphDataset(graph_df=val_df)
    val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)

    ### set train, val sampler
    train_sampler=TrainUtils.set_sampler(
        graph_df=graph_df,
        n_node=n_node,
        bipartite=bipartite,
        purpose=f"train"
    )
    val_sampler=TrainUtils.set_sampler(
        graph_df=graph_df,
        n_node=n_node,
        bipartite=bipartite,
        purpose=f"val"
    )

    """
    2. Setting Model
    """
    ### 모델 관련 파라미터
    model_name=kwargs["model_name"]
    time_dim=100
    latent_dim=172
    embed_dim=172
    n_layer=2
    n_neighbor=10
    n_head=2

    ### set model and graph_config
    graph_config={}
    match model_name:
        case "TGAT":
            model=TGAT(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head
            )
        case "TGN":
            # TGN 하이퍼 파라미터
            msg_dim=100
            mem_dim=172
            aggr_fn=f"last"
            msg_fn=f"mlp"
            model=TGN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                msg_dim=msg_dim,
                mem_dim=mem_dim,
                embed_dim=embed_dim,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head,
                aggr_fn=aggr_fn,
                msg_fn=msg_fn
            )

    ### set graph
    graph=TrainUtils.set_model_graph(
        model_name=model_name,
        graph_df=graph_df,
        directed=directed,
        bipartite=bipartite,
        one_hot=one_hot,
        node_ft=node_ft,
        edge_ft=edge_ft,
        node_dim=node_dim,
        edge_dim=edge_dim,
        **graph_config
    )
    model.set_graph(graph=graph)

    """
    3. Train Model
    """
    ### 학습 관련 파라미터
    optimizer=f"adam"
    epoch=100
    early_stop=True
    patience=10
    batch_size=200
    lr=kwargs["lr"]
    model_config={
        "model_name":kwargs["model_name"],
        "optimizer":optimizer,
        "epoch":epoch,
        "early_stop":early_stop,
        "patience":patience,
        "batch_size":batch_size,
        "seed":seed,
        "lr":lr
    }
    match model_name:
        case "TGAT":
            pass
        case "TGN":
            model=TGN_Trainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                train_sampler=train_sampler,
                val_sampler=val_sampler,
                **model_config
            )

    ### save model



if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",
        type=str,
        choices=[
            "TGAT",
            "TGN",
            "CAW-N",
            "GraphMixer",
            "DyGFormer",
            "TPNet"
        ],
        default=f"TGAT"
    )
    parser.add_argument("--seed",type=int,choices=[1,2,3],default=1)
    parser.add_argument("--lr",type=float,choices=[0.0001,0.0005],default=0.0005)
    parser.add_argument("--dataset_name",
        type=str,
        choices=[
            "enron",
            "wikipedia",
            "reddit",
            "CollegeMsg",
            "bitcoin-alpha",
            "bitcoin-otc"
        ],
        default="enron"
    )
    parser.add_argument("--directed",type=int,default=0)
    parser.add_argument("--one_hot",type=int,default=0)
    parser.add_argument("--save_model",type=int,default=0)
    args=parser.parse_args()
    app_config={
        "model_name":args.model_name,
        "seed":args.seed,
        "lr":args.lr,
        "dataset_name":args.dataset_name,
        "directed":args.directed,
        "one_hot":args.one_hot,
        "save_model":args.save_model
    }
    main(**app_config)
