import argparse
import mlflow
from torch.utils.data import DataLoader
from model import TGAT,TGN,CAW_N,GraphMixer,DyGFormer,TPNet
from model_train import Model_Trainer,TGN_Trainer,TPNet_Trainer
from utils import DataUtils,TrainUtils,TemporalGraphDataset,MLFlowUtils

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
    ### unpack kwargs
    model_name=kwargs["model_name"]
    dataset_name=kwargs["dataset_name"]
    directed=kwargs["directed"]
    one_hot=kwargs["one_hot"]
    seed=kwargs["seed"]
    lr=kwargs["lr"]
    save_model=kwargs["save_model"]

    """
    0. Set seed
    """
    ### set seed
    TrainUtils.set_seed(seed=seed)

    """
    1. Set Data
    """
    ### load data
    data=DataUtils.preprocess_dataset(
        dataset_name=dataset_name,
        directed=directed
    )
    graph_df=data["graph_df"]
    n_node=int(max(graph_df["u"].max(),graph_df["i"].max()))
    node_ft=data["node_ft"]
    edge_ft=data["edge_ft"]
    bipartite=data["bipartite"]

    ### set node_dim & edge_dim
    if node_ft is not None:
        node_dim=node_ft.shape[1]
    else:
        node_dim=n_node if one_hot else 172 
    if edge_ft is not None:
        edge_dim=edge_ft.shape[1]
    else:
        edge_dim=172

    ### set dataset_config
    dataset_config={
        "dataset_name":dataset_name,
        "bipartite":bipartite,
        "directed":directed,
        "one_hot":one_hot
    }

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
    2. Set Model
    """
    ### 모델 관련 파라미터
    time_dim=100
    latent_dim=172
    embed_dim=172
    n_layer=2
    n_neighbor=10
    n_head=2

    ### set common_model_config
    common_model_config={
        "model_name":model_name,
        "node_dim":node_dim,
        "edge_dim":edge_dim,
        "time_dim":time_dim,
        "latent_dim":latent_dim,
        "embed_dim":embed_dim,
        "n_layer":n_layer,
        "n_neighbor":n_neighbor,
        "n_head":n_head
    }

    ### set model
    model_config={}
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
            model_config["msg_dim"]=msg_dim
            model_config["mem_dim"]=mem_dim
            model_config["aggr_fn"]=aggr_fn
            model_config["msg_fn"]=msg_fn
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
        case "CAW-N":
            # CAW-N 하이퍼 파라미터
            n_walk=64
            walk_len=2
            alpha=0.000001
            model_config["n_walk"]=n_walk
            model_config["walk_len"]=walk_len
            model_config["alpha"]=alpha
            model=CAW_N(
                node_dim=node_dim,
                edge_dim=edge_dim,
                latent_dim=latent_dim,
                time_dim=time_dim,
                embed_dim=embed_dim,
                n_head=n_head,
                n_walk=n_walk,
                walk_len=walk_len,
                alpha=alpha
            )
        case "GraphMixer":
            # GraphMixer 하이퍼 파라미터
            n_last_event=50
            model_config["n_last_event"]=n_last_event
            model=GraphMixer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                n_last_event=n_last_event
            )
        case "DyGFormer":
            # DyGFormer 하이퍼 파라미터 
            co_dim=50
            common_dim=50
            max_history_len=64
            patch_size=2
            model_config["co_dim"]=co_dim
            model_config["common_dim"]=common_dim
            model_config["max_history_len"]=max_history_len
            model_config["patch_size"]=patch_size
            model=DyGFormer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                latent_dim=latent_dim,
                time_dim=time_dim,
                co_dim=co_dim,
                common_dim=common_dim,
                embed_dim=embed_dim,
                n_layer=n_layer,
                n_head=n_head,
                max_history_len=max_history_len,
                patch_size=patch_size
            )
        case "TPNet":
            # TPNet 하이퍼 파라미터
            random_ft_dim=128
            walk_len=3
            time_decay_coef=0.0000001
            n_last_event=20
            mlp_mixer_n_layer=2
            model_config["random_ft_dim"]=random_ft_dim
            model_config["walk_len"]=walk_len
            model_config["time_decay_coef"]=time_decay_coef
            model_config["n_last_event"]=n_last_event
            model_config["mlp_mixer_n_layer"]=mlp_mixer_n_layer
            model=TPNet(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                random_ft_dim=random_ft_dim,
                embed_dim=embed_dim,
                walk_len=walk_len,
                time_decay_coef=time_decay_coef,
                n_last_event=n_last_event,
                mlp_mixer_n_layer=mlp_mixer_n_layer
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
        **model_config
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
    lr=kwargs["lr"]

    ### set train_config
    train_config={
        "optimizer":optimizer,
        "epoch":epoch,
        "early_stop":early_stop,
        "patience":patience,
        "batch_size":batch_size,
        "seed":seed,
        "lr":lr
    }

    MLFlowUtils.setup(experiment_name="Temporal_GNN")
    run_name=f"{model_name}_{dataset_name}_B{batch_size}_S{seed}_LR{lr}"
    with mlflow.start_run(run_name=run_name):
        # Hyperparameter 기록
        param=dataset_config|common_model_config|model_config|train_config
        mlflow.log_params(param)
        # train model
        match model_name:
            case "TGAT"|"CAW-N"|"GraphMixer"|"DyGFormer":
                model=Model_Trainer.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    train_sampler=train_sampler,
                    val_sampler=val_sampler,
                    **train_config
                )
            case "TGN":
                model=TGN_Trainer.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    train_sampler=train_sampler,
                    val_sampler=val_sampler,
                    **train_config
                )
            case "TPNet":
                model=TPNet_Trainer.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    train_sampler=train_sampler,
                    val_sampler=val_sampler,
                    **train_config
                )
        # save model
        if save_model:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name="model"
            )

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
            "soc-sign-bitcoinalpha",
            "soc-sign-bitcoinotc"
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
