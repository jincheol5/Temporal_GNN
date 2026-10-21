import argparse
from torch.utils.data import DataLoader
from model import TGAT_LP,TGN_LP,DyGFormer_LP
from model_train import ModelTrainer
from utils import DataUtils,TrainUtils,TemporalGraphDataset,Sampler

def main(**kwargs):
    ### 모델 관련 파라미터
    model_name=kwargs["model_name"]
    time_dim=32
    latent_dim=128
    embed_dim=32
    n_layer=1
    n_neighbor=10
    n_head=3

    ### 학습 관련 파라미터
    seed=kwargs["seed"]
    optimizer=f"adam"
    epoch=100
    early_stop=True
    patience=10
    batch_size=200
    lr=kwargs["lr"]

    ### set seed
    TrainUtils.set_seed(seed=seed)

    ### load data
    dataset_name=kwargs["dataset_name"]
    data=DataUtils.preprocess_dataset(dataset_name=dataset_name)
    graph_df=data["graph_df"]
    bipartite=data["bipartite"]
    node_ft=data["node_ft"]
    edge_ft=data["edge_ft"]
    node_dim=data["node_dim"]
    if node_dim==0: 
        node_dim=32
    edge_dim=data["edge_dim"]
    if edge_dim==0:
        edge_dim=32

    ### set model
    match model_name:
        case "TGAT":
            model=TGAT_LP(
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
            ### TGN 하이퍼 파라미터
            msg_dim=32
            mem_dim=32
            msg_fn=f"mlp"
            aggr_fn=f"mean"
            model=TGN_LP(
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
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )
        case "DyGFormer":
            ### DyGFormer 하이퍼 파라미터
            co_dim=32
            common_dim=32
            max_history_len=10
            patch_size=5
            model=DyGFormer_LP(
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

    ### set graph
    graph=TrainUtils.generate_model_graph(
        model_name=model_name,
        graph_df=graph_df,
        node_ft=node_ft,
        edge_ft=edge_ft,
        node_dim=node_dim,
        edge_dim=edge_dim
    )
    model.set_graph(graph=graph)

    ### set train, val data_loader 
    train_df,val_df,_=TrainUtils.split_graph_df(graph_df=graph_df)
    train_dataset=TemporalGraphDataset(graph_df=train_df)
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
    val_dataset=TemporalGraphDataset(graph_df=val_df)
    val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)

    ### set train_sampler, val_sampler
    train_sampler=TrainUtils.set_sampler(dataset=train_dataset)
    val_sampler=TrainUtils.set_sampler(dataset=val_dataset,seed=seed)

    ### set model_config
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

    ### train model
    match model_name:
        case "TGAT"|"DyGFormer":
            model=ModelTrainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                train_sampler=train_sampler,
                val_sampler=val_sampler,
                **model_config
            )

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",
        type=str,
        choices=["TGAT","TGN","DyGFormer"],
        default=f"TGAT"
    )
    parser.add_argument("--seed",type=int,choices=[1,2,3],default=1)
    parser.add_argument("--lr",type=float,choices=[0.0005,0.0001],default=0.0005)
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
    parser.add_argument("--save_model",type=int,default=0)
    args=parser.parse_args()
    app_config={
        "model_name":args.model_name,
        "seed":args.seed,
        "lr":args.lr,
        "dataset_name":args.dataset_name,
        "save_model":args.save_model
    }
    main(**app_config)