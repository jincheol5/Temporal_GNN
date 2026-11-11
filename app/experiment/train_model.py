import argparse
import mlflow
from torch.utils.data import DataLoader
from model import *
from model_train import *
from utils import *
from .config_info import *

def main(**kwargs):
    """
    0. Init Setting
    """
    ### unpack kwargs
    model_name=kwargs["model_name"]
    dataset_name=kwargs["dataset_name"]
    directed=kwargs["directed"]
    one_hot=kwargs["one_hot"]
    seed=kwargs["seed"]
    lr=kwargs["lr"]
    use_mlflow=kwargs["use_mlflow"]
    experiment_name=kwargs["experiment_name"]
    save_model=kwargs["save_model"]

    ### set seed
    TrainUtils.set_seed(seed=seed)

    """
    1. Data
    """
    ### load data
    data=DataUtils.preprocess_dataset(
        dataset_name=dataset_name,
        directed=directed
    )
    graph_df=data["graph_df"]
    bipartite=data["bipartite"]
    node_ft=data["node_ft"]
    edge_ft=data["edge_ft"]
    n_node=int(max(graph_df["u"].max(),graph_df["i"].max()))
    node_dim=(
        node_ft.shape[1]
        if node_ft is not None
        else n_node if one_hot else 172
    )
    edge_dim=edge_ft.shape[1] if edge_ft is not None else 172
    dataset_config={
        "dataset_name":dataset_name,
        "bipartite":bipartite,
        "directed":directed,
        "one_hot":one_hot
    }

    ### set DataLoader & Sampler
    batch_size=TRAIN_CONFIG["batch_size"]
    train_df,val_df,_ =TrainUtils.split_graph_df(graph_df=graph_df)

    # DataLoader
    train_loader=DataLoader(
        dataset=TemporalGraphDataset(graph_df=train_df),
        batch_size=batch_size,
        shuffle=False
    )
    val_loader=DataLoader(
        dataset=TemporalGraphDataset(graph_df=val_df),
        batch_size=batch_size,
        shuffle=False
    )

    # Sampler
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
    2. Model
    """
    ### set model
    common_model_config={
        "model_name":model_name,
        "node_dim":node_dim,
        "edge_dim":edge_dim,
        **COMMON_MODEL_CONFIG,
    }
    specific_model_config=SPECIFIC_MODEL_CONFIG[model_name]
    model_config=common_model_config|specific_model_config

    # 모델 생성자에 필요한 파라미터만 추출
    init_param_config={
        key:model_config[key]
        for key in MODEL_PARAM[model_name]
    }
    model=MODEL_REGISTRY[model_name](**init_param_config)

    # set model graph
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
        **specific_model_config,
    )
    model.set_graph(graph=graph)

    """
    3. Train
    """
    train_config={
        **TRAIN_CONFIG,
        "seed":seed,
        "lr":lr
    }
    model_trainer=TRAINER_REGISTRY[model_name]

    if use_mlflow:
        ### set mlflow 
        MLFlowUtils.setup(experiment_name=experiment_name)
        run_name=f"{model_name}_{dataset_name}_B{batch_size}_S{seed}_LR{lr}"
        with mlflow.start_run(run_name=run_name) as run:
            # log hyperparameters
            mlflow.log_params(dataset_config|model_config|train_config)

            ### train model
            train_config["use_mlflow"]=use_mlflow
            model=model_trainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                train_sampler=train_sampler,
                val_sampler=val_sampler,
                **train_config,
            )

            ### save model
            if save_model:
                mlflow.pytorch.log_model(pytorch_model=model,name="model")

            # mlflow run_id
            run_id=run.info.run_id
        print(f"Training completed: {run_name}")
        print(f"Run ID: {run_id}")
    else:
        ### train model
        train_config["use_mlflow"]=use_mlflow
        model=model_trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            **train_config,
        )
        print(f"Training completed!")

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",
        type=str,
        choices=list(MODEL_REGISTRY),
        default=f"TGAT"
    )
    parser.add_argument("--seed",type=int,choices=[1,2,3],default=1)
    parser.add_argument("--lr",type=float,choices=[0.0001,0.0005],default=0.0005)
    parser.add_argument("--dataset_name",
        type=str,
        choices=DATASET_REGISTRY,
        default="enron"
    )
    parser.add_argument("--directed",type=int,default=0)
    parser.add_argument("--one_hot",type=int,default=0)
    parser.add_argument("--use_mlflow",type=int,default=0)
    parser.add_argument("--experiment_name",type=str,default=f"Temporal_GNN")
    parser.add_argument("--save_model",type=int,default=0)
    args=parser.parse_args()
    main(**vars(args))