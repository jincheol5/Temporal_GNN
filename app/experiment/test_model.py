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
    seed=kwargs["seed"]
    lr=kwargs["lr"]
    experiment_name=kwargs["experiment_name"]
    save_metric=kwargs["save_metric"]

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
    n_node=int(max(graph_df["u"].max(),graph_df["i"].max()))

    ### set DataLoader & Sampler
    batch_size=TRAIN_CONFIG["batch_size"]
    _,_,test_df =TrainUtils.split_graph_df(graph_df=graph_df)

    # DataLoader
    test_loader=DataLoader(
        dataset=TemporalGraphDataset(graph_df=test_df),
        batch_size=batch_size,
        shuffle=False
    )

    # Sampler
    test_sampler=TrainUtils.set_sampler(
        graph_df=graph_df,
        n_node=n_node,
        bipartite=bipartite,
        purpose=f"test"
    )
    test_sampler.set_random_state_seed(seed=seed)

    """
    2. Model
    """
    ### load model from mlflow
    # 모델 전체를 저장해두었기 때문에 set_graph 단계 필요 없음.
    MLFlowUtils.setup(experiment_name=experiment_name)
    run_ids=MLFlowUtils.search_run_ids(
        experiment_name=experiment_name,
        model_name=model_name,
        dataset_name=dataset_name,
        seed=seed,
        lr=lr
    )
    run_id=run_ids[0] # 가장 최근 run_id
    model=MLFlowUtils.load_model(run_id=run_id,artifact_name=model_name)

    """
    3. Test Model
    """
    model_trainer=TRAINER_REGISTRY[model_name]
    ### test model
    test_result=model_trainer.test(
        model=model,
        test_loader=test_loader,
        test_sampler=test_sampler,
    )
    print(f"Evaluation Result:")
    print(f"dataset: {dataset_name}")
    print(f"model: {model_name}")
    print(f"hypher parameter: seed={seed}, lr={lr}, batch_size={batch_size}")
    print(f"ACC: {test_result['acc']}")
    print(f"AP: {test_result['ap']}")
    print(f"AUC: {test_result['auc']}")

    if save_metric:
        with mlflow.start_run(run_id=run_id) as run:
            test_metrics={
                "test_acc":test_result["acc"],
                "test_ap":test_result["ap"],
                "test_auc":test_result["auc"]
            }
            mlflow.log_metrics(test_metrics)

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
    parser.add_argument("--experiment_name",type=str,default=f"Temporal_GNN")
    parser.add_argument("--save_metric",type=int,default=0)
    args=parser.parse_args()
    main(**vars(args))
