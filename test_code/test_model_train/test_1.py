import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset
from data import TemporalGraph,Memory
from model import TGAT_Link_Prediction,TGN_Link_Prediction
from model_train import ModelTrainer

"""
<< Test >> 
model_train.model_train.ModelTrainer
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. ModelTrainer.train_link_prediction
                model: TGAT
                task: link prediction
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"CollegeMsg")
            graph=TemporalGraph(df=df,node_dim=4)

            train_df,val_df,_=TrainUtils.split_graph_df(df=df)
            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            train_loader=DataLoader(dataset=train_dataset,batch_size=100,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=100,shuffle=False)
            
            model=TGAT_Link_Prediction(
                node_dim=4,
                latent_dim=4,
                time_dim=4,
                output_dim=4,
                graph=graph,
                n_layer=2,
                n_neighbor=5,
                n_head=4
            )
            config={
                "optimizer":kwargs["optimizer"],
                "lr":kwargs["lr"],
                "epoch":kwargs["epoch"]
            }
            ModelTrainer.train_link_prediction(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                **config
            )
            print(f"Model Training END!")

        case 2:
            """
            Test. ModelTrainer.train_link_prediction
                model: TGN
                task: link prediction
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"CollegeMsg")
            graph=TemporalGraph(df=df,node_dim=4)
            n_node=graph.get_num_node()
            memory=Memory(n_node=n_node,mem_dim=4)

            train_df,val_df,_=TrainUtils.split_graph_df(df=df)
            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            train_loader=DataLoader(dataset=train_dataset,batch_size=100,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=100,shuffle=False)

            model=TGN_Link_Prediction(
                node_dim=4,
                mem_dim=4,
                latent_dim=4,
                msg_dim=4,
                time_dim=4,
                output_dim=4,
                graph=graph,
                memory=memory,
                n_layer=2,
                n_neighbor=5,
                n_head=4,
                msg_fn="concat",
                aggr_fn="mean"
            )
            config={
                "optimizer":kwargs["optimizer"],
                "lr":kwargs["lr"],
                "epoch":kwargs["epoch"]
            }
            ModelTrainer.train_link_prediction(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                **config
            )
            print(f"Model Training END!")

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    parser.add_argument("--optimizer",type=str,default=f"adam")
    parser.add_argument("--lr",type=float,default=0.0005)
    parser.add_argument("--epoch",type=int,default=1)

    args=parser.parse_args()
    test_config={
        'test_num':args.test_num,
        'optimizer':args.optimizer,
        'lr':args.lr,
        'epoch':args.epoch
    }
    test_fn(**test_config)