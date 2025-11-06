import os
import random
import numpy as np
import argparse
import wandb
import torch
from tqdm import tqdm
from tgnn import DataUtils,ModelTrainer,ModelTrainUtils,TGAT,TGN

def app_train(config: dict):
    """
    seed setting
    """
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed']) 
    os.environ["PYTHONHASHSEED"]=str(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic=True 
    torch.backends.cudnn.benchmark=False

    match config['app_num']:
        case 1:
            """
            App 1.
            train 
            """
            if config['wandb']:
                if config['processor']=='mpnn':
                    wandb.init(project="TGNN",name=f"{config['model']}_{config['aggr']}_{config['seed']}_{config['lr']}_{config['batch_size']}")
                else:
                    wandb.init(project="TGNN",name=f"{config['model']}_{config['processor']}_{config['seed']}_{config['lr']}_{config['batch_size']}")
                wandb.config.update(config)

            """
            Data load and preprocess
                graph_list_dict:
                    graph_list_dict['ladder']=ladder_graph_list
                    graph_list_dict['grid']=grid_graph_list
                    graph_list_dict['tree']=tree_graph_list
                    graph_list_dict['erdos_renyi']=Erdos_Renyi_graph_list
                    graph_list_dict['barabasi_albert']=Barabasi_Albert_graph_list
                    graph_list_dict['community']=community_graph_list
                    graph_list_dict['caveman']=caveman_graph_list
            """
            train_20=DataUtils.DataLoader.load_from_pickle("train_20","graph")
            val_20=DataUtils.DataLoader.load_from_pickle("val_20","graph")

            train_graph_list=[]
            for _,graph_list in tqdm(train_20.items()):
                train_graph_list+=graph_list
            train_batch_loader_list=ModelTrainUtils.get_batch_loader_list(graph_list=train_graph_list,random_src=config['random_src'],batch_size=config['batch_size'])

            print(f"preprocess train finish")

            val_graph_list=[]
            for _,graph_list in tqdm(val_20.items()):
                val_graph_list+=graph_list
            val_batch_loader_list=ModelTrainUtils.get_batch_loader_list(graph_list=val_graph_list,random_src=config['random_src'],batch_size=config['batch_size'])

            print(f"preprocess val finish")

            """
            model setting and training
            """
            match config['model']:
                case 'tgat':
                    model=TGAT(node_dim=1,latent_dim=config['latent_dim'])
                case 'tgn':
                    model=TGN(node_dim=1,latent_dim=config['latent_dim'],emb=config['emb'])
            ModelTrainer.train(model=model,train_batch_loader_list=train_batch_loader_list,val_batch_loader_list=val_batch_loader_list,validate=True,config=config)

            """
            save model
            """
            if config['save_model']:
                match config['model']:
                    case 'tgat':
                        model_name=f"tgat_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    case 'tgn':
                        model_name=f"tgn_{config['emb']}_{config['seed']}_{config['lr']}_{config['batch_size']}"
            DataUtils.DataLoader.save_model_parameter(model=model,model_name=model_name)

if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    
    # setting
    parser.add_argument("--model",type=str,default='tgat') # tgat,tgn
    parser.add_argument("--emb",type=str,default='attn') # time, attn, sum
    parser.add_argument("--random_src",type=int,default=1)

    # train
    parser.add_argument("--optimizer",type=str,default='adam') # adam, sgd
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--early_stop",type=int,default=1)
    parser.add_argument("--patience",type=int,default=10)
    parser.add_argument("--seed",type=int,default=1) # 1, 2, 3
    parser.add_argument("--lr",type=float,default=0.0005) # 0.001, 0,0005
    parser.add_argument("--batch_size",type=int,default=32) # 8, 16
    parser.add_argument("--latent_dim",type=int,default=32)
    
    # 학습 로그 및 저장
    parser.add_argument("--wandb",type=int,default=0)
    parser.add_argument("--save_model",type=int,default=0)

    # 평가
    parser.add_argument("--test_num_nodes",type=int,default=20)
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        # setting
        'model':args.model,
        'emb':args.emb,
        'random_src':args.random_src,
        # train
        'optimizer':args.optimizer,
        'epochs':args.epochs,
        'early_stop':args.early_stop,
        'patience':args.patience,
        'seed':args.seed,
        'lr':args.lr,
        'batch_size':args.batch_size,
        'latent_dim':args.latent_dim,
        # 학습 로그 및 저장
        'wandb':args.wandb,
        'save_model':args.save_model,
        # 평가
        'test_num_nodes':args.test_num_nodes
    }
    app_train(config=config)