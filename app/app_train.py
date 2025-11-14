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
        case 0:
            """
            test
            """
            dataset_dict_list=DataUtils.load_from_pickle(file_name=f"val_20_grid",path="tgnn",dir_type="val")
            train_dataset_list=[]
            for dataset_dict in dataset_dict_list:
                random_src_id=random.randrange(20)
                dataset=dataset_dict[random_src_id]
                train_dataset_list.append(dataset)

            train_data_loader_list=[]
            for dataset in train_dataset_list:
                data_loader=ModelTrainUtils.get_data_loader(dataset=dataset,batch_size=config['batch_size'])
                train_data_loader_list.append(data_loader)
            print(f"train_data_loader_list is ready!")

            dataset_dict_list=DataUtils.load_from_pickle(file_name=f"val_20_ladder",path="tgnn",dir_type="val")
            val_dataset_list=[]
            for dataset_dict in dataset_dict_list:
                for _,dataset in dataset_dict.items():
                    val_dataset_list.append(dataset)

            val_data_loader_list=[]
            for dataset in val_dataset_list:
                data_loader=ModelTrainUtils.get_data_loader(dataset=dataset,batch_size=config['batch_size'])
                val_data_loader_list.append(data_loader)
            print(f"val_data_loader_list is ready!")

            """
            model setting and training
            """
            match config['model']:
                case 'tgat':
                    model=TGAT(node_dim=1,latent_dim=config['latent_dim'])
                case 'tgn':
                    model=TGN(node_dim=1,latent_dim=config['latent_dim'],emb=config['emb'])
            ModelTrainer.train(model=model,train_data_loader_list=train_data_loader_list,val_data_loader_list=val_data_loader_list,validate=True,config=config)

        case 1:
            """
            App 1.
            train 
            """
            """
            wandb
            """
            if config['wandb']:
                if config['model']=='tgat':
                    wandb.init(project="TGNN",name=f"{config['model']}_{config['seed']}_{config['lr']}_{config['batch_size']}")
                else: # tgn
                    wandb.init(project="TGNN",name=f"{config['model']}_{config['emb']}_{config['seed']}_{config['lr']}_{config['batch_size']}")
                wandb.config.update(config)

            """
            train data loader list
            """
            graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
            train_dataset_list=[]
            for graph_type in graph_type_list:
                dataset_dict_list=DataUtils.load_from_pickle(file_name=f"train_20_{graph_type}",path="tgnn",dir_type="train")
                for dataset_dict in dataset_dict_list:
                    if config['random_src']:
                        random_src_id=random.randrange(20)
                        dataset=dataset_dict[random_src_id]
                        train_dataset_list.append(dataset)
                    else:
                        for _,dataset in dataset_dict.items():
                            train_dataset_list.append(dataset)
            
            train_data_loader_list=[]
            for dataset in train_dataset_list:
                data_loader=ModelTrainUtils.get_data_loader(dataset=dataset,batch_size=config['batch_size'])
                train_data_loader_list.append(data_loader)
            print(f"train_data_loader_list is ready!")

            """
            val data loader list
            """
            graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
            val_dataset_list=[]
            for graph_type in graph_type_list:
                dataset_dict_list=DataUtils.load_from_pickle(file_name=f"val_20_{graph_type}",path="tgnn",dir_type="val")
                for dataset_dict in dataset_dict_list:
                    for _,dataset in dataset_dict.items():
                        val_dataset_list.append(dataset)

            val_data_loader_list=[]
            for dataset in val_dataset_list:
                data_loader=ModelTrainUtils.get_data_loader(dataset=dataset,batch_size=config['batch_size'])
                val_data_loader_list.append(data_loader)
            print(f"val_data_loader_list is ready!")

            """
            model setting and training
            """
            match config['model']:
                case 'tgat':
                    model=TGAT(node_dim=1,latent_dim=config['latent_dim'])
                case 'tgn':
                    model=TGN(node_dim=1,latent_dim=config['latent_dim'],emb=config['emb'])
            ModelTrainer.train(model=model,train_data_loader_list=train_data_loader_list,val_data_loader_list=val_data_loader_list,validate=True,config=config)

            """
            save model
            """
            if config['save_model']:
                match config['model']:
                    case 'tgat':
                        model_name=f"tgat_{config['seed']}_{config['lr']}_{config['batch_size']}"
                    case 'tgn':
                        model_name=f"tgn_{config['emb']}_{config['seed']}_{config['lr']}_{config['batch_size']}"
                DataUtils.save_model_parameter(model=model,model_name=model_name)

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
    parser.add_argument("--num_nodes",type=int,default=20)
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
        'num_nodes':args.num_nodes
    }
    app_train(config=config)