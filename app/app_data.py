import os
import random
import numpy as np
import argparse
import wandb
import torch
from tqdm import tqdm
from tgnn import DataUtils,GraphUtils

def app_data(config: dict):
    match config['app_num']:
        case 1:
            """
            App 1.
            Convert and save all type graph_list_dict to dataset_dict_list_all_type
                num_nodes: 20, 50, 100 
            """
            train_20=DataUtils.load_from_pickle(file_name="train_20",path="trne",dir_type="graph")
            val_20=DataUtils.load_from_pickle(file_name="val_20",path="trne",dir_type="graph")
            test_20=DataUtils.load_from_pickle(file_name="test_20",path="trne",dir_type="graph")
            test_50=DataUtils.load_from_pickle(file_name="test_50",path="trne",dir_type="graph")
            test_100=DataUtils.load_from_pickle(file_name="test_100",path="trne",dir_type="graph")

            dataset_dict_list_all_type=GraphUtils.convert_to_dataset_dict_list_all_type(graph_list_dict=train_20)
            DataUtils.save_dataset_dict_list_all_type(dataset_dict_list_all_type=dataset_dict_list_all_type,file_name="train_20",dir_type="train")
            print(f"Finish train_20")
            
            dataset_dict_list_all_type=GraphUtils.convert_to_dataset_dict_list_all_type(graph_list_dict=val_20)
            DataUtils.save_dataset_dict_list_all_type(dataset_dict_list_all_type=dataset_dict_list_all_type,file_name="val_20",dir_type="val")
            print(f"Finish val_20")

            dataset_dict_list_all_type=GraphUtils.convert_to_dataset_dict_list_all_type(graph_list_dict=test_20)
            DataUtils.save_dataset_dict_list_all_type(dataset_dict_list_all_type=dataset_dict_list_all_type,file_name="test_20",dir_type="test")
            print(f"Finish test_20")

            dataset_dict_list_all_type=GraphUtils.convert_to_dataset_dict_list_all_type(graph_list_dict=test_50)
            DataUtils.save_dataset_dict_list_all_type(dataset_dict_list_all_type=dataset_dict_list_all_type,file_name="test_50",dir_type="test")
            print(f"Finish test_50")

            dataset_dict_list_all_type=GraphUtils.convert_to_dataset_dict_list_all_type(graph_list_dict=test_100)
            DataUtils.save_dataset_dict_list_all_type(dataset_dict_list_all_type=dataset_dict_list_all_type,file_name="test_100",dir_type="test")
            print(f"Finish test_100")

        case 2:
            """
            App 2.
            Convert and save single type graph_list to dataset_dict_list 
            """
            graph_list_dict=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}",path="trne",dir_type="graph")
            graph_list=graph_list_dict[config['graph_type']]
            dataset_dict_list=GraphUtils.convert_to_dataset_dict_list(graph_list=graph_list,graph_type=config['graph_type'])
            DataUtils.save_dataset_dict_list(dataset_dict_list=dataset_dict_list,file_name=f"{config['mode']}_{config['num_nodes']}_{config['graph_type']}",dir_type=config['mode'])

if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--mode",type=str,default='test')
    parser.add_argument("--num_nodes",type=int,default=20)
    parser.add_argument("--graph_type",type=str,default='default')
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        'mode':args.mode,
        'num_nodes':args.num_nodes,
        'graph_type':args.graph_type
    }
    app_data(config=config)