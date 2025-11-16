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
            Convert and save all type graph_list_dict into dataset_list and store them in chunks.
            Only train
            """
            graph_list_dict=DataUtils.load_from_pickle(file_name="train_20",path="trne",dir_type="graph")
            dataset_list_dict=GraphUtils.convert_to_dataset_list_dict(graph_list_dict=graph_list_dict)
            DataUtils.save_dataset_list_dict(dataset_list_dict=dataset_list_dict,file_name="train_20",chunk_size=config['chunk_size'],dir_type="train")

        case 2:
            """
            App 2.
            Convert and save all type graph_list_dict into each type dataset_list and store them in chunks.
            Only val, test
            """

        case 3:
            """
            App 3.
            Convert and save single type graph_list to dataset_dict_list 
            """
            graph_list_dict=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}",path="trne",dir_type="graph")
            dataset_list_dict=GraphUtils.convert_to_dataset_list_dict(graph_list_dict=graph_list_dict)
            print(f"Done!")


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
    parser.add_argument("--chunk_size",type=int,default=1)
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        'mode':args.mode,
        'num_nodes':args.num_nodes,
        'graph_type':args.graph_type,
        'chunk_size':args.chunk_size
    }
    app_data(config=config)