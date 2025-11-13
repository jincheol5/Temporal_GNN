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
            Convert graph_list_dict to all_dataset_dict_list 
            """
            train_20=DataUtils.load_from_pickle("train_20","graph")
            val_20=DataUtils.load_from_pickle("val_20","graph")
            test_20=DataUtils.load_from_pickle("test_20","graph")
            test_50=DataUtils.load_from_pickle("test_50","graph")
            test_100=DataUtils.load_from_pickle("test_100","graph")

            dataset_dict_list_all_type=GraphUtils.convert_to_dataset_dict_list_all_type(graph_list_dict=train_20)
            DataUtils.save_dataset_dict_list_all_type(dataset_dict_list_all_type=dataset_dict_list_all_type)
            
            dataset_dict_list_all_type=GraphUtils.convert_to_dataset_dict_list_all_type(graph_list_dict=val_20)
            DataUtils.save_dataset_dict_list_all_type(dataset_dict_list_all_type=dataset_dict_list_all_type)

            dataset_dict_list_all_type=GraphUtils.convert_to_dataset_dict_list_all_type(graph_list_dict=test_20)
            DataUtils.save_dataset_dict_list_all_type(dataset_dict_list_all_type=dataset_dict_list_all_type)

            dataset_dict_list_all_type=GraphUtils.convert_to_dataset_dict_list_all_type(graph_list_dict=test_50)
            DataUtils.save_dataset_dict_list_all_type(dataset_dict_list_all_type=dataset_dict_list_all_type)

            dataset_dict_list_all_type=GraphUtils.convert_to_dataset_dict_list_all_type(graph_list_dict=test_100)
            DataUtils.save_dataset_dict_list_all_type(dataset_dict_list_all_type=dataset_dict_list_all_type)

if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
    }
    app_data(config=config)