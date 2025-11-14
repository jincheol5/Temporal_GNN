import os
import pickle
import torch
from tqdm import tqdm
from typing_extensions import Literal

class DataUtils:
    tgnn_path=os.path.join('..','data','tgnn')
    trne_path=os.path.join('..','data','trne')
    @staticmethod
    def save_to_pickle(data,file_name:str,path:Literal['tgnn','trne'],dir_type:Literal['graph','train','val','test']):
        file_name=file_name+".pkl"
        if path=='tgnn':
            file_path=os.path.join(DataUtils.tgnn_path,dir_type,file_name)
        else: # trne
            file_path=os.path.join(DataUtils.trne_path,dir_type,file_name)
        with open(file_path,'wb') as f:
            pickle.dump(data,f)
        print(f"Save {file_name}")

    @staticmethod
    def load_from_pickle(file_name:str,path:Literal['tgnn','trne'],dir_type:Literal['graph','train','val','test']):
        file_name=file_name+".pkl"
        if path=='tgnn':
            file_path=os.path.join(DataUtils.tgnn_path,dir_type,file_name)
        else: # trne
            file_path=os.path.join(DataUtils.trne_path,dir_type,file_name)
        with open(file_path,'rb') as f:
            data=pickle.load(f)
        print(f"Load {file_name}")
        return data

    @staticmethod
    def save_dataset_dict_list(dataset_dict_list:list,file_name:str,graph_type:str,dir_type:Literal['train','val','test']):
        DataUtils.save_to_pickle(data=dataset_dict_list,file_name=f"{file_name}_{graph_type}",path='tgnn',dir_type=dir_type)
        print(f"Save {file_name}_{graph_type}!")

    @staticmethod
    def save_dataset_dict_list_all_type(dataset_dict_list_all_type:dict,file_name:str,dir_type:Literal['train','val','test']):
        for key,value in tqdm(dataset_dict_list_all_type.items(),desc=f"Save {file_name} ..."):
            match key:
                case 'ladder':
                    DataUtils.save_to_pickle(data=value,file_name=file_name+"_ladder",path='tgnn',dir_type=dir_type)
                case 'grid':
                    DataUtils.save_to_pickle(data=value,file_name=file_name+"_grid",path='tgnn',dir_type=dir_type)
                case 'tree':
                    DataUtils.save_to_pickle(data=value,file_name=file_name+"_tree",path='tgnn',dir_type=dir_type)
                case 'erdos_renyi':
                    DataUtils.save_to_pickle(data=value,file_name=file_name+"_erdos_renyi",path='tgnn',dir_type=dir_type)
                case 'barabasi_albert':
                    DataUtils.save_to_pickle(data=value,file_name=file_name+"_barabasi_albert",path='tgnn',dir_type=dir_type)
                case 'community':
                    DataUtils.save_to_pickle(data=value,file_name=file_name+"_community",path='tgnn',dir_type=dir_type)
                case 'caveman':
                    DataUtils.save_to_pickle(data=value,file_name=file_name+"_caveman",path='tgnn',dir_type=dir_type)

    @staticmethod
    def save_model_parameter(model,model_name:str):
        file_name=model_name+".pt"
        file_path=os.path.join(DataUtils.tgnn_path,"inference",file_name)
        torch.save(model.state_dict(),file_path)
        print(f"Save {model_name} model parameter")

    @staticmethod
    def load_model_parameter(model,model_name:str):
        file_name=model_name+".pt"
        file_path=os.path.join(DataUtils.tgnn_path,"inference",file_name)
        model.load_state_dict(torch.load(file_path))
        return model