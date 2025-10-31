import os
import pickle
import torch
from tqdm import tqdm
from typing_extensions import Literal

class DataUtils:
    class DataLoader:
        dataset_path=os.path.join('..','data','tgnn')
        @staticmethod
        def save_to_pickle(data,file_name:str,dir_type:Literal['graph','train','val','test']):
            file_name=file_name+".pkl"
            file_path=os.path.join(DataUtils.DataLoader.dataset_path,dir_type,file_name)
            with open(file_path,'wb') as f:
                pickle.dump(data,f)
            print(f"Save {file_name}")

        @staticmethod
        def load_from_pickle(file_name:str,dir_type:Literal['graph','train','val','test']):
            file_name=file_name+".pkl"
            file_path=os.path.join(DataUtils.DataLoader.dataset_path,dir_type,file_name)
            with open(file_path,'rb') as f:
                data=pickle.load(f)
            print(f"Load {file_name}")
            return data
        
        @staticmethod
        def save_model_parameter(model,model_name:str):
            file_name=model_name+".pt"
            file_path=os.path.join(DataUtils.DataLoader.dataset_path,"inference",file_name)
            torch.save(model.state_dict(),file_path)
            print(f"Save {model_name} model parameter")
        
        @staticmethod
        def load_model_parameter(model,model_name:str):
            file_name=model_name+".pt"
            file_path=os.path.join(DataUtils.DataLoader.dataset_path,"inference",file_name)
            model.load_state_dict(torch.load(file_path))
            return model