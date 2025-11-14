import random
import numpy as np
import pandas as pd
import networkx as nx
import torch
from typing_extensions import Literal
from tqdm import tqdm
from .graph_utils import GraphUtils


class ModelTrainUtils:
    @staticmethod
    def get_data_loader(dataset:list,batch_size:int):
        """
        Input:
            dataset: list of event_dict
            batch_size: batch size of edge_events
        Output:
            data_loader: List of batch_dict
                x: [B,N,1]
                t: [B,N,1]
                src: [B,1]
                tar: [B,1]
                n_mask: [B,N,], neighbor mask of target nodes
                tar_label: [B,1]
                label: [B,N,1]
                edge_index: [2,E]
        """
        data_loader=[]
        for i in range(0,len(dataset),batch_size):
            batch_event_dicts=dataset[i:i+batch_size]

            batch_x=torch.stack([e['x'] for e in batch_event_dicts],dim=0) # [B,N,1]
            batch_t=torch.stack([e['t'] for e in batch_event_dicts],dim=0) # [B,N,1]
            batch_n_mask=torch.stack([e['n_mask'] for e in batch_event_dicts],dim=0) # [B,N,]
            batch_label=torch.stack([e['label'] for e in batch_event_dicts],dim=0) # [B,N,1]

            batch_src=torch.tensor([e['src'] for e in batch_event_dicts]).unsqueeze(1) # [B,1]
            batch_tar=torch.tensor([e['tar'] for e in batch_event_dicts]).unsqueeze(1) # [B,1]
            batch_tar_label=torch.tensor([e['tar_label'] for e in batch_event_dicts]).unsqueeze(1) # [B,1]

            edge_index=batch_event_dicts[0]['edge_index'] # [2,E]

            data_loader.append({
                'x':batch_x,
                't':batch_t,
                'src':batch_src,
                'tar':batch_tar,
                'n_mask':batch_n_mask,
                'tar_label':batch_tar_label,
                'label':batch_label,
                'edge_index':edge_index
            })
        return data_loader

class EarlyStopping:
    def __init__(self,patience=1):
        self.patience=patience
        self.patience_count=0
        self.prev_loss=np.inf
        self.best_state=None
        self.early_stop=False
    def __call__(self,val_loss:float,model:torch.nn.Module):
        if self.prev_loss==np.inf:
            self.prev_loss=val_loss
            self.best_state={k: v.clone() for k,v in model.state_dict().items()}
            return None
        else:
            if not np.isfinite(val_loss):
                print(f"Loss is NaN or Inf!")
                self.early_stop=True
                model.load_state_dict(self.best_state)
                return model
            
            if self.prev_loss<=val_loss:
                self.patience_count+=1
                if self.patience<self.patience_count:
                    print(f"Loss increases during {self.patience_count} patience!")
                    self.early_stop=True
                    model.load_state_dict(self.best_state)
                    return model
            else:
                self.patience_count=0
                self.prev_loss=val_loss
                self.best_state={k: v.clone() for k,v in model.state_dict().items()}
                return None