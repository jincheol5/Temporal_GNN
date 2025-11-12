import numpy as np
import pandas as pd
import networkx as nx
import copy
import torch
from typing_extensions import Literal
from torch_geometric.utils import sort_edge_index

class GraphUtils:
    @staticmethod
    def get_event_stream_df(graph:nx.DiGraph):
        rows=[]
        for u,v,data in graph.edges(data=True):
            time_list=data['T']
            for timestamp in time_list:
                rows.append((int(u),int(v),float(timestamp)))
        df=pd.DataFrame(rows,columns=['src','tar','ts'])
        df=df.sort_values('ts').reset_index(drop=True)
        return df
    
    @staticmethod
    def get_event_stream(graph:nx.DiGraph):
        """
        Output:
            rows: tuple list
        """
        event_stream=[]
        for u,v,data in graph.edges(data=True):
            time_list=data['T']
            for timestamp in time_list:
                event_stream.append((int(u),int(v),float(timestamp)))
        event_stream=sorted(event_stream,key=lambda x:x[2])
        return event_stream

    @staticmethod
    def get_node_raw_feature(num_nodes:int,source_id:int=0,batch_size:int=1):
        raw_node_feature=torch.zeros((batch_size,num_nodes,),dtype=torch.float)
        raw_node_feature[:,source_id]=1.0
        return raw_node_feature.unsqueeze(-1) # [B,N,1]

    @staticmethod
    def get_node_time_feature(gamma:np.ndarray):
        """
        gamma: [N,2], ndarray
        """
        return torch.from_numpy(gamma[:,1:2]) # [N,1]

    @staticmethod
    def get_edge_index(graph:nx.DiGraph):
        edge_list=list(graph.edges()) 
        edge_index=torch.tensor(edge_list,dtype=torch.long).t().contiguous()
        sorted_edge_index=sort_edge_index(edge_index=edge_index,sort_by_row=False)
        return sorted_edge_index

    @staticmethod
    def get_neighbor_node_mask(edge_index:torch.Tensor,target_node:int,num_nodes:int):
        """
        Input:
            edge_index: [2,sub_E]
            target_node: int
            num_nodes: int
        Output:
            neighbor_mask: [N,]
        """
        src,tar=edge_index
        mask=(tar==target_node)
        neighbor_nodes=src[mask]
        neighbor_mask=torch.zeros(num_nodes,dtype=torch.bool,device=edge_index.device)
        neighbor_mask[neighbor_nodes]=True
        return neighbor_mask

    @staticmethod
    def get_neighbor_mask_list(event_stream:list,num_nodes:int,batch_size:int):
        """
        Compute:
            neighbor_mask: [E,N], boolean tensor
            split to neighbor_mask_list
        Output:
            neighbor_mask_list: List of [B,N], boolean tensor list
        """
        num_edge_events=len(event_stream)
        neighbor_mask=torch.zeros((num_edge_events,num_nodes),dtype=torch.bool)
        neighbors=[torch.zeros(num_nodes,dtype=torch.bool) for _ in range(num_nodes)]
        for i,(src,tar,ts) in enumerate(event_stream):
            neighbors[tar][src]=True
            neighbor_mask[i]=neighbors[tar] # 참조가 아닌 복사(tensor index 대입)
        neighbor_mask_list=[neighbor_mask[i:i+batch_size] for i in range(0,num_edge_events,batch_size)]
        return neighbor_mask_list

    @staticmethod
    def compute_tR_step(num_nodes:int,source_id:int=0,edge_event:tuple=None,gamma:np.ndarray=None,init:bool=False):
        """
        gamma: [N,2], np.ndarray
        """
        if init:
            gamma=np.zeros((num_nodes,2),dtype=float)
            gamma[:,0]=0.0 # tR
            gamma[:,1]=1.1 # time
            gamma[source_id]=[1.0,0.0]
        else:
            src,tar,ts=edge_event
            if gamma[src,0]==1.0 and gamma[tar,0]==0.0 and gamma[src,1]<ts:
                gamma[tar,0]=1.0
                gamma[tar,1]=ts
        return gamma