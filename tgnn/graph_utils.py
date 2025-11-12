import numpy as np
import pandas as pd
import networkx as nx
import copy
import torch
from typing_extensions import Literal
from torch_geometric.utils import sort_edge_index

class GraphUtils:
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
    def get_edge_index(graph:nx.DiGraph):
        edge_list=list(graph.edges()) 
        edge_index=torch.tensor(edge_list,dtype=torch.long).t().contiguous()
        sorted_edge_index=sort_edge_index(edge_index=edge_index,sort_by_row=False)
        return sorted_edge_index

    @staticmethod
    def compute_tR_step(num_nodes:int,source_id:int=0,edge_event:tuple=None,gamma:torch.Tensor=None,init:bool=False):
        """
        gamma: [N,3] tensor, tR and time
        """
        if init:
            gamma=torch.zeros((num_nodes,3),dtype=torch.float) # tR,time,ts
            gamma[:,0]=0.0 # tR
            gamma[:,1]=1.1 # time
            gamma[:,2]=0.0 # ts

            gamma[source_id,0]=1.0
            gamma[source_id,1]=0.0
        else:
            src,tar,ts=edge_event
            if gamma[src,0].item()==1.0 and gamma[tar,0].item()==0.0 and gamma[src,1].item()<ts:
                gamma[tar,0]=1.0
                gamma[tar,1]=ts
                gamma[tar,2]=ts
        return gamma

    @staticmethod
    def compute_dataset(event_stream:list,num_nodes:int,source_id:int):
        """
        Input:
            event_stream: List of edge_event tuple
            num_nodes: number of nodes
            source_id: source node id
        Output:
            dataset: List of data_dict
                data_dict:
                    x: [N,1]
                    t: [N,1]
                    src: src id
                    tar: tar id
                    n_mask: [N,]
                    label: label (1.0 or 0.0)
        """
        dataset=[]
        num_edge_events=len(event_stream)
        raw_node_feature=torch.zeros((num_edge_events,num_nodes),dtype=torch.float)
        raw_node_feature[:,source_id]=1.0 
        raw_node_feature=raw_node_feature.unsqueeze(-1) # [E,N,1]
        x_list=list(torch.unbind(raw_node_feature,dim=0)) # List of [N,1]
        neighbor_mask=torch.zeros((num_edge_events,num_nodes),dtype=torch.bool) # [E,N]
        neighbor_history=[torch.zeros(num_nodes,dtype=torch.bool) for _ in range(num_nodes)] # List of [N,]
        gamma=GraphUtils.compute_tR_step(num_nodes=num_nodes,source_id=source_id,init=True) # [N,3]
        for i,(edge_event,x) in enumerate(zip(event_stream,x_list)):
            src,tar,ts=edge_event

            # compute tR step
            gamma=GraphUtils.compute_tR_step(num_nodes=num_nodes,source_id=source_id,edge_event=edge_event,gamma=gamma) # [N,3]
            
            # save x,t,src,tar,n_mask,label to data_dict
            data_dict={}
            data_dict['x']=x

            cur_t=gamma[:,-1:] # [N,1]
            data_dict['t']=torch.abs(cur_t-ts) # [N,1] 
            
            data_dict['src']=src
            data_dict['tar']=tar
            
            neighbor_history[tar][src]=True
            neighbor_mask[i]=neighbor_history[tar] # 참조가 아닌 복사(tensor index 대입)
            data_dict['n_mask']=neighbor_mask[i]

            data_dict['label']=gamma[tar,0].item()
            dataset.append(data_dict)
        return dataset