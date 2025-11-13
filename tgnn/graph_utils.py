import numpy as np
import pandas as pd
import networkx as nx
import copy
import torch
from typing_extensions import Literal
from tqdm import tqdm
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
    def convert_to_dataset(event_stream:list,num_nodes:int,source_id:int):
        """
        Input:
            event_stream: List of edge_event tuple
            num_nodes: number of nodes
            source_id: source node id
        Output:
            dataset: List of event_dict
                event_dict:
                    x: [N,1]
                    t: [N,1]
                    src: src id
                    tar: tar id
                    n_mask: [N,]
                    tar_label: tar label (1.0 or 0.0)
                    label: [N,1], label of all nodes  
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
            
            # save x,t,src,tar,n_mask,tar_label,label to event_dict
            event_dict={}
            event_dict['x']=x

            cur_t=gamma[:,-1:] # [N,1]
            event_dict['t']=torch.abs(cur_t-ts) # [N,1] 
            
            event_dict['src']=src
            event_dict['tar']=tar
            
            neighbor_history[tar][src]=True
            neighbor_mask[i]=neighbor_history[tar] # 참조가 아닌 복사(tensor index 대입)
            event_dict['n_mask']=neighbor_mask[i]

            event_dict['tar_label']=gamma[tar,0].item()
            event_dict['label']=gamma[:,0:1]
            dataset.append(event_dict)

        return dataset

    @staticmethod
    def convert_to_dataset_dict(graph:nx.DiGraph):
        """
        Input:
            graph
        Output:
            dataset_dict
                key: source_id
                value: dataset
        """
        num_nodes=graph.number_of_nodes()
        event_stream=GraphUtils.get_event_stream(graph=graph)
        dataset_dict={}
        for source_id in range(num_nodes):
            dataset=GraphUtils.compute_dataset(event_stream=event_stream,num_nodes=num_nodes,source_id=source_id)
            dataset_dict[source_id]=dataset
        return dataset_dict
    
    @staticmethod
    def convert_to_dataset_dict_list(graph_list:list,graph_type:str):
        """
        Input:
            graph_list
        Output:
            list of dataset_dict
        """
        dataset_dict_list=[]
        for graph in tqdm(graph_list,desc=f"Convert {graph_type} graph_list..."):
            graph.remove_edges_from(nx.selfloop_edges(graph))
            dataset_dict=GraphUtils.compute_dataset_dict(graph=graph)
            dataset_dict_list.append(dataset_dict)
        return dataset_dict_list
    
    @staticmethod
    def convert_to_dataset_dict_list_all_type(graph_list_dict:dict):
        """
        Input:
            graph_list_dict
        Output:
            dataset_dict_list_all_type
            dict of dataset_dict_list for all type graphs
        """
        dataset_dict_list_all_type={}
        for graph_type,graph_list in tqdm(graph_list_dict.items(),desc=f"Converting all type graphs..."):
            dataset_dict_list=GraphUtils.convert_to_dataset_dict_list(graph_list=graph_list,graph_type=graph_type)
            dataset_dict_list_all_type[graph_type]=dataset_dict_list

class GraphAnalysis:
    @staticmethod
    def check_elements(graph:nx.DiGraph):
        num_nodes=graph.number_of_nodes()
        num_static_edges=graph.number_of_edges()
        num_edge_events=0
        for src,tar in graph.edges():
            count=0
            for time in graph[src][tar]['T']:
                if time!=0.0:
                    count+=1
            num_edge_events+=count
        return num_nodes,num_static_edges,num_edge_events

    @staticmethod
    def check_min_max_edge_time_len(graph:nx.DiGraph):
        min_T_len=min((len(data.get('T',[])) for _,_,data in graph.edges(data=True)),default=0)
        max_T_len=max((len(data.get('T',[])) for _,_,data in graph.edges(data=True)),default=0)
        return min_T_len,max_T_len

    @staticmethod
    def check_reachability_ratio(r:torch.Tensor):
        r_flat=r.view(-1).to(torch.float32)  # [N,1] -> [N,]
        return r_flat.mean().item()