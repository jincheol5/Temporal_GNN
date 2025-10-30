import numpy as np
import pandas as pd
import networkx as nx
import copy
import torch
from typing_extensions import Literal

class GraphUtils:
    @staticmethod
    def get_event_stream_df(graph:nx.DiGraph):
        df=pd.DataFrame({
            'src': pd.Series(dtype='int'), # source node
            'tar': pd.Series(dtype='int'), # target node
            'ts': pd.Series(dtype='float') # timestamp
            })
        for u,v,data in graph.edges(data=True):
            time_list=data['t']
            for timestamp in time_list:
                df.loc[len(df)]=[u,v,timestamp]
        df=df.sort_values(['ts']).reset_index(drop=True)
        return df

    @staticmethod
    def get_node_raw_feature(num_nodes:int,source_id:int=0):
        raw_node_feature=torch.zeros((num_nodes,),dtype=torch.float)
        raw_node_feature[source_id]=1.0
        return raw_node_feature.unsqueeze(-1) # [N,1]

    @staticmethod
    def get_node_time_feature(num_nodes:int,gamma:dict):
        node_time_feature=torch.zeros((num_nodes,),dtype=torch.float)
        for node in gamma.keys():
            if gamma[node][1]!=1.1:
                node_time_feature[node]=gamma[node][1]
        return node_time_feature.unsqueeze(-1) # [N,1]
    
    @staticmethod
    def get_sub_edge_index(df:pd.DataFrame,edge_event:tuple):
        """
        Input:
            df: DataFrame
            edge_event: (src,tar,timestamp) tuple
        Output:
            sub edge_index: [2,sub_E]
        """
        src,tar,ts=edge_event
        new_row=pd.DataFrame({'src':[src],'tar':[tar],'ts':[ts]})
        df_prev=df[df['ts']<ts]
        df_prev=pd.concat([df_prev,new_row],ignore_index=True)
        df_latest=df_prev.drop_duplicates(subset=['src','tar'],keep='last').reset_index(drop=True)
        edge_index=torch.tensor(df_latest[['src','tar']].values.T,dtype=torch.long) # [2,sub_E]
        return edge_index

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

    class GraphAlgorithm:
        @staticmethod
        def compute_tR_one_pass_step(graph:nx.DiGraph,source_id:int=0,init:bool=False,edge_event:tuple=None,gamma:dict=None):
            if init:
                gamma={}
                for node in graph.nodes():
                    gamma[node]=[0.0,1.1] # [reachability,visited_time]
                gamma[source_id]=[1.0,0.0]
            else:
                src,tar,ts=edge_event
                if gamma[src][0]==1.0 and gamma[tar][0]==0.0:
                    if gamma[src][1]<ts:
                        gamma[tar][0]=1.0
                        gamma[tar][1]=ts
            return gamma