import numpy as np
import pandas as pd
import networkx as nx
import copy
import torch
from typing_extensions import Literal

class GraphUtils:
    class GraphManager:
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
        def get_raw_node_feature(graph:nx.DiGraph,source_id:int=0):
            raw_node_feature=torch.zeros((graph.number_of_nodes(),1),dtype=torch.float)
            raw_node_feature[source_id]=1.0
            return raw_node_feature

        @staticmethod
        def get_node_label_from_gamma(gamma:dict):
            r_values=[gamma[node][0] for node in sorted(gamma.keys())]
            label=torch.tensor(r_values,dtype=torch.float).view(-1, 1)
            return label

    class GraphAlgorithm:
        @staticmethod
        def compute_tR_one_pass_step(graph:nx.DiGraph,source_id:int=0,init:bool=False,edge_event_list:list=None,gamma:dict=None):
            if init:
                gamma={}
                for node in graph.nodes():
                    gamma[node]=[0.0,1.1] # [reachability,visited_time]
                gamma[source_id]=[1.0,0.0]
            else:
                for edge_event in edge_event_list:
                    # edge_event=(src,tar,ts)
                    src=edge_event[0]
                    tar=edge_event[1]
                    ts=edge_event[2]
                    if gamma[src][0]==1.0 and gamma[tar][0]==0.0:
                        if gamma[src][1]<ts:
                            gamma[tar][0]=1.0
                            gamma[tar][1]=ts
            return gamma