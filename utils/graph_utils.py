import numpy as np
import pandas as pd
import networkx as nx
import copy
import torch
from typing_extensions import Literal

class GraphUtils:
    class GraphManager:
        @staticmethod
        def initialize_node_attr(graph:nx.DiGraph,source_id:int=0):
            for node in graph.nodes():
                if node==source_id:
                    graph.nodes[node]['r']=1.0
                    graph.nodes[node]['a']=0.0
                else:
                    graph.nodes[node]['r']=0.0
                    graph.nodes[node]['a']=1.1
                graph.nodes[node]['p']=node
            GraphUtils.GraphManager.remap_node_predecessor_to_sorted_index(graph=graph)

        @staticmethod
        def get_node_attr_tensor(graph:nx.DiGraph,attr:str='x'):
            match attr:
                case 'r'|'a'|'pos':
                    attr_array=np.array([graph.nodes[node_id][attr] for node_id in range(graph.number_of_nodes())],dtype=np.float32)
                    attr_tensor=torch.tensor(attr_array,dtype=torch.float32)
                    attr_tensor=attr_tensor.unsqueeze(-1)
                case 'p':
                    attr_array=np.array([graph.nodes[node_id][attr] for node_id in range(graph.number_of_nodes())],dtype=np.int64)
                    attr_tensor=torch.tensor(attr_array,dtype=torch.int64)
                    attr_tensor=attr_tensor.unsqueeze(-1)
            return attr_tensor
        
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
            df=df.sort_values(['ts'])
            return df