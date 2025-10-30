import os
import pickle
import pandas as pd
import networkx as nx
import torch
from typing_extensions import Literal
from graph_utils import GraphUtils

def get_sub_edge_index_and_timestamp(df:pd.DataFrame,edge_event:tuple):
    src,tar,ts=edge_event
    new_row=pd.DataFrame({'src':[src],'tar':[tar],'ts':[ts]})
    df_prev=df[df['ts']<ts]
    df_prev=pd.concat([df_prev,new_row],ignore_index=True)
    df_latest=df_prev.drop_duplicates(subset=['src','tar'],keep='last').reset_index(drop=True)
    edge_index=torch.tensor(df_latest[['src','tar']].values.T,dtype=torch.long) # [2,sub_E]
    edge_timestamp=torch.tensor(df_latest['ts'].values,dtype=torch.float).view(-1,1) # [sub_E,1]
    return edge_index,edge_timestamp

def compute_neighbor_node_tensor(edge_index:torch.Tensor,edge_timestamp:torch.Tensor,target_node:int):
    """
    edge_index: [2,E]
    edge_timestamp: [E,1]
    target_node: int
    """
    src,tar=edge_index
    ts=edge_timestamp.view(-1) # [E,]
    mask=(tar==target_node)

    neighbor_ids=src[mask]
    neighbor_ts=ts[mask]
    return neighbor_ids,neighbor_ts

