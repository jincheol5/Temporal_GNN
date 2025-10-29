import os
import pickle
import pandas as pd
import networkx as nx
import torch
from typing_extensions import Literal
from graph_utils import GraphUtils

def get_sub_edge_index_and_timestamp(df:pd.DataFrame,edge_event:tuple):
    src=edge_event[0]
    tar=edge_event[1]
    ts=edge_event[2]
    mask=(df['src']==src)&(df['tar']==tar)&(df['ts']==ts)
    target_idx=df.index[mask][0]
    df_prev=df.iloc[:target_idx+1].copy()
    df_latest=(
        df_prev.drop_duplicates(subset=['src','tar'],keep='last').reset_index(drop=True)
    )
    edge_index=torch.tensor(df_latest[['src','tar']].values.T,dtype=torch.long)
    edge_timestamp=torch.tensor(df_latest['ts'].values,dtype=torch.float).view(-1, 1)
    return edge_index,edge_timestamp

def compute_batch(graph:nx.DiGraph,df:pd.DataFrame,batch_size:int=1):
    batch_df_list=[df.iloc[i:i+batch_size] for i in range(0,len(df),batch_size)]
