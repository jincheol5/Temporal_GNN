import pandas as pd
import networkx as nx
import torch
from graph_utils import GraphUtils


class ModelTrainUtils:
    @staticmethod
    def get_batch_dict_list(graph:nx.DiGraph,source_id:int=0,batch_size:int=1):
        batch_dict_list=[]
        df=GraphUtils.GraphManager.get_event_stream_df(graph=graph)
        node_raw_feature=GraphUtils.GraphManager.get_node_raw_feature(graph=graph,source_id=source_id)
        batch_label_list=ModelTrainUtils.compute_batch_label_list(graph=graph,df=df,source_id=source_id,batch_size=batch_size)
        batch_df_list=[df.iloc[i:i+batch_size] for i in range(0,len(df),batch_size)]
        for idx,batch_df in enumerate(batch_df_list):
            batch_dict={}
            last_row=batch_df.iloc[-1]
            last_edge_event=(last_row.src,last_row.tar,last_row.ts)
            sub_edge_index,sub_edge_timestamp=ModelTrainUtils.get_sub_edge_index_and_timestamp(df=df,edge_event=last_edge_event)
            batch_dict['source_id']=source_id
            batch_dict['x']=node_raw_feature
            batch_dict['y']=batch_label_list[idx]
            batch_dict['edge_index']=sub_edge_index
            batch_dict['edge_timestamp']=sub_edge_timestamp
            batch_dict_list.append(batch_dict)
        return batch_dict_list

    @staticmethod
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
        edge_index=torch.tensor(df_latest[['src','tar']].values.T,dtype=torch.long) # [2,sub_E]
        edge_timestamp=torch.tensor(df_latest['ts'].values,dtype=torch.float).view(-1, 1) # [sub_E,1]
        return edge_index,edge_timestamp

    @staticmethod
    def compute_batch_label_list(graph:nx.DiGraph,df:pd.DataFrame,source_id:int=0,batch_size:int=1):
        batch_df_list=[df.iloc[i:i+batch_size] for i in range(0,len(df),batch_size)]
        batch_label_list=[]

        gamma=GraphUtils.GraphAlgorithm.compute_tR_one_pass_step(graph=graph,source_id=source_id,init=True)
        for batch_df in batch_df_list:
            edge_event_list=[]
            for row in batch_df.itertuples():
                edge_event=(row.src,row.tar,row.ts)
                edge_event_list.append(edge_event)
            gamma=GraphUtils.GraphAlgorithm.compute_tR_one_pass_step(graph=graph,source_id=source_id,edge_event_list=edge_event_list,gamma=gamma)
            batch_label=GraphUtils.GraphManager.get_node_label_from_gamma(gamma=gamma) # [N,1]
            batch_label_list.append(batch_label)
        return batch_label_list