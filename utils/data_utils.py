import os
import pickle
import pandas as pd
import networkx as nx
from typing_extensions import Literal


def convert_graph_to_df(graph:nx.DiGraph):
    """
    convert nx DiGraph to event stream df
    """
    df=pd.DataFrame({
        'src': pd.Series(dtype='int'),
        'tar': pd.Series(dtype='int'),
        'timestamp': pd.Series(dtype='float')
        })
    for u,v,data in graph.edges(data=True):
        time_list=data['t']
        for timestamp in time_list:
            df.loc[len(df)]=[u,v,timestamp]
    df=df.sort_values(['timestamp'])
    return df