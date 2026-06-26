import os
import argparse
import pandas as pd
import numpy as np
from utils import DataUtils

"""
<< Test >> 
utils.data_utils.DataUtils
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. DataUtils.preprocess_dataset
            """
            dataset_name=kwargs["dataset_name"]

            dataset_path=os.path.join("..","data","temporal_graph",dataset_name)
            raw_graph_df_path=os.path.join(dataset_path,f"ml_{dataset_name}.csv")
            raw_node_ft_np_path=os.path.join(dataset_path,f"ml_{dataset_name}_node.npy")
            raw_edge_ft_np_path=os.path.join(dataset_path,f"ml_{dataset_name}.npy")
            raw_graph_df=pd.read_csv(raw_graph_df_path,index_col=0)
            raw_node_ft_np=np.load(raw_node_ft_np_path)
            raw_edge_ft_np=np.load(raw_edge_ft_np_path)

            dataset=DataUtils.preprocess_dataset(dataset_name=dataset_name)
            graph_type=dataset["graph_type"]
            graph_df=dataset["graph_df"]
            node_ft_np=dataset["node_ft_np"]
            edge_ft_np=dataset["edge_ft_np"]
            max_u=dataset["max_u"]
            max_i=dataset["max_i"]

            print(f"dataset name: {dataset_name}")
            print(f"<< raw dataset info >>")
            print(f"number of nodes (include padding node): {pd.concat([raw_graph_df['u'],raw_graph_df['i']]).nunique()+1}")
            print(f"number of edge events: {len(raw_graph_df)}")
            print(f"node_ft shape: {raw_node_ft_np.shape}")
            print(f"edge_ft shape: {raw_edge_ft_np.shape}",end="\n\n")

            print(f"<< processed dataset info (remove self-loop)>>")
            print(f"number of nodes (include padding node): {pd.concat([graph_df['u'],graph_df['i']]).nunique()+1}")
            print(f"number of edge events: {len(graph_df)}")
            print(f"node_ft shape: {node_ft_np.shape}")
            print(f"edge_ft shape: {edge_ft_np.shape}")
            
            if graph_type=="homogeneous":
                print(f"dataset is homogeneous graph.")
            else:
                print(f"dataset is bipartite graph.")
                print(f"range of u is 0 to {max_u}")
                print(f"range of i is {max_u+1} to {max_i}")

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    parser.add_argument("--dataset_name",type=str,default=f"wikipedia")
    args=parser.parse_args()
    test_config={
        "test_num":args.test_num,
        "dataset_name":args.dataset_name
    }
    test_fn(**test_config)