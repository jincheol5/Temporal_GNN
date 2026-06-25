import os
import argparse
import pandas as pd
import numpy as np

"""
<< Test >> 
Check Dataset.
"""
def test_fn(**kwargs):
    dataset_name=kwargs["dataset_name"]
    dataset_path=os.path.join("..","data","temporal_graph",dataset_name)
    match kwargs['test_num']:
        case 1:
            """
            Test. Check "ml_dataset_name.csv" info. 
                columns: u, i, ts, label, idx
            """
            file_path=os.path.join(dataset_path,f"ml_{dataset_name}.csv")
            df=pd.read_csv(file_path)
            print(df.columns)
            print(df.head())

        case 2:
            """
            Test. Check node_id mapping in "ml_dataset_name.csv" 
            """
            file_path=os.path.join(dataset_path,f"ml_{dataset_name}.csv")
            df=pd.read_csv(file_path)
            nodes=sorted(set(df["u"]) | set(df["i"]))
            print(f"start node id: {nodes[0]}, end node id: {nodes[-1]}") # 1, N
            print(f"number of edge events: {len(df)}")
            print(f"len nodes: {len(nodes)}, max node id: {max(nodes)}") # 같으면 연속 ID
            print(f"node mapping is 1~N ?: {nodes==list(range(1,max(nodes)+1))}") # true이면 노드 ID가 1~N까지 빈 번호 없이 연속적으로 매핑된 것

        case 3:
            """
            Test. Check self-loop in "ml_dataset_name.csv" 
            """
            file_path=os.path.join(dataset_path,f"ml_{dataset_name}.csv")
            df=pd.read_csv(file_path)
            print(f"number of self-loop: {(df['u']==df['i']).sum()}")

        case 4:
            """
            Test. Check node_id mapping in 
                "ml_dataset_name.npy",
                "ml_dataset_name_node.npy"  
            """
            node_ft_path=os.path.join(dataset_path,f"ml_{dataset_name}_node.npy")
            edge_ft_path=os.path.join(dataset_path,f"ml_{dataset_name}.npy")
            node_ft_np=np.load(node_ft_path)
            edge_ft_np=np.load(edge_ft_path)
            print(f"node_ft_np shape: {node_ft_np.shape}") # (N+1,1), padding node 포함
            print(f"edge_ft_np shape: {edge_ft_np.shape}") # (E+1,1), padding edge 포함
            print(f"node[0]={node_ft_np[0]}")
            print(f"edge[0]={edge_ft_np[0]}")

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    parser.add_argument("--dataset_name",type=str,default=f"enron")
    args=parser.parse_args()
    test_config={
        "test_num":args.test_num,
        "dataset_name":args.dataset_name
    }
    test_fn(**test_config)