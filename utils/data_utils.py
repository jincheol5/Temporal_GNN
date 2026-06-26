import os
import pandas as pd
import numpy as np
from typing import Literal

"""
To do list:
- negative sampling for bipartite graph
"""

class DataUtils:
    base_path=os.path.join("..","data","temporal_graph")

    @staticmethod
    def preprocess_ex_dataset(dataset_name:str):
        """
        Input:
            dataset_name: str
                - simple
                - CollegeMsg
        Return
            df: pd.DataFrame
        """
        dataset_path=os.path.join("dataset",dataset_name,f"{dataset_name}.txt") # ex
        u_list,i_list,ts_list,label_list,idx_list=[],[],[],[]
        with open(dataset_path) as f:
            for idx,line in enumerate(f):
                e=line.strip().split()
                u=int(e[0])
                i=int(e[1])
                ts=int(e[2])
                u_list.append(u)
                i_list.append(i)
                ts_list.append(ts)
                label_list.append(0)
                idx_list.append(idx)
        df=pd.DataFrame(
            {
                "u":u_list,
                "i":i_list,
                "ts":ts_list,
                "label":label_list,
                "idx":idx_list
            }
        )

        # remove self-loop and reindex edge_id
        df=df[df["u"]!=df["i"]].reset_index(drop=True)
        
        # remapping node_id to 1 ~ N
        unique_nodes=sorted(set(df["u"])|set(df["i"]))
        node_mapping={
            old_id:new_id
            for new_id,old_id in enumerate(unique_nodes,start=1)
        }
        df["u"]=df["u"].map(node_mapping)
        df["i"]=df["i"].map(node_mapping)
        
        # sort by time
        df=df.sort_values(
            by=["ts","idx"],
            kind="stable"
        ).reset_index(drop=True)
        
        # reindex edge_id
        df["idx"]=range(1,len(df)+1)
        return df

    @staticmethod
    def preprocess_dataset(dataset_name:Literal[
                "enron",
                "wikipedia",
                "reddit"
            ]
        ):
        """
        Temporal graph dataset
            - ml_dataset.csv
                - col: u,i,ts,label,idx
                - ts 기준 오름차순 정렬된 상태
                - self-loop 제거 필요
            - ml_dataset_node.npy, (N+1,node_dim)
            - ml_dataset.npy, (E+1,edge_dim)
                - self-loop 제거에 따라 edge_id remapping 필요

        Input:
            dataset_name: str
                Homogeneous graph
                - enron

                Bipartite graph
                - wikipedia
                - reddit
        Return
            dict:
                graph_type: homogeneous or bipartite
                max_u: int, max user node id
                max_i: int, max item node id
                graph_df: pd.DataFrame
                node_ft_np: np.array
                edge_ft_np: np.array
        """
        match dataset_name:
            case "enron":
                graph_type="homogeneous"
            case "wikipedia":
                graph_type="bipartite"

        dataset_path=os.path.join(DataUtils.base_path,dataset_name)
        graph_path=os.path.join(dataset_path,f"ml_{dataset_name}.csv")
        node_ft_path=os.path.join(dataset_path,f"ml_{dataset_name}_node.npy")
        edge_ft_path=os.path.join(dataset_path,f"ml_{dataset_name}.npy")

        graph_df=pd.read_csv(graph_path,index_col=0)
        node_ft_np=np.load(node_ft_path)
        edge_ft_np=np.load(edge_ft_path)

        ### remove self-loop and remapping edge_ft
        mask=graph_df["u"]!=graph_df["i"]
        new_graph_df=graph_df[mask].copy().reset_index(drop=True)

        ### remapping edge id
        # 기존 idx 기준으로 edge feature 선택
        old_edge_ids=new_graph_df["idx"].to_numpy()
        new_edge_ft_np=edge_ft_np[old_edge_ids]

        # padding edge feature 다시 추가
        new_edge_ft_np=np.vstack(
            [
                np.zeros((1,edge_ft_np.shape[1])),
                new_edge_ft_np
            ]
        )

        # idx 재부여
        new_graph_df["idx"]=np.arange(1,len(new_graph_df)+1)

        ### remapping node id
        used_node_ids=np.sort(
            pd.concat([new_graph_df["u"],new_graph_df["i"]])
            .unique()
            .astype(int)
        )
        node_id_map={
            old_id:new_id
            for new_id,old_id in enumerate(used_node_ids,start=1)
        }
        new_graph_df["u"]=new_graph_df["u"].map(node_id_map).astype(int)
        new_graph_df["i"]=new_graph_df["i"].map(node_id_map).astype(int)
        
        # Remap node features with the same node order
        new_node_ft_np=np.vstack([
            np.zeros((1,node_ft_np.shape[1]),dtype=node_ft_np.dtype),
            node_ft_np[used_node_ids]
        ])

        return {
            "graph_type":graph_type,
            "max_u":new_graph_df["u"].max(),
            "max_i":new_graph_df["i"].max(),
            "graph_df":new_graph_df,
            "node_ft_np":new_node_ft_np,
            "edge_ft_np":new_edge_ft_np
        }