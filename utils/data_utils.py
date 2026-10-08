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
    def _preprocess_zenodo_dataset(
            dataset_name:Literal[
                    "enron",
                    "wikipedia",
                    "reddit"
                ]
        ):
        """
        timestamp: UTC unix timestamp day 단위로 변환
        """
        graph_path=os.path.join(BASE_PATH,f"temporal_graph",dataset_name,f"ml_{dataset_name}.csv")
        node_ft_path=os.path.join(BASE_PATH,f"temporal_graph",dataset_name,f"ml_{dataset_name}_node.npy")
        edge_ft_path=os.path.join(BASE_PATH,f"temporal_graph",dataset_name,f"ml_{dataset_name}.npy")

        graph_df=pd.read_csv(
            graph_path,
            index_col=0,
        )[["u","i","ts","idx"]].rename(
            columns={"ts": "t"}
        )

        node_ft=np.load(node_ft_path)
        edge_ft=np.load(edge_ft_path)

        # timestamp(초)를 Unix timestamp의 일(day) 단위 정수로 변환
        graph_df["t"]=(
            pd.to_numeric(graph_df["t"],errors="raise")
            .floordiv(24*60*60)
            .astype(np.int64)
        )

        # 최초 event 시각을 0으로 맞춘 상대 Unix timestamp(day)로 변환
        min_event_time=graph_df["t"].min()
        graph_df["t"]=(graph_df["t"]-min_event_time).astype(np.int64)

        ### remove self-loop, 동일한 시각의 동일한 방향 edge(u -> i)는 하나만 유지
        graph_df=(
            graph_df[graph_df["u"]!=graph_df["i"]]
            .drop_duplicates(subset=["u","i","t"],keep="first")
            .reset_index(drop=True)
        )

        ### remap edge index
        edge_ft=np.vstack([
            np.zeros(
                (1,edge_ft.shape[1]),
                dtype=edge_ft.dtype
            ),
            edge_ft[graph_df["idx"].to_numpy()]
        ])
        graph_df["idx"]=np.arange(1,len(graph_df)+1)

        # remap node id
        used_nodes=np.sort(np.unique(graph_df[["u","i"]].to_numpy()))
        node_map={
            old_id:new_id
            for new_id,old_id
            in enumerate(used_nodes,start=1)
        }
        graph_df[["u","i"]]=(
            graph_df[["u","i"]]
            .replace(node_map)
            .astype(int)
        )
        node_ft=np.vstack([
            np.zeros(
                (1,node_ft.shape[1]),
                dtype=node_ft.dtype
            ),
            node_ft[used_nodes]
        ])

        return {
            "graph_df": graph_df,
            "n_node": len(used_nodes),
            "max_u": graph_df["u"].max(),
            "node_dim": node_ft.shape[1],
            "edge_dim": edge_ft.shape[1],
            "node_ft": node_ft,
            "edge_ft": edge_ft
        }

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
                graph_df: pd.DataFrame
                n_node: int
                node_dim: int
                edge_dim: int
                node_ft_np: np.array
                edge_ft_np: np.array
                bipartite: homogeneous or bipartite
                max_u: int, max user node id
        """
        match dataset_name:
            case "enron":
                bipartite=False
            case "wikipedia":
                bipartite=True

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

        # get n_node, node_dim, edge_dim
        n_node=max(new_graph_df["u"].max(),new_graph_df["i"].max())
        node_dim=new_node_ft_np.shape[1]
        edge_dim=new_edge_ft_np.shape[1]

        return {
            "graph_df":new_graph_df,
            "n_node":n_node,
            "node_dim":node_dim,
            "edge_dim":edge_dim,
            "node_ft_np":new_node_ft_np,
            "edge_ft_np":new_edge_ft_np,
            "bipartite":bipartite,
            "max_u":new_graph_df["u"].max()
        }