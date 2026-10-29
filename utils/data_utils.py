import os
import pandas as pd
import numpy as np
from typing import Literal

BASE_PATH=os.path.join("..","data")
class DataUtils:
    @staticmethod
    def _preprocess_ZENODO_dataset(
            dataset_name:Literal[
                "enron",
                "wikipedia",
                "reddit"
            ]
        ):
        """
        ZENODO link: https://zenodo.org/records/8267830
        Dataset for "BenchTemp: A General Benchmark for Evaluating Temporal Graph Neural Networks"
        Bipartite graph dataset의 경우 u, i node ID 범위가 전처리 되어 있음 (max_u < min_i)

        1. Load graph_df, node_ft, edge_ft
        2. Check bipartite
        3. Preprocess timestamp to unix timestamp second 
        4. Remove self-loop, timestamp 기준으로 오름차순 정렬
        5. timestamp range를 0 ~ (max_t-min_t)로 상대적 조정
        6. Remap edge_id
        7. Remap node_id
        """
        ### 1. Load graph_df, node_ft, edge_ft
        graph_df_path=os.path.join(BASE_PATH,f"temporal_graph",dataset_name,f"ml_{dataset_name}.csv")
        node_ft_path=os.path.join(BASE_PATH,f"temporal_graph",dataset_name,f"ml_{dataset_name}_node.npy")
        edge_ft_path=os.path.join(BASE_PATH,f"temporal_graph",dataset_name,f"ml_{dataset_name}.npy")
        graph_df=pd.read_csv(
            graph_df_path,
            index_col=0,
        )[["u","i","ts","idx"]].rename(
            columns={"ts": "t"}
        )
        node_ft=np.load(node_ft_path)
        edge_ft=np.load(edge_ft_path)

        ### 2. Check bipartite using the preprocessed node ID ranges
        max_u=graph_df["u"].max()
        min_i=graph_df["i"].min()
        bipartite=bool(max_u<min_i)

        ### 3. Preprocess timestamp to Unix timestamp second
        graph_df["t"]=(
            pd.to_numeric(graph_df["t"],errors="raise")
            .astype(np.int64)
        )

        ### 4. Remove self-loop, timestamp 기준으로 오름차순 정렬 
        graph_df=(
            graph_df[graph_df["u"]!=graph_df["i"]]
            .sort_values(by="t",kind="stable")
            .reset_index(drop=True)
        )

        ### 5. timestamp range를 0 ~ (max_t-min_t)로 상대적 조정
        min_event_time=graph_df["t"].min()
        graph_df["t"]=(graph_df["t"]-min_event_time).astype(np.int64)

        ### 6. Remap edge id
        edge_ft=np.vstack([
            np.zeros(
                (1,edge_ft.shape[1]),
                dtype=edge_ft.dtype
            ),
            edge_ft[graph_df["idx"].to_numpy()]
        ])
        graph_df["idx"]=np.arange(1,len(graph_df)+1)

        ### 7. Remap node ID
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
            "node_dim": node_ft.shape[1],
            "edge_dim": edge_ft.shape[1],
            "node_ft": node_ft,
            "edge_ft": edge_ft,
            "bipartite": bipartite
        }

    @staticmethod
    def _preprocess_SNAP_dataset(
            dataset_name:Literal[
                "CollegeMsg",
                "soc-sign-bitcoinotc",
                "soc-sign-bitcoinalpha"
            ]
        ):
        """
        Stanford Network Analysis Project (SNAP) link: https://snap.stanford.edu/data
        Bipartite graph dataset의 경우 u, i node ID 범위가 전처리 되어 있지 않음.

        1. Check bipartite
        2. Load graph_df (node_ft, edge_ft if exists.)
        3. Preprocess bipartite graph dataset
        4. Preprocess timestamp to unix timestamp second 
        5. Remove self-loop, timestamp 기준으로 오름차순 정렬
        6. timestamp range를 0 ~ (max_t-min_t)로 상대적 조정
        7. Remap edge_id
        8. Remap node_id
        """
        bipartite_dataset=(
            "act-mooc",
        )
        ### 1. Check bipartite
        bipartite=dataset_name in bipartite_dataset

        ### 2. Load graph_df (node_ft, edge_ft if exists.)
        match dataset_name:
            case "CollegeMsg":
                dataset_path=os.path.join(BASE_PATH,f"temporal_graph",dataset_name,f"{dataset_name}.txt")
                graph_df=pd.read_csv(
                    dataset_path,
                    header=None,
                    sep=r"\s+",
                    usecols=[0,1,2],
                    names=["u","i","t"],
                )
                node_ft=None
                edge_ft=None
            case "soc-sign-bitcoinotc"|"soc-sign-bitcoinalpha":
                dataset_path=os.path.join(BASE_PATH,f"temporal_graph",dataset_name,f"{dataset_name}.csv")
                graph_df=pd.read_csv(
                    dataset_path,
                    header=None,
                    usecols=[0,1,2,3],
                    names=["u","i","edge_w","t"],
                )
                node_ft=None
                edge_ft=graph_df.pop("edge_w").to_numpy(dtype=np.float32).reshape(-1,1) # [E,1]
        graph_df["idx"]=np.arange(len(graph_df)) # 원본 edge feature의 행 위치를 보존 (패딩이 없으므로 0부터 시작)

        ### 3. Preprocess bipartite graph dataset

        ### 4. Preprocess timestamp to unix timestamp second
        graph_df["t"]=(
            pd.to_numeric(graph_df["t"],errors="raise")
            .astype(np.int64)
        )

        ### 5. Remove self-loop and timestamp 기준으로 오름차순 정렬
        graph_df=(
            graph_df[graph_df["u"]!=graph_df["i"]]
            .sort_values(by="t",kind="stable")
            .reset_index(drop=True)
        )

        ### 6. timestamp range를 0 ~ (max_t-min_t)로 상대적 조정
        min_event_time=graph_df["t"].min()
        graph_df["t"]=(graph_df["t"]-min_event_time).astype(np.int64)

        ### 7. Remap edge_id
        if edge_ft is not None:
            edge_ft=np.vstack([
                np.zeros((1,edge_ft.shape[1]),dtype=edge_ft.dtype),
                edge_ft[graph_df["idx"].to_numpy()]
            ])
        graph_df["idx"]=np.arange(1,len(graph_df)+1)

        ### 8. Remap node_id
        used_nodes=np.sort(np.unique(graph_df[["u","i"]].to_numpy()))
        node_map={
            old_id:new_id
            for new_id,old_id in enumerate(used_nodes,start=1)
        }
        graph_df[["u","i"]]=(
            graph_df[["u","i"]].replace(node_map).astype(int)
        )
        if node_ft is not None:
            node_ft=np.vstack([
                np.zeros((1,node_ft.shape[1]),dtype=node_ft.dtype),
                node_ft[used_nodes]
            ])
        return {
            "graph_df": graph_df,
            "node_dim": node_ft.shape[1] if node_ft is not None else 0,
            "edge_dim": edge_ft.shape[1] if edge_ft is not None else 0,
            "node_ft": node_ft,
            "edge_ft": edge_ft,
            "bipartite": bipartite
        }

    @staticmethod
    def preprocess_dataset(
            dataset_name:Literal[
                    # ZENODO
                    "enron",
                    "wikipedia",
                    "reddit",
                    # SNAP
                    "CollegeMsg",
                    "soc-sign-bitcoinotc",
                    "soc-sign-bitcoinalpha"
                ]
        ):
        ZENODO_dataset=(
            "enron",
            "wikipedia",
            "reddit"
        )
        SNAP_dataset=(
            "CollegeMsg",
            "soc-sign-bitcoinotc",
            "soc-sign-bitcoinalpha"
        )
        if dataset_name in ZENODO_dataset:
            return DataUtils._preprocess_ZENODO_dataset(dataset_name=dataset_name)
        if dataset_name in SNAP_dataset:
            return DataUtils._preprocess_SNAP_dataset(dataset_name=dataset_name)