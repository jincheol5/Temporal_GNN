import torch
import pandas as pd
import numpy as np
from collections import defaultdict

class TemporalGraph:
    """
    node_id=0: padding node
    edge_id=0: padding edge
    """
    def __init__(self,
            graph_df:pd.DataFrame,
            node_ft_df:pd.DataFrame=None,
            edge_ft_df:pd.DataFrame=None,
            node_dim:int=32,
            edge_dim:int=32
        ):
        """
        Input:
            graph_df: pd.DataFrame, sorted by event time
            node_ft_df: pd.DataFrame (node_id,ft_n...), 1 부터 n_node 까지 연속적으로 오름차순 정렬
            edge_ft_df: pd.DataFrame (edge_id,ft_e...), 1 부터 n_edge 까지 연속적으로 오름차순 정렬
            node_dim: int
            edge_dim: int
        """
        # set adj, adj_t
        self.adj=defaultdict(list)
        self.adj_t=defaultdict(list)
        for event in graph_df.itertuples(index=False):
            src=int(event.src)
            dst=int(event.dst)
            t=float(event.t)
            edge_id=int(event.edge_id)
            # edge 양방향 저장
            self.adj[dst].append((src,edge_id))
            self.adj_t[dst].append(t)
            self.adj[src].append((dst,edge_id))
            self.adj_t[src].append(t)

        # set n_node, n_edge
        self.n_node=max(graph_df["src"].max(),graph_df["dst"].max())
        self.n_edge=graph_df["edge_id"].max()

        # set node_ft
        if node_ft_df is None:
            self.set_node_ft(node_dim=node_dim)
        else:
            self.set_node_ft(node_ft_df=node_ft_df)
        
        # set edge_ft
        if edge_ft_df is None:
            self.set_edge_ft(edge_dim=edge_dim)
        else:
            self.set_edge_ft(edge_ft_df=edge_ft_df)

    def set_node_ft(self,
            node_ft_df:pd.DataFrame=None,
            node_dim:int=32
        ):
        """
        Input:
            node_ft_df: DataFrame (node_id,ft_n...), 1 부터 n_node 까지 연속적으로 오름차순 정렬
            n_node: int
            node_dim: int
        """
        if node_ft_df is None:
            self.node_dim=node_dim
            self.node_ft=torch.zeros(
                (self.n_node+1,node_dim),
                dtype=torch.float32
            )
        else:
            ft_col=node_ft_df.columns.drop("node_id")
            self.node_dim=len(ft_col)
            self.node_ft=torch.zeros(
                (self.n_node+1,node_dim),
                dtype=torch.float32
            )
            self.node_ft[1:]=torch.tensor(
                node_ft_df[ft_col].to_numpy(),
                dtype=torch.float32
            )

    def set_edge_ft(self,
            edge_ft_df:pd.DataFrame=None,
            edge_dim:int=32
        ):
        """
        Input:
            edge_ft_df: DataFrame (edge_id,ft_e...), 1 부터 n_edge 까지 연속적으로 오름차순 정렬
            n_edge: int
            edge_dim: int
        """
        if edge_ft_df is None:
            self.edge_dim=edge_dim
            self.edge_ft=torch.zeros(
                (self.n_edge+1,edge_dim),
                dtype=torch.float32
            )
        else:
            ft_col=edge_ft_df.columns.drop("edge_id")
            self.edge_dim=len(ft_col)
            self.edge_ft=torch.zeros(
                (self.n_edge+1,edge_dim),
                dtype=torch.float32
            )
            self.edge_ft[1:]=torch.tensor(
                edge_ft_df[ft_col].to_numpy(),
                dtype=torch.float32
            )

    def set_graph(self,
            graph_df:pd.DataFrame,
            node_ft_df:pd.DataFrame=None,
            edge_ft_df:pd.DataFrame=None,
            node_dim:int=32,
            edge_dim:int=32
        ):
        """
        Input:
            graph_df: pd.DataFrame, sorted by event time
            node_ft_df: pd.DataFrame (node_id,ft_n...), 1 부터 n_node 까지 연속적으로 오름차순 정렬
            edge_ft_df: pd.DataFrame (edge_id,ft_e...), 1 부터 n_edge 까지 연속적으로 오름차순 정렬
            node_dim: int
            edge_dim: int
        """
        # set adj, adj_t
        self.adj=defaultdict(list)
        self.adj_t=defaultdict(list)
        for event in graph_df.itertuples(index=False):
            src=int(event.src)
            dst=int(event.dst)
            t=int(event.t)
            edge_id=int(event.edge_id)
            self.adj[dst].append((src,edge_id))
            self.adj_t[dst].append(t)

        # set n_node, n_edge
        self.n_node=max(graph_df["src"].max(),graph_df["dst"].max())
        self.n_edge=graph_df["edge_id"].max()

        # set node_ft
        if node_ft_df is None:
            self.set_node_ft(node_dim=node_dim)
        else:
            self.set_node_ft(node_ft_df=node_ft_df)
        
        # set edge_ft
        if edge_ft_df is None:
            self.set_edge_ft(edge_dim=edge_dim)
        else:
            self.set_edge_ft(edge_ft_df=edge_ft_df)

    def get_num_node(self):
        return self.n_node

    def get_num_edge(self):
        return self.n_edge

    def get_node_ft(self,node:torch.Tensor=None):
        """
        Input:
            node: [B,]
        Return:
            node_ft
        """
        device=node.device
        return self.node_ft.to(device=device)[node]

    def get_edge_ft(self,edge:torch.Tensor=None):
        """
        Input:
            edge: [E,]
        Return:
            edge_ft
        """
        device=edge.device
        return self.edge_ft.to(device=device)[edge]

    def get_temporal_neighbor(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor,
            n_neighbor:int
        ):
        """
        Input:
            tar: [B,]
            tar_t: [B,]
            num_neighbor: int
            seed: int
        Return:
            neighbor: [B,num_neighbor]
            neighbor_t: [B,num_neighbor]
            neighbor_ts: [B,num_neighbor]
            neighbor_edge: [B,num_neighbor]
        """
        device=tar.device
        batch_size=tar.size(0)
        neighbor=torch.zeros(
            (batch_size,n_neighbor),
            dtype=torch.long,
            device=device
        )
        neighbor_t=torch.zeros(
            (batch_size,n_neighbor),
            dtype=torch.float,
            device=device
        )
        neighbor_ts=torch.zeros(
            (batch_size,n_neighbor),
            dtype=torch.float,
            device=device
        )
        neighbor_edge=torch.zeros(
            (batch_size,n_neighbor),
            dtype=torch.long,
            device=device
        )
        for b in range(batch_size):
            tar_id=int(tar[b].item())
            cut_time=float(tar_t[b].item())

            neighbors=self.adj.get(tar_id,[])
            times=self.adj_t.get(tar_id,[])

            if len(neighbors)==0:
                continue

            # t < cut_time 인 마지막 위치까지 선택
            times_np=np.asarray(times,dtype=np.float32)
            cut_idx=np.searchsorted(
                times_np,
                cut_time,
                side="left"
            )

            # 최근 num_neighbor개만 선택
            start_idx=max(0,cut_idx-n_neighbor)
            selected_neighbors=neighbors[start_idx:cut_idx]
            selected_times=times[start_idx:cut_idx]

            # 앞은 0 padding, 뒤에 실제 neighbor 저장
            offset=n_neighbor-len(selected_neighbors)
            for idx,((src,e_id),t) in enumerate(zip(selected_neighbors,selected_times)):
                neighbor[b,offset+idx]=src
                neighbor_t[b,offset+idx]=t
                neighbor_ts[b,offset+idx]=abs(cut_time-t)
                neighbor_edge[b,offset+idx]=e_id
        return {
            "neighbor":neighbor,
            "neighbor_t":neighbor_t,
            "neighbor_ts":neighbor_ts,
            "neighbor_edge":neighbor_edge
        }
    
    def get_historical_seq(self,
            node:torch.Tensor,
            event_t:torch.Tensor,
            max_n_neighbor:int
        ):
        """
        Input:
            node: [B,]
            event_t: [B,]
        Return:
            seq_node: list of [history_len] arr
            seq_edge: list of [history_len] arr
            seq_ts: list of [history_len] arr
        """
        seq_node=[]
        seq_edge=[]
        seq_ts=[]
        node_np=node.detach().cpu().numpy()
        event_t_np=event_t.detach().cpu().numpy()

        for node_id,cut_time in zip(node_np,event_t_np):
            node_id=int(node_id)
            cut_time=float(cut_time)

            neighbors=self.adj.get(node_id,[])
            times=self.adj_t.get(node_id,[])

            if len(neighbors)==0:
                seq_node.append(np.array([],dtype=np.longlong))
                seq_edge.append(np.array([],dtype=np.longlong))
                seq_ts.append(np.array([],dtype=np.float32))
                continue

            times_np=np.asarray(times,dtype=np.float32)

            # t < cut_time 인 interaction만 선택
            cut_idx=np.searchsorted(
                times_np,
                cut_time,
                side="left"
            )

            selected_neighbors=neighbors[:cut_idx]
            selected_times=times_np[:cut_idx]
            history_nodes=np.array(
                [src for src,_ in selected_neighbors],
                dtype=np.longlong
            )
            history_edges=np.array(
                [edge_id for _,edge_id in selected_neighbors],
                dtype=np.longlong
            )
            history_ts=(
                cut_time-selected_times
            ).astype(np.float32)

            if len(history_nodes)>max_n_neighbor:
                history_nodes=history_nodes[-max_n_neighbor:]
                history_edges=history_edges[-max_n_neighbor:]
                history_ts=history_ts[-max_n_neighbor:]

            seq_node.append(history_nodes)
            seq_edge.append(history_edges)
            seq_ts.append(history_ts)
        return {
            "seq_node": seq_node,
            "seq_edge": seq_edge,
            "seq_ts": seq_ts
        }

    def get_co_occurrence(self,
            src_seq_node:list,
            dst_seq_node:list,
        ):
        """
        DyGFormer Neighbor Co-occurrence

        Input:
            src_seq_node: list of [history_len] arr
            dst_seq_node: list of [history_len] arr
        Return:
            src_seq_co: list of [src_history_len,2] tensor
            dst_seq_co: list of [dst_history_len,2] tensor
        """
        src_seq_co=[]
        dst_seq_co=[]
        for src_ids,dst_ids in zip(src_seq_node,dst_seq_node):
            src_unique,src_inverse,src_counts=np.unique(
                src_ids,
                return_inverse=True,
                return_counts=True
            )
            dst_unique,dst_inverse,dst_counts=np.unique(
                dst_ids,
                return_inverse=True,
                return_counts=True
            )
            # src neighbor node가 src sequence에서 등장한 횟수
            src_neighbor_count_in_src_seq=torch.from_numpy(
                src_counts[src_inverse]
            ).float()  # [src_history_len]

            # dst neighbor node가 dst sequence에서 등장한 횟수
            dst_neighbor_count_in_dst_seq=torch.from_numpy(
                dst_counts[dst_inverse]
            ).float()  # [dst_history_len]

            src_mapping=dict(zip(src_unique,src_counts))
            dst_mapping=dict(zip(dst_unique,dst_counts))

            # src neighbor node가 dst sequence에서 등장한 횟수
            src_neighbor_count_in_dst_seq=torch.tensor(
                [dst_mapping.get(x,0) for x in src_ids],
                dtype=torch.float32
            )

            # dst neighbor node가 src sequence에서 등장한 횟수
            dst_neighbor_count_in_src_seq=torch.tensor(
                [src_mapping.get(x,0) for x in dst_ids],
                dtype=torch.float32
            )

            src_seq_co.append(
                torch.stack(
                    [
                        src_neighbor_count_in_src_seq,
                        src_neighbor_count_in_dst_seq
                    ],
                    dim=1
                )
            )  # [src_history_len,2]

            dst_seq_co.append(
                torch.stack(
                    [
                        dst_neighbor_count_in_src_seq,
                        dst_neighbor_count_in_dst_seq
                    ],
                    dim=1
                )
            )  # [dst_history_len,2]

        return {
            "src_seq_co": src_seq_co, # list of [src_history_len,2]
            "dst_seq_co": dst_seq_co # list of [dst_history_len,2]
        }