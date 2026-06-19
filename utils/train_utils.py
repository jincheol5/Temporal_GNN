import pandas as pd
import torch
from torch.utils.data import Dataset

class TemporalGraphDataset(Dataset):
    def __init__(self,df:pd.DataFrame):
        self.src=torch.tensor(df["src"].values,dtype=torch.long)
        self.dst=torch.tensor(df["dst"].values,dtype=torch.long)
        self.edge=torch.tensor(df["edge_id"].values,dtype=torch.long)
        self.t=torch.tensor(df["t"].values,dtype=torch.float32)

    def __len__(self):
        return len(self.src)

    def __getitem__(self,idx):
        return self.src[idx],self.dst[idx],self.edge[idx],self.t[idx]

class TrainUtils:
    @staticmethod
    def split_graph_df(
            df:pd.DataFrame,
            train_ratio:float=0.7,
            val_ratio:float=0.1
        ):
        """
        Input:
            df
            train_ratio
            val_ratio
        Return:
            train_df
            val_df
            test_df
        """
        n=len(df)
        train_end=int(n*train_ratio)
        val_end=int(n*(train_ratio+val_ratio))
        train_df=df.iloc[:train_end].reset_index(drop=True)
        val_df=df.iloc[train_end:val_end].reset_index(drop=True)
        test_df=df.iloc[val_end:].reset_index(drop=True)
        return train_df,val_df,test_df

    @staticmethod
    def get_edge_label(
            pos_edge_size:int,
            neg_edge_size:int,
            device:torch.device
        ):
        """
        """
        pos_label=torch.ones(
            (pos_edge_size,1),
            device=device,
            dtype=torch.float32,
        ) # [B,1]
        neg_label=torch.zeros(
            (neg_edge_size,1),
            device=device,
            dtype=torch.float32,
        ) # [B,1]
        edge_label=torch.cat([pos_label,neg_label],dim=0) # [2B,1]
        return edge_label

    @staticmethod
    def padding_seq(
            src_seq_node:list,
            src_seq_edge:list,
            src_seq_ts:list,
            src_seq_co:list,
            dst_seq_node:list,
            dst_seq_edge:list,
            dst_seq_ts:list,
            dst_seq_co:list,
            n_patch:int,
            device:torch.device
        ):
        """
        Padding Sequence in DyGFormer

        Step:
            1. src_seq_node와 dst_seq_node, n_patch 정보로 max_seq_len 을 구한다.
                1-1. src_seq_node와 dst_seq_node에서 max_history_len 구한다.
                1-2. max_history_len이 n_patch의 배수가 아니라면, 
                    n_patch 배수가 되도록 증가시켜서 max_seq_len 구한다.

            2. padded tensor로 변환 후 device로 보낸다.
                2-1. 각 seq_node, seq_edge를 [B,max_seq_len] padded tensor로 변환
                2-2. 각 seq_ts를 [B,max_seq_len,1] padded tensor로 변환
                2-3. 각 seq_co를 [B,max_seq_len,2] padded tensor로 변환

        Input:
            src_seq_node: list of [history_len] arr
            src_seq_edge: list of [history_len] arr
            src_seq_ts: list of [history_len] arr
            src_seq_co: list of [history_len,2] tensor
            dst_seq_node: list of [history_len] arr
            dst_seq_edge: list of [history_len] arr
            dst_seq_ts: list of [history_len] arr
            dst_seq_co: list of [history_len,2] tensor
            n_patch: int
            device: torch.device
        Return:
            dict of padded tensor:
                src_seq_node: [B,max_seq_len]
                src_seq_edge: [B,max_seq_len]
                src_seq_ts: [B,max_seq_len,1]
                src_seq_co: [B,max_seq_len,2]
                dst_seq_node: [B,max_seq_len]
                dst_seq_edge: [B,max_seq_len]
                dst_seq_ts: [B,max_seq_len,1]
                dst_seq_co: [B,max_seq_len,2]
        """
        batch_size=len(src_seq_node)

        # 1. src/dst 전체에서 batch 내 최대 history 길이 계산
        max_history_len=0
        for b in range(batch_size):
            max_history_len=max(
                max_history_len,
                len(src_seq_node[b]),
                len(dst_seq_node[b])
            )
        
        # history가 모두 비어 있어도 최소 1은 유지
        max_seq_len=max(1,max_history_len)

        # 2. n_patch 배수로 맞춤
        if max_seq_len%n_patch!=0:
            max_seq_len+=n_patch-(max_seq_len%n_patch)
        
        # 3. padded tensor 생성
        padded_src_seq_node=torch.zeros(
            (batch_size,max_seq_len),
            dtype=torch.long,
            device=device
        )
        padded_src_seq_edge=torch.zeros(
            (batch_size,max_seq_len),
            dtype=torch.long,
            device=device
        )
        padded_src_seq_ts=torch.zeros(
            (batch_size,max_seq_len,1),
            dtype=torch.float32,
            device=device
        )
        padded_src_seq_co=torch.zeros(
            (batch_size,max_seq_len,2),
            dtype=torch.float32,
            device=device
        )
        padded_dst_seq_node=torch.zeros(
            (batch_size,max_seq_len),
            dtype=torch.long,
            device=device
        )
        padded_dst_seq_edge=torch.zeros(
            (batch_size,max_seq_len),
            dtype=torch.long,
            device=device
        )
        padded_dst_seq_ts=torch.zeros(
            (batch_size,max_seq_len,1),
            dtype=torch.float32,
            device=device
        )
        padded_dst_seq_co=torch.zeros(
            (batch_size,max_seq_len,2),
            dtype=torch.float32,
            device=device
        )

        # 4. 실제 sequence 채우기
        # DyGFormer 공식 코드 방식에 맞춰 left padding:
        # [0, 0, 0, history...]
        for b in range(batch_size):
            src_len=len(src_seq_node[b])
            dst_len=len(dst_seq_node[b])
            if src_len>0:
                src_start=max_seq_len-src_len
                padded_src_seq_node[b,src_start:]=torch.as_tensor(
                    src_seq_node[b],
                    dtype=torch.long,
                    device=device
                )
                padded_src_seq_edge[b,src_start:]=torch.as_tensor(
                    src_seq_edge[b],
                    dtype=torch.long,
                    device=device
                )
                padded_src_seq_ts[b,src_start:,0]=torch.as_tensor(
                    src_seq_ts[b],
                    dtype=torch.float32,
                    device=device
                )
                padded_src_seq_co[b,src_start:]=torch.as_tensor(
                    src_seq_co[b],
                    dtype=torch.float32,
                    device=device
                )
            if dst_len > 0:
                dst_start=max_seq_len-dst_len
                padded_dst_seq_node[b,dst_start:]=torch.as_tensor(
                    dst_seq_node[b],
                    dtype=torch.long,
                    device=device
                )
                padded_dst_seq_edge[b,dst_start:]=torch.as_tensor(
                    dst_seq_edge[b],
                    dtype=torch.long,
                    device=device
                )
                padded_dst_seq_ts[b,dst_start:,0]=torch.as_tensor(
                    dst_seq_ts[b],
                    dtype=torch.float32,
                    device=device
                )
                padded_dst_seq_co[b,dst_start:]=torch.as_tensor(
                    dst_seq_co[b],
                    dtype=torch.float32,
                    device=device
                )
        return {
            "src_seq_node": padded_src_seq_node,
            "src_seq_edge": padded_src_seq_edge,
            "src_seq_ts": padded_src_seq_ts,
            "src_seq_co": padded_src_seq_co,
            "dst_seq_node": padded_dst_seq_node,
            "dst_seq_edge": padded_dst_seq_edge,
            "dst_seq_ts": padded_dst_seq_ts,
            "dst_seq_co": padded_dst_seq_co
        }
    
    @staticmethod
    def patching_seq(
            seq_node_ft:torch.Tensor,
            seq_edge_ft:torch.Tensor,
            seq_ts_ft:torch.Tensor,
            seq_co_ft:torch.Tensor,
            n_patch:int
        ):
        """
        Patching Technique in DyGFormer

        Input:
            seq_node_ft: [B,max_seq_len,node_dim]
            seq_edge_ft: [B,max_seq_len,edge_dim]
            seq_ts_ft: [B,max_seq_len,time_dim]
            seq_co_ft: [B,max_seq_len,co_dim]
            n_patch: int
        Return:
            p=patch_size
            l=max_seq_len/patch_size
            dict:
                M_n: [B,l,node_dim x p]
                M_e: [B,l,edge_dim x p]
                M_t: [B,l,time_dim x p]
                M_c: [B,l,co_dim x p]
        """
        batch_size,max_seq_len,node_dim=seq_node_ft.size()
        edge_dim=seq_edge_ft.size(2)
        time_dim=seq_ts_ft.size(2)
        co_dim=seq_co_ft.size(2)

        l=int(max_seq_len//n_patch)
        M_n=seq_node_ft.reshape(
            batch_size,
            l,
            n_patch*node_dim
        ) # [B,l,node_dim x p]
        M_e=seq_edge_ft.reshape(
            batch_size,
            l,
            n_patch*edge_dim
        ) # [B,l,edge_dim x p]
        M_t=seq_ts_ft.reshape(
            batch_size,
            l,
            n_patch*time_dim
        ) # [B,l,time_dim x p]
        M_c=seq_co_ft.reshape(
            batch_size,
            l,
            n_patch*co_dim 
        ) # [B,l,co_dim x p]
        return {
            "M_n":M_n,
            "M_e":M_e,
            "M_t":M_t,
            "M_c":M_c
        }