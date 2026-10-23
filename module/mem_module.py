import torch
import torch.nn as nn
from typing import Literal
from collections import defaultdict
from graph import GNN_Graph
from .time_module import TimeEncoder

class Memory(nn.Module):
    def __init__(self,
            n_node:int,
            mem_dim:int=32
        ):
        """
        set mem_ft and mem_t to zero vectors
            - mem_vec: memory vector
            - mem_t: last update time of memory 
        set raw_message_store
            - raw_msg_store: 
        """
        super().__init__()
        self.n_node=n_node
        self.mem_dim=mem_dim
        self.register_buffer(
            "mem_vec",
            torch.zeros(self.n_node+1,self.mem_dim),
        )
        self.register_buffer(
            "mem_t",
            torch.zeros(self.n_node+1,dtype=torch.int64),
        )
        self.raw_msg_store=defaultdict(list)
        self.init_memory_state()

    def init_memory_state(self):
        """
        initialize all the mem_ft and mem_t to zero vectors, which should be called at the start of each epoch
        """
        self.mem_vec.data.zero_()
        self.mem_t.data.zero_()

    def clean_raw_msg_store(self,node:torch.Tensor|None=None):
        if node is None:
            self.raw_msg_store=defaultdict(list)
        else:
            for node_id in node.detach().cpu().tolist():
                self.raw_msg_store.pop(node_id,None)

    def detach_memory(self):
        self.mem_vec.detach_()
        self.mem_t.detach_()

    def detach_raw_msg_store(self):
        for node,messages in self.raw_msg_store.items():
            self.raw_msg_store[node]=[
                (message.detach(),timestamp.detach())
                for message,timestamp in messages
            ]

    def store_raw_msg(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor,
            src_raw_msg:torch.Tensor,
            dst_raw_msg:torch.Tensor
        ):
        src_ids=src.detach().cpu().tolist()
        dst_ids=dst.detach().cpu().tolist()
        for idx,(src_id,dst_id) in enumerate(zip(src_ids,dst_ids)):
            self.raw_msg_store[src_id].append(
                (src_raw_msg[idx].clone(),event_t[idx].clone())
            )
            self.raw_msg_store[dst_id].append(
                (dst_raw_msg[idx].clone(),event_t[idx].clone())
            )

    def get_mem_vec(self,node:torch.Tensor|None=None):
        if node is None:
            return self.mem_vec
        else:
            return self.mem_vec[node]

    def get_mem_t(self,node:torch.Tensor|None=None):
        if node is None:
            return self.mem_t
        else:
            return self.mem_t[node]

    def get_node_timespan(self,
            node:torch.Tensor,
            event_t:torch.Tensor
        ):
        node_ts=torch.abs(event_t-self.mem_t[node])
        return node_ts.unsqueeze(-1) # [N,1]

    def update_memory_state(self,
            node:torch.Tensor,
            mem_vec:torch.Tensor,
            mem_t:torch.Tensor
        ):
        """
        Batch 내의 N개의 node들의 새로운 mem_vec와 mem_t 업데이트 
        """
        self.mem_vec[node]=mem_vec
        self.mem_t[node]=mem_t
        self.clean_raw_msg_store(node=node)

class MemoryUpdater(nn.Module):
    def __init__(self,
            mem_dim:int,
            edge_dim:int,
            time_dim:int,
            msg_dim:int,
            time_encoder:TimeEncoder,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super().__init__()
        self.mem_dim=mem_dim
        self.edge_dim=edge_dim
        self.time_dim=time_dim
        self.msg_dim=msg_dim
        self.msg_fn=msg_fn
        self.aggr_fn=aggr_fn

        # module
        self.time_encoder=time_encoder
        if msg_fn=="mlp":
            self.msg_mlp=nn.Sequential(
                nn.Linear(
                    in_features=mem_dim+mem_dim+time_dim+edge_dim,
                    out_features=msg_dim
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=msg_dim,
                    out_features=msg_dim
                )
            )

    def set_graph(self,graph:GNN_Graph):
        self.graph=graph
        self.n_node=graph.n_node

    def create_raw_msg(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            edge:torch.Tensor,
            event_t:torch.Tensor,
            mem_vec:torch.Tensor,
            mem_t:torch.Tensor
        ):
        """
        Input:
            src: [B,]
            dst: [B,]
            edge: [B,]
            event_t: [B,]
            mem_vec: [N,mem_dim]
            mem_t: [N,]
        Output:
            src_msg: [B,msg_dim]
            dst_msg: [B,msg_dim]
        """
        src_mem=mem_vec[src]
        src_ts=torch.abs(
            event_t-mem_t[src]
        ).unsqueeze(-1)
        src_ts_vec=self.time_encoder(src_ts)

        dst_mem=mem_vec[dst]
        dst_ts=torch.abs(
            event_t-mem_t[dst]
        ).unsqueeze(-1)
        dst_ts_vec=self.time_encoder(dst_ts)

        edge_ft=self.graph.get_edge_ft(edge=edge)

        src_raw_msg=torch.concat(
            [
                src_mem,
                dst_mem,
                src_ts_vec,
                edge_ft
            ],
            dim=-1
        ) # [B,mem_dim+mem_dim+time_dim+edge_dim]

        dst_raw_msg=torch.concat(
            [
                dst_mem,
                src_mem,
                dst_ts_vec,
                edge_ft
            ],
            dim=-1
        ) # [B,mem_dim+mem_dim+time_dim,edge_dim]
        return src_raw_msg,dst_raw_msg

    def aggregate_msg(self,
            raw_msg_store:dict[int,list[tuple[torch.Tensor,torch.Tensor]]]
        ):
        aggr_node=[]
        aggr_msg_vec=[]
        aggr_event_t=[]
        for node,messages in raw_msg_store.items():
            if not messages:
                continue
            raw_msgs=torch.stack([msg for msg,_ in messages],dim=0)
            times=torch.stack([t for _,t in messages],dim=0)
            msg_vec=self.msg_mlp(raw_msgs) if self.msg_fn=="mlp" else raw_msgs
            last_idx=torch.argsort(times,stable=True)[-1]
            aggr_node.append(int(node))
            aggr_msg_vec.append(
                msg_vec[last_idx] if self.aggr_fn=="last" else msg_vec.mean(dim=0)
            )
            aggr_event_t.append(times[last_idx])
        return {
            "aggr_node":torch.tensor(aggr_node,dtype=torch.long,device=aggr_msg_vec[0].device),
            "aggr_msg_vec":torch.stack(aggr_msg_vec,dim=0),
            "aggr_event_t":torch.stack(aggr_event_t,dim=0)
        }

    def update_memory(self,
            raw_msg_store:dict[int,list[tuple[torch.Tensor,torch.Tensor]]],
            mem_vec:torch.Tensor
        ):
        """
        자식 class의 update_memory 실행으로 자식 class에서 구현된 update_memory_implement 호출
        raw_msg_store가 비었을 경우 (첫 번째 batch) None 반환 
        
        Input:
            src: [B,]
            tar: [B,]
            edge: [B,]
            event_t: [B,]
            mem_vec: [N,mem_dim]
            mem_t: [N,]
        Return:
            node: [unique_N,]
            mem_vec: [unique_N,mem_dim]
            mem_t: [unique_N,]
        """
        if not raw_msg_store:
            return None

        aggr_msg=self.aggregate_msg(raw_msg_store=raw_msg_store)
        updated_result=self.update_memory_implement(
            aggr_node=aggr_msg["aggr_node"],
            aggr_msg_vec=aggr_msg["aggr_msg_vec"],
            aggr_event_t=aggr_msg["aggr_event_t"],
            mem_vec=mem_vec
        )
        return updated_result

    def update_memory_implement(self,
            aggr_node,
            aggr_msg_vec,
            aggr_event_t,
            mem_vec
        ):
        return NotImplemented

class GRUMemoryUpdater(MemoryUpdater):
    def __init__(self,
            mem_dim:int,
            edge_dim:int,
            time_dim:int,
            msg_dim:int,
            time_encoder:TimeEncoder,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super(GRUMemoryUpdater,self).__init__(
            mem_dim=mem_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            msg_dim=msg_dim,
            time_encoder=time_encoder,
            msg_fn=msg_fn,
            aggr_fn=aggr_fn
        )
        if self.msg_fn=="concat":
            input_size=mem_dim+mem_dim+time_dim+edge_dim
        else:
            input_size=msg_dim
        self.memory_updater=nn.GRUCell(
            input_size=input_size,
            hidden_size=mem_dim
        )

    def update_memory_implement(self,
            aggr_node:torch.Tensor,
            aggr_msg_vec:torch.Tensor,
            aggr_event_t:torch.Tensor,
            mem_vec:torch.Tensor
        ):
        """
        Input:
            aggr_node: [unique_N,]
            aggr_msg_vec: [unique_N,msg_dim]
            aggr_event_t: [unique_N,]
            mem_vec: [N,mem_dim]
        Return:
            node: [unique_N,]
            mem_vec: [unique_N,mem_dim]
            mem_t: [unique_N,]
        """
        pre_mem_vec=mem_vec[aggr_node] # [unique_N,mem_dim]
        updated_mem_vec=self.memory_updater(aggr_msg_vec,pre_mem_vec) # [unique_N,mem_dim]
        return {
            "node":aggr_node, # [unique_N]
            "mem_vec":updated_mem_vec, # [unique_N,mem_dim]
            "mem_t":aggr_event_t # [unique_N,]
        }
