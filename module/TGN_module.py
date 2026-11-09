import torch
import torch.nn as nn
from typing import Literal
from graph import GNN_Graph
from .time_module import TimeEncoder
from .mlp_module import MLP

class Memory(nn.Module):
    def __init__(self,
            n_node:int,
            mem_dim:int,
            edge_dim:int,
            time_dim:int
        ):
        super().__init__()
        self.n_node=n_node
        self.mem_dim=mem_dim
        self.edge_dim=edge_dim
        self.time_dim=time_dim

        # memory
        self.register_buffer("mem_vec",torch.zeros(self.n_node+1,self.mem_dim))
        self.register_buffer("mem_t",torch.zeros(self.n_node+1,dtype=torch.int64))

        # pre_msg_store
        self.register_buffer("pre_msg",torch.zeros(self.n_node+1,2*mem_dim+edge_dim+time_dim))
        self.register_buffer("pre_msg_t",torch.zeros(self.n_node+1,dtype=torch.int64))
        self.register_buffer("pre_msg_mask",torch.zeros(self.n_node+1,dtype=torch.bool))

        # init memory module
        self.init_memory()
        self.init_pre_msg_store()

    def init_memory(self):
        self.mem_vec.zero_()
        self.mem_t.zero_()

    def init_pre_msg_store(self):
        self.pre_msg.zero_()
        self.pre_msg_t.zero_()
        self.pre_msg_mask.zero_()

    def detach_memory(self):
        self.mem_vec.detach_()
        self.mem_t.detach_()

    def detach_pre_msg_store(self):
        self.pre_msg.detach_()
        self.pre_msg_t.detach_()
        self.pre_msg_mask.detach_()

    def update_memory_state(self,
            node:torch.Tensor,
            mem_vec:torch.Tensor,
            mem_t:torch.Tensor
        ):
        self.mem_vec[node]=mem_vec
        self.mem_t[node]=mem_t

    def update_pre_msg_store(self,
            node:torch.Tensor,
            pre_msg:torch.Tensor,
            pre_msg_t:torch.Tensor
        ):
        # init pre_msg store
        self.detach_pre_msg_store()
        self.init_pre_msg_store()
        
        # update pre_msg store
        self.pre_msg[node]=pre_msg
        self.pre_msg_t[node]=pre_msg_t
        self.pre_msg_mask[node]=True

    def get_pre_msg(self):
        return {
            "node":torch.where(self.pre_msg_mask)[0],
            "pre_msg":self.pre_msg[self.pre_msg_mask],
            "pre_msg_t":self.pre_msg_t[self.pre_msg_mask]
        }

class MemoryUpdater(nn.Module):
    def __init__(self,
            mem_dim:int,
            edge_dim:int,
            time_dim:int,
            msg_dim:int,
            time_encoder:TimeEncoder,
            aggr_fn:Literal["last","mean"]="last",
            msg_fn:Literal["concat","mlp"]="concat"
        ):
        super().__init__()
        self.mem_dim=mem_dim
        self.edge_dim=edge_dim
        self.time_dim=time_dim
        self.msg_dim=msg_dim
        self.aggr_fn=aggr_fn
        self.msg_fn=msg_fn

        # module
        self.time_encoder=time_encoder
        if msg_fn=="mlp":
            self.msg_mlp=MLP(
                input_dim=mem_dim+mem_dim+time_dim+edge_dim,
                latent_dim=msg_dim,
                output_dim=msg_dim
            )

    def set_graph(self,graph:GNN_Graph):
        self.graph=graph
        self.n_node=graph.n_node

    def get_aggregated_msg(self,
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
            mem_vec: [N+1,mem_dim]
            mem_t: [N+1,]
        """
        ### 0. device
        device=src.device

        ### 1. create raw msg
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
        ) # [B,2*mem_dim+time_dim+edge_dim]

        dst_raw_msg=torch.concat(
            [
                dst_mem,
                src_mem,
                dst_ts_vec,
                edge_ft
            ],
            dim=-1
        ) # [B,2*mem_dim+time_dim+edge_dim]

        ### 2. aggregate msg
        nodes=torch.concat([src,dst],dim=0) # [2B,]
        msgs=torch.concat([src_raw_msg,dst_raw_msg],dim=0) # [2B,2*mem_dim+time_dim+edge_dim]
        times=torch.concat([event_t,event_t],dim=0) # [2B,]

        # event time 순으로 오름차순 정렬
        sorted_idx=torch.argsort(times)
        nodes=nodes[sorted_idx]
        msgs=msgs[sorted_idx]
        times=times[sorted_idx]

        msg_dict={}
        for node,msg,t in zip(nodes,msgs,times):
            node=node.item()
            if node not in msg_dict:
                msg_dict[node]=[]
            msg_dict[node].append((msg,t))

        aggr_node=[]
        aggr_msg=[]
        aggr_event_t=[]
        match self.aggr_fn:
            case "last":
                for node in msg_dict.keys():
                    last_msg,last_t=msg_dict[node][-1]
                    aggr_node.append(node)
                    aggr_msg.append(last_msg)
                    aggr_event_t.append(last_t)
            case "mean":
                for node in msg_dict.keys():
                    msg_list=[msg for msg,_ in msg_dict[node]]
                    last_t=msg_dict[node][-1][1]
                    aggr_node.append(node)
                    aggr_msg.append(torch.mean(torch.stack(msg_list,dim=0),dim=0))
                    # interaction time은 가장 최근 시각 사용 (TGN github 공식구현 설정)
                    aggr_event_t.append(last_t)
        aggr_node=torch.tensor(aggr_node,device=device) # [unique_N,]
        aggr_msg=torch.stack(aggr_msg,dim=0) # [unique_N,2*mem_dim+time_dim+edge_dim]
        aggr_event_t=torch.stack(aggr_event_t,dim=0) # [unique_N,]
        return {
            "aggr_node":aggr_node,
            "aggr_msg":aggr_msg,
            "aggr_event_t":aggr_event_t
        }

    def update_memory(self,
            node:torch.Tensor,
            msg:torch.Tensor,
            msg_t:torch.Tensor,
            mem_vec:torch.Tensor
        ):
        """
        Input:
            node: [unique_N,]
            msg: [unique_N,2*mem_dim+time_dim+edge_dim]
            msg_t: [unique_N,]
            mem_vec: [N+1,mem_dim]
        """
        if self.msg_fn==f"mlp":
            msg=self.msg_mlp(msg) # -> [unique_N,msg_dim]
        updated_result=self.update_memory_implement(
            node=node,
            msg=msg,
            msg_t=msg_t,
            mem_vec=mem_vec
        )
        return updated_result

    def update_memory_implement(self,
            node,
            msg,
            msg_t,
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
            aggr_fn:Literal["last","mean"]="last",
            msg_fn:Literal["concat","mlp"]="concat"
        ):
        super(GRUMemoryUpdater,self).__init__(
            mem_dim=mem_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            msg_dim=msg_dim,
            time_encoder=time_encoder,
            aggr_fn=aggr_fn,
            msg_fn=msg_fn
        )
        if self.msg_fn=="concat":
            input_size=2*mem_dim+time_dim+edge_dim
        else:
            input_size=msg_dim
        self.memory_updater=nn.GRUCell(
            input_size=input_size,
            hidden_size=mem_dim
        )

    def update_memory_implement(self,
            node:torch.Tensor,
            msg:torch.Tensor,
            msg_t:torch.Tensor,
            mem_vec:torch.Tensor
        ):
        """
        Input:
            node: [unique_N,]
            msg: [unique_N,2*mem_dim+time_dim+edge_dim] or [unique_N,msg_dim]
            msg_t: [unique_N,]
            mem_vec: [N,mem_dim]
        Return:
            node: [unique_N,]
            mem_vec: [unique_N,mem_dim]
            mem_t: [unique_N,]
        """
        pre_mem_vec=mem_vec[node] # [unique_N,mem_dim]
        updated_mem_vec=self.memory_updater(msg,pre_mem_vec) # [unique_N,mem_dim]
        return {
            "node":node, # [unique_N]
            "mem_vec":updated_mem_vec, # [unique_N,mem_dim]
            "mem_t":msg_t # [unique_N,]
        }
