import torch

class Memory:
    """
    """
    def __init__(self,
            n_node:int,
            mem_dim:int,
        ):
        self.n_node=n_node
        self.mem_dim=mem_dim
        self.mem_ft=torch.zeros(
            (self.n_node+1,mem_dim),
            dtype=torch.float32,
        )
        self.last_update_t=torch.zeros(
            (self.n_node+1,),
            dtype=torch.float32,
        )

    def set_memory(self,
            mem_ft:torch.Tensor
        ):
        self.mem_ft=mem_ft

    def set_last_update_t(self,
            last_update_t:torch.Tensor
        ):
        self.last_update_t=last_update_t

    def get_memory(self,
            node:torch.Tensor
        ):
        """
        Input:
            node: [N,]
        Output:
            mem_ft: [N,mem_dim]
        """
        device=node.device
        return self.mem_ft[node.cpu()].to(device=device)

    def get_node_timespan(self,
            node:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        Input:
            node: [N,]
            event_t: [N,]
        Return:
            node_ts: [N,1]
        """
        device=node.device
        node=node.cpu()
        event_t=event_t.cpu()
        node_ts=torch.abs(
            event_t-self.last_update_t[node]
        )
        return node_ts.unsqueeze(-1).to(device=device) # [N,1]

    def update_memory(self,
            node:torch.Tensor,
            mem_ft:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        Batch 내의 N개의 node들의 새로운 memory와 last_update 시간 업데이트 

        Input:
            node: [N,]
            mem_ft: [N,mem_dim]
            event_t: [N,]
        """
        # update memory, last_update time
        self.mem_ft[node.cpu()]=mem_ft.detach().cpu()
        self.last_update_t[node.cpu()]=event_t.detach().cpu()
