import torch

class Memory:
    """
    """
    def __init__(self,
            num_node:int,
            mem_dim:int,
        ):
        self.num_node=num_node
        self.mem_dim=mem_dim
        self.mem_ft=torch.zeros(
            (self.num_node+1,mem_dim),
            dtype=torch.float32,
        )
        self.interact_t=torch.zeros(
            (self.num_node+1,),
            dtype=torch.float32,
        )
    
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

    def update_memory(self,
            node:torch.Tensor,
            mem_ft:torch.Tensor,
            interact_t:torch.Tensor
        ):
        """
        Batch 내의 N개의 node들의 새로운 memory와 interact 시간 업데이트 

        Input:
            node: [N,]
            mem_ft: [N,mem_dim]
            interact_t: [N,]
        """
        # update memory, interact time
        self.mem_ft[node.cpu()]=mem_ft.detach().cpu()
        self.interact_t[node.cpu()]=interact_t.detach().cpu()

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
        node_ts=torch.abs(
            self.interact_t[node.cpu()]-event_t
        )
        return node_ts.unsqueeze(-1).to(device=device) # [N,1]
