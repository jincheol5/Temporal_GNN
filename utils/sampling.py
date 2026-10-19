import torch

class Sampling:
    @staticmethod
    def random_negative_sampling(
        src:torch.Tensor,
        dst:torch.Tensor,
        n_node:int,
        seed:int|None=None,
        bipartite:bool=False,
        max_u:int=None
    ):
        """
        Train 시 seed=None
        Val/Test 시 seed=int -> 같은 sample 재현

        Input:
            src: [B,]
            dst: [B,]
            n_node: int, node_id = 1 ~ n_node
            seed: random seed
            bipartite: bool,
            max_u: int
        Return:
            negative_dst: [B,]
        """
        device=src.device

