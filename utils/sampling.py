import torch

class Sampling:
    @staticmethod
    def random_negative_sampling(
        src:torch.Tensor,
        dst:torch.Tensor,
        n_node:int,
        seed:int|None=None,
        bipartite:bool=False,
        u_max:int=None
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
            u_max: int
            i_max: int
        Return:
            negative_dst: [B,]
        """
        device=src.device
        dtype=src.dtype
        batch_size=src.size(0)

        # local random generator
        generator=None
        if seed is not None:
            generator=torch.Generator(device=device)
            generator.manual_seed(seed)

        negative_dst=torch.empty(
            batch_size,
            device=device,
            dtype=dtype,
        )

        # sampling range 설정
        if bipartite:
            low=u_max+1
        else:
            low=1
        high=n_node+1

        used=set()
        for i in range(batch_size):
            while True:
                neg=torch.randint(
                    low=low,
                    high=high,
                    size=(1,),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                ).item()

                if (
                    neg!=src[i].item() # 자기 자신 아님
                    and neg!=dst[i].item() # positive dst 아님
                    and neg not in used # negative끼리 중복 없음
                ):
                    negative_dst[i]=neg
                    used.add(neg)
                    break
        return negative_dst