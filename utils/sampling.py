import numpy as np
import torch

class Sampling:
    @staticmethod
    def random_negative_sampling(
            src:torch.Tensor,
            dst:torch.Tensor,
            n_node:int
        ):
        """
        Input:
            src: [B,]
            dst: [B,]
            n_node: int, 전체 노드 수, node_id=1~N
        Return:
            negative_dst: [B,]
        """
        device=src.device
        dtype=src.dtype
        batch_size=src.size(0)

        negative_dst=torch.empty(batch_size,device=device,dtype=dtype)
        used=set()
        for i in range(batch_size):
            while True: # 조건을 만족하는 negative node가 나올 때까지 계속 랜덤 샘플링
                neg=torch.randint(
                    low=1,
                    high=n_node+1,
                    size=(1,),
                    device=device,
                    dtype=dtype,
                ).item()

                if (
                    neg!=src[i].item()      # 자기 자신 아님
                    and neg!=dst[i].item()  # positive dst 아님
                    and neg not in used     # negative끼리 중복 없음
                ):
                    negative_dst[i]=neg
                    used.add(neg)
                    break
        return negative_dst # [B,]