import numpy as np
import torch

class Sampler:
    """
    Train
    - train eventstream에서 등장한 dst들을 후보로 사용.  
    
    Val
    - train, val eventstream에서 등장한 dst들을 후보로 사용.  

    Test
    - all eventstream에서 등장한 dst들을 후보로 사용.
    """
    def __init__(self,
            dst_list:list,
            seed:int|None=None
        ):
        self.dst_candidate=np.unique(dst_list)
        self.set_random_state_seed(seed=seed)

    def set_random_state_seed(self,seed:int|None):
        """
        seed=None인 경우 OS에서 제공하는 엔트로피 등을 이용해 난수 생성기를 초기화
        -> 실행마다 다른 난수 시퀀스가 생성

        Test sampling 시 seed 고정.
        """
        self.seed=seed
        self.random_state=np.random.RandomState(seed) 

    def random_negative_sampling(self,
            src:torch.Tensor,
            dst:torch.Tensor
        ):
        """
        self.dst_candidate에서 neg_dst sampling 수행.
        sampling한 neg_dst[idx]가 
        -> src[idx]와 동일하지 않도록 sampling 수행.
        -> dst[idx]와 동일하지 않도록 sampling 수행.
        
        Input:
            src: [B,]
            dst: [B,]
        Return:
            neg_dst: [B,]
        """
        device=src.device
        src_np=src.detach().cpu().numpy()
        dst_np=dst.detach().cpu().numpy()

        # Negative sampling
        neg_dst=self.random_state.choice(
            self.dst_candidate,
            size=len(src_np),
            replace=True
        )

        # src 또는 dst와 동일한 경우 resampling (유효하지 않은 위치만 resampling)
        invalid_mask=(neg_dst==src_np)|(neg_dst==dst_np)
        while invalid_mask.any():
            neg_dst[invalid_mask]=self.random_state.choice(
                self.dst_candidate,
                size=invalid_mask.sum(),
                replace=True
            )
            invalid_mask=(neg_dst==src_np)|(neg_dst==dst_np)
        return torch.as_tensor(
            neg_dst,
            dtype=dst.dtype,
            device=device
        )




