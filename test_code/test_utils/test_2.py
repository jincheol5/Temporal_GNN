import argparse
import torch
from utils import Sampling

"""
<< Test >> 
utils.sampling.Sampling
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. Sampling.random_negative_sampling
            """
            src=torch.tensor([1,2,3],dtype=torch.long)
            dst=torch.tensor([4,5,6],dtype=torch.long)
            n_node=10
            neg_dst=Sampling.random_negative_sampling(
                src=src,
                dst=dst,
                n_node=n_node
            )
            print(f"positive edges:")
            for i in range(src.size(0)):
                print(f"{src[i]} -> {dst[i]} at time {11+i}")
            print(f"\ncreated negative edges:")
            for i in range(src.size(0)):
                print(f"{src[i]} -> {neg_dst[i]} at time {11+i}")
        
        case 2:
            """
            Test. Sampling.random_negative_sampling
            bipartite graph case
            """
            src=torch.tensor([1,2,3],dtype=torch.long)
            dst=torch.tensor([11,12,13],dtype=torch.long)
            n_node=20
            max_u=9
            neg_dst=Sampling.random_negative_sampling(
                src=src,
                dst=dst,
                n_node=n_node,
                bipartite=True,
                max_u=max_u
            )
            print(f"positive edges:")
            for i in range(src.size(0)):
                print(f"{src[i]} -> {dst[i]} at time {11+i}")
            print(f"\ncreated negative edges:")
            for i in range(src.size(0)):
                print(f"{src[i]} -> {neg_dst[i]} at time {11+i}")

        case 3:
            """
            Test. Sampling.random_negative_sampling
            seed 고정 재현성 테스트
            """
            src=torch.tensor([1,2,3],dtype=torch.long)
            dst=torch.tensor([4,5,6],dtype=torch.long)
            n_node=10
            seed=42
            neg_dst_1=Sampling.random_negative_sampling(
                src=src,
                dst=dst,
                n_node=n_node,
                seed=seed
            )
            neg_dst_2=Sampling.random_negative_sampling(
                src=src,
                dst=dst,
                n_node=n_node,
                seed=seed
            )
            neg_dst_3=Sampling.random_negative_sampling(
                src=src,
                dst=dst,
                n_node=n_node,
                seed=seed
            )
            print(f"seed 고정 재현성 테스트")
            print(f"\ncreated negative edges 1:")
            for i in range(src.size(0)):
                print(f"{src[i]} -> {neg_dst_1[i]} at time {11+i}")
            print(f"\ncreated negative edges 2:")
            for i in range(src.size(0)):
                print(f"{src[i]} -> {neg_dst_2[i]} at time {11+i}")
            print(f"\ncreated negative edges 3:")
            for i in range(src.size(0)):
                print(f"{src[i]} -> {neg_dst_3[i]} at time {11+i}")

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    args=parser.parse_args()
    test_config={
        'test_num':args.test_num
    }
    test_fn(**test_config)