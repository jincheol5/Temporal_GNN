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
            Test. DataUtils.preprocess_dataset_to_df
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