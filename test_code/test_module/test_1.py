import argparse
import torch
from data import Memory
from module import TimeEncoder,GRUMemoryUpdater

"""
<< Test >> 
module.mem_module
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. module.mem_module.GRUMemoryUpdater
            """



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