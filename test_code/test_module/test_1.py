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
            n_node=5
            mem_dim=4
            msg_dim=4
            time_dim=4
            memory=Memory(n_node=n_node,mem_dim=mem_dim)
            time_encoder=TimeEncoder(time_dim=time_dim)
            memory_updater=GRUMemoryUpdater(
                mem_dim=mem_dim,
                msg_dim=msg_dim,
                time_dim=time_dim,
                time_encoder=time_encoder,
                memory=memory
            )
            node=torch.tensor([0,1,2,3,4],dtype=torch.long)

            print(f"previous memory:")
            pre_mem_ft=memory.get_memory(node=node)
            print(pre_mem_ft)
            print(f"device: {pre_mem_ft.device}",end="\n\n")

            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            memory_updater=memory_updater.to(device=device)
            src=torch.tensor([0,1,2],dtype=torch.long,device=device)
            dst=torch.tensor([2,3,4],dtype=torch.long,device=device)
            event_t=torch.tensor([1.0,2.0,3.0],dtype=torch.float32,device=device)
            updated_mem_ft=memory_updater.update_memory(
                src=src,
                dst=dst,
                event_t=event_t
            )

            print(f"updated memory:")
            print(updated_mem_ft)
            print(f"device: {updated_mem_ft.device}",end="\n\n")

            print(f"check upudated memory in Memory data:")
            updated_mem_ft_in_Memory=memory.get_memory(node=node)
            print(updated_mem_ft_in_Memory)
            print(f"device: {updated_mem_ft_in_Memory.device}")


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