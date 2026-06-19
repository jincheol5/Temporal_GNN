import argparse
import torch
from torch.utils.data import DataLoader
from data import TemporalGraph
from utils import DataUtils,TemporalGraphDataset,TrainUtils

"""
<< Test >> 
utils.train_utils
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. train_utils.TemporalGraphDataset
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"CollegeMsg")
            dataset=TemporalGraphDataset(df=df)
            for idx in range(5):
                src,dst,edge,event_t=dataset.__getitem__(idx=idx)
                print(f"{idx} row values: {src} to {dst} at {event_t} (edge_id: {edge})")

        case 2:
            """
            Test. train_utils.TemporalGraphDataset using PyTorch DataLoader
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"CollegeMsg")
            dataset=TemporalGraphDataset(df=df)
            loader=DataLoader(
                dataset=dataset,
                batch_size=4,
                shuffle=False
            )
            for batch_idx,(src,dst,edge,event_t) in enumerate(loader):
                print(f"{batch_idx+1} batch values:")
                print(src)
                print(dst)
                print(edge)
                print(event_t)
                break
        
        case 3:
            """
            Test. train_utils.TrainUtils.split_graph_df
            """
        
        case 4:
            """
            Test. train_utils.TrainUtils.get_edge_label
            """

        case 5:
            """
            Test. train_utils.TrainUtils.get_padded_seq
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"simple")
            graph=TemporalGraph(df=df,node_dim=4,edge_dim=4)

            src=torch.tensor([2,3,4],dtype=torch.long)
            dst=torch.tensor([7,6,5],dtype=torch.long)
            event_t=torch.tensor([10.0,10.0,10.0],dtype=torch.float32)
            src_history=graph.get_historical_seq(node=src,event_t=event_t)
            dst_history=graph.get_historical_seq(node=dst,event_t=event_t)
            co=graph.get_co_occurrence(src_seq_node=src_history["seq_node"],dst_seq_node=dst_history["seq_node"])

            padded_seq=TrainUtils.get_padded_seq(
                src_seq_node=src_history["seq_node"],
                src_seq_edge=src_history["seq_edge"],
                src_seq_ts=src_history["seq_ts"],
                src_seq_co=co["src_seq_co"],
                dst_seq_node=dst_history["seq_node"],
                dst_seq_edge=dst_history["seq_edge"],
                dst_seq_ts=dst_history["seq_ts"],
                dst_seq_co=co["dst_seq_co"],
                n_patch=1,
                device=src.device
            )

            print(f"src_seq_node:")
            print(padded_seq["src_seq_node"],end="\n\n")
            print(f"src_seq_edge:")
            print(padded_seq["src_seq_edge"],end="\n\n")
            print(f"src_seq_ts:")
            print(padded_seq["src_seq_ts"],end="\n\n")
            print(f"src_seq_co:")
            print(padded_seq["src_seq_co"],end="\n\n")

            print(f"dst_seq_node:")
            print(padded_seq["dst_seq_node"],end="\n\n")
            print(f"dst_seq_edge:")
            print(padded_seq["dst_seq_edge"],end="\n\n")
            print(f"dst_seq_ts:")
            print(padded_seq["dst_seq_ts"],end="\n\n")
            print(f"dst_seq_co:")
            print(padded_seq["dst_seq_co"],end="\n\n")

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