import argparse
import torch
from utils import DataUtils
from data import TemporalGraph

"""
<< Test >> 
data.temporal_graph.TemporalGraph
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. data.temporal_graph.TemporalGraph.get_num_node
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"simple")
            graph=TemporalGraph(df=df,node_dim=4,edge_dim=4)
            n_node=graph.get_num_node()
            print(f"number of nodes: {n_node}")
        case 2:
            """
            Test. data.temporal_graph.TemporalGraph.get_node_ft
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"simple")
            graph=TemporalGraph(df=df,node_dim=4,edge_dim=4)
            node=torch.tensor([1,2,3,4,5,6,7,8],dtype=torch.long)
            node_ft=graph.get_node_ft(node=node)
            print(f"All node feature:")
            print(node_ft)
        case 3:
            """
            Test. data.temporal_graph.TemporalGraph.get_temporal_neighbor
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"simple")
            graph=TemporalGraph(df=df,node_dim=4,edge_dim=4)

            tar=torch.tensor([1,2,3],dtype=torch.long)
            tar_t=torch.tensor([10.0,10.0,10.0],dtype=torch.float32)
            neighbor_id,neighbor_t,neighbor_ts,edge_id=graph.get_temporal_neighbor(
                tar=tar,
                tar_t=tar_t,
                n_neighbor=5
            )
            print(f"neighbor_id:")
            print(neighbor_id,end="\n\n")
            print(f"neighbor_t:")
            print(neighbor_t,end="\n\n")
            print(f"neighbor_ts:")
            print(neighbor_ts,end="\n\n")
            print(f"edge_id:")
            print(edge_id,end="\n\n")
        
        case 4:
            """
            Test. data.temporal_graph.TemporalGraph.get_historical_seq
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"simple")
            graph=TemporalGraph(df=df,node_dim=4,edge_dim=4)

            tar=torch.tensor([4,5,6],dtype=torch.long)
            tar_t=torch.tensor([10.0,10.0,10.0],dtype=torch.float32)
            seq_N,seq_E,seq_T=graph.get_historical_seq(node=tar,event_t=tar_t,seq_len=5)
            print(f"seq_N:")
            print(seq_N,end="\n\n")
            print(f"seq_E:")
            print(seq_E,end="\n\n")
            print(f"seq_T:")
            print(seq_T,end="\n\n")

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