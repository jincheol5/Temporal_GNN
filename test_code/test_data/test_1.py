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
            temporal_neighbor=graph.get_temporal_neighbor(
                tar=tar,
                tar_t=tar_t,
                n_neighbor=5
            )
            print(f"neighbor:")
            print(temporal_neighbor["neighbor"],end="\n\n")
            print(f"neighbor_t:")
            print(temporal_neighbor["neighbor_t"],end="\n\n")
            print(f"neighbor_ts:")
            print(temporal_neighbor["neighbor_ts"],end="\n\n")
            print(f"neighbor_edge:")
            print(temporal_neighbor["neighbor_edge"],end="\n\n")
        
        case 4:
            """
            Test. data.temporal_graph.TemporalGraph.get_historical_seq
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"simple")
            graph=TemporalGraph(df=df,node_dim=4,edge_dim=4)

            tar=torch.tensor([4,5,6],dtype=torch.long)
            tar_t=torch.tensor([10.0,10.0,10.0],dtype=torch.float32)
            historical_seq=graph.get_historical_seq(node=tar,event_t=tar_t)
            print(f"seq_node:")
            print(historical_seq["seq_node"],end="\n\n")
            print(f"seq_edge:")
            print(historical_seq["seq_edge"],end="\n\n")
            print(f"seq_ts:")
            print(historical_seq["seq_ts"],end="\n\n")
        
        case 5:
            """
            Test. data.temporal_graph.TemporalGraph.get_co_occurrence
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"simple")
            graph=TemporalGraph(df=df,node_dim=4,edge_dim=4)

            src=torch.tensor([2,3,4],dtype=torch.long)
            dst=torch.tensor([7,6,5],dtype=torch.long)
            event_t=torch.tensor([10.0,10.0,10.0],dtype=torch.float32)
            src_history=graph.get_historical_seq(node=src,event_t=event_t)
            dst_history=graph.get_historical_seq(node=dst,event_t=event_t)
            src_seq_node=src_history["seq_node"]
            dst_seq_node=dst_history["seq_node"]
            co=graph.get_co_occurrence(src_seq_node=src_seq_node,dst_seq_node=dst_seq_node)
            print(f"src_seq_co:")
            print(co["src_seq_co"],end="\n\n")
            print(f"dst_seq_co:")
            print(co["dst_seq_co"],end="\n\n")

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