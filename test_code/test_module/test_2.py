import argparse
import torch
from utils import DataUtils
from data import TemporalGraph,Memory
from module import TimeEncoder,IdentityEmbedding,TimeProjectionEmbedding,GraphSumEmbedding,GraphAttnEmbedding

"""
<< Test >> 
module.embed_module
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. module.embed_module.IdentityEmbedding
            """
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            node_dim=4
            mem_dim=4
            latent_dim=4
            output_dim=4
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"simple")
            graph=TemporalGraph(df=df,node_dim=node_dim)

            n_node=graph.get_num_node()
            memory=Memory(n_node=n_node,mem_dim=mem_dim)

            embed_module=IdentityEmbedding(
                node_dim=node_dim,
                mem_dim=mem_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                memory=memory
            )

            tar=torch.tensor([6,7,8],dtype=torch.long,device=device)
            updated_tar_ft=embed_module.compute_embedding(tar=tar)
            print(f"udpated_tar_ft using IdentityEmbedding:")
            print(updated_tar_ft)

        case 2:
            """
            Test. module.embed_module.TimeProjectionEmbedding
            """
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            node_dim=4
            mem_dim=4
            latent_dim=4
            output_dim=4
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"simple")
            graph=TemporalGraph(df=df,node_dim=node_dim)

            n_node=graph.get_num_node()
            memory=Memory(n_node=n_node,mem_dim=mem_dim)

            embed_module=TimeProjectionEmbedding(
                node_dim=node_dim,
                mem_dim=mem_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                memory=memory
            )

            tar=torch.tensor([6,7,8],dtype=torch.long,device=device)
            tar_t=torch.tensor([10.0,10.0,10.0],dtype=torch.float32,device=device)
            updated_tar_ft=embed_module.compute_embedding(tar=tar,tar_t=tar_t)
            print(f"udpated_tar_ft using TimeProjectionEmbedding:")
            print(updated_tar_ft)
        
        case 3:
            """
            Test. module.embed_module.GraphSumEmbedding
            """
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            node_dim=4
            mem_dim=4
            latent_dim=4
            output_dim=4
            
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"simple")
            graph=TemporalGraph(df=df,node_dim=node_dim)

            n_node=graph.get_num_node()
            memory=Memory(n_node=n_node,mem_dim=mem_dim)

            time_dim=4
            time_encoder=TimeEncoder(time_dim=time_dim)

            n_layer=3
            n_neighbor=3
            use_memory=True
            embed_module=GraphSumEmbedding(
                node_dim=node_dim,
                mem_dim=mem_dim,
                latent_dim=latent_dim, 
                output_dim=output_dim,
                time_dim=time_dim,
                graph=graph,
                memory=memory,
                n_layer=n_layer,
                use_memory=use_memory,
                time_encoder=time_encoder
            )

            tar=torch.tensor([6,7,8],dtype=torch.long,device=device)
            tar_t=torch.tensor([10.0,10.0,10.0],dtype=torch.float32,device=device)
            updated_tar_ft=embed_module.compute_embedding(tar=tar,tar_t=tar_t,n_layer=n_layer,n_neighbor=n_neighbor)
            print(f"udpated_tar_ft using GraphSumEmbedding:")
            print(updated_tar_ft)
        
        case 4:
            """
            Test. module.embed_module.GraphAttnEmbedding
            """
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            node_dim=4
            mem_dim=4
            latent_dim=4
            output_dim=4
            
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"simple")
            graph=TemporalGraph(df=df,node_dim=node_dim)

            n_node=graph.get_num_node()
            memory=Memory(n_node=n_node,mem_dim=mem_dim)

            time_dim=4
            time_encoder=TimeEncoder(time_dim=time_dim)

            n_layer=3
            n_head=4
            n_neighbor=3
            use_memory=True
            embed_module=GraphAttnEmbedding(
                node_dim=node_dim,
                mem_dim=mem_dim,
                latent_dim=latent_dim, 
                output_dim=output_dim,
                time_dim=time_dim,
                graph=graph,
                memory=memory,
                n_layer=n_layer,
                n_head=n_head,
                use_memory=use_memory,
                time_encoder=time_encoder
            )

            tar=torch.tensor([6,7,8],dtype=torch.long,device=device)
            tar_t=torch.tensor([10.0,10.0,10.0],dtype=torch.float32,device=device)
            updated_tar_ft=embed_module.compute_embedding(tar=tar,tar_t=tar_t,n_layer=n_layer,n_neighbor=n_neighbor)
            print(f"udpated_tar_ft using GraphAttnEmbedding:")
            print(updated_tar_ft)

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