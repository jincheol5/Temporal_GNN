import networkx as nx
import numpy as np
import argparse
from tgnn import DataUtils,GraphAnalysis

def app_analysis(config:dict):
    match config['app_num']:
        case 0:
            """
            App 0.
            """
            print(f"<<{config['mode']}_{config['num_nodes']}_{config['graph_type']} graphs statistics result>>")
            
            data_dict=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}_parallel_{graph_type}",path='trne',dir_type=config['mode'])
            data_list=[]
            for _,src_dict in data_dict.items():
                for _,data in src_dict.items():
                    data_list.append(data)

            reachability_ratio_list=[]
            for data in data_list:
                reachability_ratio_list.append(GraphAnalysis.check_reachability_ratio(r=data.r[-1]))
            lst=np.array(reachability_ratio_list,dtype=float)
            mean_ratio=lst.mean()
            max_ratio=lst.max()
            min_ratio=lst.min()
            print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs tR ratio mean: {mean_ratio} max: {max_ratio} min: {min_ratio}")
            print()

        case 1:
            """
            App 1.
            Check statistics of num_nodes,num_edge_events,num_static_edges after remove self-loop
            """
            print(f"<<{config['mode']}_{config['num_nodes']} graphs statistics result>>")
            graph_list_dict=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}",path='trne',dir_type="graph")
            for graph_type,graph_list in graph_list_dict.items():
                N_list=[]
                E_s_list=[]
                E_list=[]
                for graph in graph_list:
                    graph.remove_edges_from(nx.selfloop_edges(graph))
                    N,E_s,E=GraphAnalysis.check_elements(graph=graph)
                    N_list.append(N)
                    E_s_list.append(E_s)
                    E_list.append(E)
                print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs mean of num_nodes: {np.mean(N_list)}")
                print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs mean of num_static_edgs: {np.mean(E_s_list)}")
                print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs mean of num_edge_events: {np.mean(E_list)}")
                print()
        case 2:
            """
            App 2. 
            Analysis ratio of temporal reachability
            """
            print(f"<<{config['mode']}_{config['num_nodes']}_parallel all graphs tR ratio result>>")
            data_list_dict={}
            graph_type_list=['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman']
            for graph_type in graph_type_list:
                data_list=[]
                data_dict=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}_parallel_{graph_type}",path='trne',dir_type=config['mode'])
                for _,src_dict in data_dict.items():
                    for _,data in src_dict.items():
                        data_list.append(data)
                data_list_dict[graph_type]=data_list

            for graph_type,data_list in data_list_dict.items():
                reachability_ratio_list=[]
                for data in data_list:
                    reachability_ratio_list.append(GraphAnalysis.check_reachability_ratio(r=data.r[-1]))
                lst=np.array(reachability_ratio_list,dtype=float)
                mean_ratio=lst.mean()
                max_ratio=lst.max()
                min_ratio=lst.min()
                print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs tR ratio mean: {mean_ratio} max: {max_ratio} min: {min_ratio}")
                print()

if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--graph_type",type=str,default="ladder")
    parser.add_argument("--mode",type=str,default="train")
    parser.add_argument("--num_nodes",type=int,default=20)
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        "graph_type":args.graph_type,
        "mode":args.mode,
        "num_nodes":args.num_nodes
    }
    app_analysis(config=config)