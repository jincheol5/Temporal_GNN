import os
import pandas as pd

class DataUtils:
    base_path=os.path.join("..","data","temporal_gnn")
    @staticmethod
    def preprocess_dataset_to_df(dataset_name:str):
        """
        Input:
            dataset_name: str
        Return
            df: pd.DataFrame
        """
        dataset_path=os.path.join("dataset",dataset_name,f"{dataset_name}.txt") # ex

        src_list,dst_list,t_list,edge_id_list=[],[],[],[]
        with open(dataset_path) as f:
            for idx,line in enumerate(f):
                e=line.strip().split()
                src=int(e[0])
                dst=int(e[1])
                t=int(e[2])
                src_list.append(src)
                dst_list.append(dst)
                t_list.append(t)
                edge_id_list.append(idx)
        df=pd.DataFrame(
            {
                "src":src_list,
                "dst":dst_list,
                "t":t_list,
                "edge_id":edge_id_list
            }
        )

        # remove self-loop and reindex edge_id
        df=df[df["src"]!=df["dst"]].reset_index(drop=True)
        
        # remapping node_id to 1 ~ N
        unique_nodes=sorted(set(df["src"])|set(df["dst"]))
        node_mapping={
            old_id:new_id
            for new_id,old_id in enumerate(unique_nodes,start=1)
        }
        df["src"]=df["src"].map(node_mapping)
        df["dst"]=df["dst"].map(node_mapping)
        
        # sort by time
        df=df.sort_values(
            by=["t","edge_id"],
            kind="stable"
        ).reset_index(drop=True)
        
        # reindex edge_id
        df["edge_id"]=range(1,len(df)+1)
        return df