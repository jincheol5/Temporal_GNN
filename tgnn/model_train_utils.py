import numpy as np
import networkx as nx
import torch
from .graph_utils import GraphUtils


class ModelTrainUtils:
    @staticmethod
    def get_batch_loader(graph:nx.DiGraph,source_id:int=0,batch_size:int=1):
        num_nodes=graph.number_of_nodes()
        df=GraphUtils.get_event_stream_df(graph=graph)
        batch_row_list=[df.iloc[i:i+batch_size] for i in range(0,len(df),batch_size)]

        batch_loader=[]

        gamma=GraphUtils.GraphAlgorithm.compute_tR_one_pass_step(graph=graph,source_id=source_id,init=True)
        for batch_row in batch_row_list:
            batch_node_raw_feature_list=[]
            batch_node_time_feature_list=[]
            batch_neighbor_mask_list=[]
            batch_target_node_list=[]
            batch_label_list=[]
            for row in batch_row.itertuples():
                edge_event=(row.src,row.tar,row.ts)
                
                node_raw_feature=GraphUtils.get_node_raw_feature(num_nodes=num_nodes,source_id=source_id) # [N,1]
                cur_node_time_feature=GraphUtils.get_node_time_feature(num_nodes=num_nodes,gamma=gamma) # [N,1]
                node_time_feature=torch.abs(cur_node_time_feature-row.ts) # [N,1]
                sub_edge_index=GraphUtils.get_sub_edge_index(df=df,edge_event=edge_event) # [2,sub_E]
                neighbor_mask=GraphUtils.get_neighbor_node_mask(edge_index=sub_edge_index,target_node=row.tar,num_nodes=num_nodes) # [N,]
                
                # gamma update
                gamma=GraphUtils.GraphAlgorithm.compute_tR_one_pass_step(graph=graph,source_id=source_id,edge_event=edge_event,gamma=gamma)

                batch_node_raw_feature_list.append(node_raw_feature)
                batch_node_time_feature_list.append(node_time_feature)
                batch_neighbor_mask_list.append(neighbor_mask)
                batch_target_node_list.append(row.tar)
                batch_label_list.append(gamma[row.tar][0])

            batch_node_raw_feature=torch.stack(batch_node_raw_feature_list,dim=0) # [batch_size,N,1]
            batch_node_time_feature=torch.stack(batch_node_time_feature_list,dim=0) # [batch_size,N,1]
            batch_neighbor_mask=torch.stack(batch_neighbor_mask_list,dim=0) # [batch_size,N,]
            batch_target_node=torch.tensor(batch_target_node_list,dtype=torch.long).unsqueeze(-1) # [batch_size,1]
            batch_label=torch.tensor(batch_label_list,dtype=torch.float32).unsqueeze(-1) # [batch_size,1]

            batch_dict={}
            batch_dict['x']=batch_node_raw_feature
            batch_dict['t']=batch_node_time_feature
            batch_dict['neighbor_mask']=batch_neighbor_mask
            batch_dict['target']=batch_target_node
            batch_dict['label']=batch_label

            batch_loader.append(batch_dict)
        
        return batch_loader

class EarlyStopping:
    def __init__(self,patience=1):
        self.patience=patience
        self.patience_count=0
        self.prev_loss=np.inf
        self.best_state=None
        self.early_stop=False
    def __call__(self,val_loss:float,model:torch.nn.Module):
        if self.prev_loss==np.inf:
            self.prev_loss=val_loss
            self.best_state={k: v.clone() for k,v in model.state_dict().items()}
            return None
        else:
            if not np.isfinite(val_loss):
                print(f"Loss is NaN or Inf!")
                self.early_stop=True
                model.load_state_dict(self.best_state)
                return model
            
            if self.prev_loss<=val_loss:
                self.patience_count+=1
                if self.patience<self.patience_count:
                    print(f"Loss increases during {self.patience_count} patience!")
                    self.early_stop=True
                    model.load_state_dict(self.best_state)
                    return model
            else:
                self.patience_count=0
                self.prev_loss=val_loss
                self.best_state={k: v.clone() for k,v in model.state_dict().items()}
                return None