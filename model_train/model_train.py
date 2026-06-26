import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import Sampling,TrainUtils,Metric

class ModelTrainer:
    @staticmethod
    def train_link_prediction(
            model:nn.Module,
            train_loader:DataLoader,
            val_loader:DataLoader,
            **kwargs
        ):
        """
        """
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=model.to(device)
        if kwargs["optimizer"]=="adam":
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=kwargs["lr"]
            )
        else:
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=kwargs["lr"]
            )
        
        """
        model train
        """
        n_node=model.graph.get_num_node()
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            model.train()
            batch_count=0
            for src,dst,event_t,_,edge in tqdm(
                    train_loader,
                    desc=f"Training epoch: {epoch+1}..."
                ):
                batch_count+=1

                src=src.to(device) # [B,]
                dst=dst.to(device) # [B,]
                edge=edge.to(device) # [B,]
                event_t=event_t.to(device) # [B,]

                # negative sampling
                if kwargs["bipartite"]:
                    neg_dst=Sampling.random_negative_sampling(
                        src=src,
                        dst=dst,
                        n_node=n_node,
                        bipartite=True,
                        u_max=kwargs["u_max"]
                    ) # [B,]
                else:
                    neg_dst=Sampling.random_negative_sampling(
                        src=src,
                        dst=dst,
                        n_node=n_node
                    ) # [B,]

                pos_event={
                    "src":src,
                    "dst":dst,
                    "edge":edge,
                    "event_t":event_t
                }
                neg_event={
                    "src":src,
                    "dst":neg_dst,
                    "edge":edge,
                    "event_t":event_t
                }

                ### label
                edge_label=TrainUtils.get_edge_label(
                    pos_edge_size=src.size(0),
                    neg_edge_size=neg_dst.size(0),
                    device=device
                ) # [2B,1]

                ### predict
                pred_edge_logit=model(
                    pos_event=pos_event,
                    neg_event=neg_event
                ) # [2B,1]

                ### Loss
                criterion=nn.BCEWithLogitsLoss()
                loss=criterion(pred_edge_logit,edge_label)
                print(f"{epoch+1} epoch {batch_count} batch_count loss: {loss.item()}")

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            """
            validate model
            """
            ModelTrainer.evaluate_link_prediction(model=model,data_loader=val_loader)
        return model

    @staticmethod
    def evaluate_link_prediction(
            model:nn.Module,
            data_loader:DataLoader,
            **kwargs
        ):
        """
        """
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=model.to(device)
        model.eval()

        """
        model evaluate
        """
        n_node=model.graph.get_num_node()
        acc_list=[]
        with torch.no_grad():
            for src,dst,edge,event_t in tqdm(
                    data_loader,
                    desc=f"Evaluating..."
                ):
                src=src.to(device) # [B,]
                dst=dst.to(device) # [B,]
                edge=edge.to(device) # [B,]
                event_t=event_t.to(device) # [B,]

                # negative sampling
                if kwargs["bipartite"]:
                    neg_dst=Sampling.random_negative_sampling(
                        src=src,
                        dst=dst,
                        n_node=n_node,
                        seed=kwargs["seed"]
                    ) # [B,]
                else:
                    neg_dst=Sampling.random_negative_sampling(
                        src=src,
                        dst=dst,
                        n_node=n_node,
                        seed=kwargs["seed"],
                        bipartite=True,
                        u_max=kwargs["u_max"]
                    ) # [B,]

                pos_event={
                    "src":src,
                    "dst":dst,
                    "edge":edge,
                    "event_t":event_t
                }
                neg_event={
                    "src":src,
                    "dst":neg_dst,
                    "edge":edge,
                    "event_t":event_t
                }

                ### label
                edge_label=TrainUtils.get_edge_label(
                    pos_edge_size=src.size(0),
                    neg_edge_size=neg_dst.size(0),
                    device=device
                ) # [2B,1]

                ### predict
                pred_edge_logit=model(
                    pos_event=pos_event,
                    neg_event=neg_event
                ) # [2B,1]

                ### compute ACC
                batch_acc=Metric.compute_accuracy(
                    pred_logit=pred_edge_logit,
                    label=edge_label
                )
                acc_list.append(batch_acc)
        print(f"ACC: {sum(acc_list)/len(acc_list)}")