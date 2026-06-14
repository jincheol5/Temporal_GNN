import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import Sampling

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
            for src,dst,event_t in tqdm(
                    train_loader,
                    desc=f"Training epoch: {epoch+1}..."
                ):
                batch_count+=1

                batch_size=src.size(0)
                src=src.to(device) # [B,]
                dst=dst.to(device) # [B,]
                event_t=event_t.to(device) # [B,]
                neg_dst=Sampling.random_negative_sampling(
                    src=src,
                    dst=dst,
                    n_node=n_node
                ) # [B,]
                src=torch.concat([src,src],dim=0) # [2B,]
                dst=torch.concat([dst,neg_dst],dim=0) # [2B,]
                event_t=torch.concat([event_t,event_t],dim=0) # [2B,]

                ### label
                pos_label=torch.ones(
                    (batch_size,1),
                    device=device,
                    dtype=torch.float32,
                ) # [B,1]
                neg_label=torch.zeros(
                    (batch_size,1),
                    device=device,
                    dtype=torch.float32,
                ) # [B,1]
                edge_label=torch.cat([pos_label,neg_label],dim=0) # [2B,1]

                ### predict
                pred_edge_logit=model(
                    src=src,
                    dst=dst,
                    event_t=event_t
                ) # [2B,1]

                ### Loss
                criterion=nn.BCEWithLogitsLoss()
                loss=criterion(pred_edge_logit,edge_label)
                print(f"{epoch+1} epoch {batch_count} batch_count loss: {loss.item()}")

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()