import wandb
import torch
import numpy as np
from tqdm import tqdm
from .model import TGAT
from .model_train_utils import EarlyStopping
from .metrics import Metrics
from .data_utils import DataUtils

class ModelTrainer:
    @staticmethod
    def train(model,train_data_loader,val_data_loader,config:dict):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['lr']) if config['optimizer']=='adam' else torch.optim.SGD(model.parameters(),lr=config['lr'])

        """
        Early stopping
        """
        if config['early_stop']:
            early_stop=EarlyStopping(patience=config['patience'])

        """
        model train
        """
        for epoch in tqdm(range(config['epochs']),desc=f"Training..."):
            # wandb
            epoch_loss=[]
            model.train()
            for batch_dict in tqdm(train_data_loader,desc=f"Epoch {epoch+1}..."):
                batch_dict={k:v.to(device) for k,v in batch_dict.items()}
                x=batch_dict['x'] # [batch_size,N,1]
                t=batch_dict['t'] # [batch_size,N,1]
                neighbor_mask=batch_dict['neighbor_mask'] # [batch_size,N,]
                target=batch_dict['target'] # [batch_size,1]
                label=batch_dict['label'] # [batch_size,1]

                pred_logit=model(target=target,x=x,t=t,neighbor_mask=neighbor_mask) # [batch_size,1]

                loss=Metrics.compute_tR_loss(logit=pred_logit,label=label)

                # wandb
                epoch_loss.append(loss)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            """
            Early stopping
            """
            if config['early_stop']:
                val_loss=torch.stack(epoch_loss).mean().item()
                pre_model=early_stop(val_loss=val_loss,model=model)
                if early_stop.early_stop:
                    model=pre_model
                    print(f"Early Stopping in epoch {epoch+1}")
                    break
                    
            """
            wandb log
            """
            if config['wandb']:
                wandb.log({
                    f"loss":torch.stack(epoch_loss).mean().item(),
                },step=epoch)
            
            """
            validate
            """
            ModelTrainer.test(model=model,graph_type='all',data_loader=val_data_loader,config=config)

    @staticmethod
    def test(model,graph_type,data_loader,config):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        acc_list=[]

        """
        model test
        """
        with torch.no_grad():
            for batch_dict in tqdm(data_loader,desc=f"{config['mode']} {graph_type} graph..."):
                batch_dict={k:v.to(device) for k,v in batch_dict.items()}
                x=batch_dict['x'] # [batch_size,N,1]
                t=batch_dict['t'] # [batch_size,N,1]
                neighbor_mask=batch_dict['neighbor_mask'] # [batch_size,N,]
                target=batch_dict['target'] # [batch_size,1]
                label=batch_dict['label'] # [batch_size,1]

                pred_logit=model(target=target,x=x,t=t,neighbor_mask=neighbor_mask) # [batch_size,1]

                acc=Metrics.compute_tR_acc(logit=pred_logit,label=label)
                acc_list.append(acc)

        """
        acc
        """
        acc_mean=np.mean(acc_list)
        print(f"{graph_type} graph tR acc: {acc_mean}")

        return acc_mean