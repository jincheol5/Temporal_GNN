import wandb
import torch
import numpy as np
from tqdm import tqdm
from .model_train_utils import EarlyStopping
from .metrics import Metrics

class ModelTrainer:
    @staticmethod
    def train_old(model,train_data_loader,val_data_loader,config:dict):
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
        total_epoch_loss=[]
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

            total_epoch_loss.append(torch.stack(epoch_loss).mean().item())
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
        
        return total_epoch_loss

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
        loss_list=[]
        for epoch in tqdm(range(config['epochs']),desc=f"Training..."):
            model.train()
            
            label_list=[batch['label'] for batch in train_data_loader]
            label=torch.stack(label_list,dim=0) # [seq_len,batch_size,1]

            pred_logit=model(batch_list=train_data_loader,device=device) # [seq_len,batch_size,1]
            loss=Metrics.compute_tR_loss(logit=pred_logit,label=label)
            loss_list.append(loss)

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """
            Early stopping
            """
            if config['early_stop']:
                val_loss=loss.item()
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
                    f"loss":loss.item(),
                },step=epoch)
            
            """
            validate
            """
            ModelTrainer.test(model=model,graph_type='all',data_loader=val_data_loader,config=config)

        return loss_list

    @staticmethod
    def test(model,graph_type,data_loader,config):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        """
        model test
        """
        with torch.no_grad():
            label_list=[batch['label'] for batch in data_loader]
            label=torch.stack(label_list,dim=0) # [seq_len,batch_size,1]

            pred_logit=model(batch_list=data_loader,device=device) # [seq_len,batch_size,1]
            acc=Metrics.compute_tR_acc(logit=pred_logit,label=label)
        print(f"{graph_type} graph tR acc: {acc}")
        return acc
