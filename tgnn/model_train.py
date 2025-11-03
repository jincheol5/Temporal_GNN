import wandb
import torch
import numpy as np
from tqdm import tqdm
from .model_train_utils import EarlyStopping
from .metrics import Metrics

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
        loss_list=[]
        for epoch in tqdm(range(config['epochs']),desc=f"Training..."):
            model.train()
            
            label_list=[batch['label'] for batch in train_data_loader]
            label=torch.stack(label_list,dim=0) # [seq_len,batch_size,1]
            label=label.to(device)

            pred_logit=model(batch_list=train_data_loader,device=device) # [seq_len,batch_size,1]
            loss=Metrics.compute_tR_loss(logit=pred_logit,label=label)
            loss_list.append(loss.item())

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
            acc=ModelTrainer.test(model=model,data_loader=val_data_loader,config=config)
            print(f"{epoch+1} epoch tR validation acc: {acc}")
        return loss_list

    @staticmethod
    def test(model,data_loader,config):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        """
        model test
        """
        with torch.no_grad():
            label_list=[batch['label'] for batch in data_loader]
            label=torch.stack(label_list,dim=0) # [seq_len,batch_size,1]
            label=label.to(device)

            pred_logit=model(batch_list=data_loader,device=device) # [seq_len,batch_size,1]
            acc=Metrics.compute_tR_acc(logit=pred_logit,label=label)
        return acc
