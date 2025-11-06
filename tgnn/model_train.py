import wandb
import torch
import numpy as np
from tqdm import tqdm
from .model_train_utils import EarlyStopping
from .metrics import Metrics

class ModelTrainer:
    @staticmethod
    def train(model,train_batch_loader_list,val_batch_loader_list=None,validate:bool=False,config:dict=None):
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
            model.train()
            loss_list=[]
            for train_batch_loader in train_batch_loader_list:
                label_list=[batch['label'] for batch in train_batch_loader]
                label=torch.stack(label_list,dim=0) # [seq_len,batch_size,1]
                label=label.to(device)

                pred_logit=model(batch_list=train_batch_loader,device=device) # [seq_len,batch_size,1]
                loss=Metrics.compute_tR_loss(logit=pred_logit,label=label)
                loss_list.append(loss)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss=torch.stack(loss_list).mean().item()
            """
            Early stopping
            """
            if config['early_stop']:
                val_loss=epoch_loss
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
                    f"loss":epoch_loss,
                },step=epoch)

            """
            validate
            """
            if validate:
                acc=ModelTrainer.test(model=model,batch_loader_list=val_batch_loader_list)
                print(f"{epoch+1} epoch tR validation acc: {acc}")

    @staticmethod
    def test(model,batch_loader_list):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        """
        model test
        """
        with torch.no_grad():
            acc_list=[]
            for batch_loader in batch_loader_list:
                label_list=[batch['label'] for batch in batch_loader]
                label=torch.stack(label_list,dim=0) # [seq_len,batch_size,1]
                label=label.to(device)

                pred_logit=model(batch_list=batch_loader,device=device) # [seq_len,batch_size,1]
                acc=Metrics.compute_tR_acc(logit=pred_logit,label=label)
                acc_list.append(acc)
        return torch.stack(acc_list).mean().item()
