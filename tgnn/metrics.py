import torch
import torch.nn.functional as F

class Metrics:
    @staticmethod
    def compute_step_tR_loss(logit_list:list,label_list:list):
        """
        Input:
            logit: List of [B,1]
            label: List of [B,1]
        Output:
            loss scalar tensor: [] (0차원)
        """
        loss_list=[]
        for logit,label in zip(logit_list,label_list):
            step_loss=F.binary_cross_entropy_with_logits(
                logit,label.float(),reduction='mean'
            )
            loss_list.append(step_loss)
        loss=torch.stack(loss_list).mean()
        return loss

    @staticmethod
    def compute_last_tR_loss(logit:torch.Tensor,label:torch.Tensor):
        """
        Input:
            logit: [N,1]
            label: [N,1]
        Output:
            loss scalar tensor: [] (0차원)
        """
        loss_per_node=F.binary_cross_entropy_with_logits(
            logit,label.float(),reduction='none'
        ) # [N,1]
        loss=loss_per_node.mean()
        return loss

    @staticmethod
    def compute_step_tR_acc(logit_list:list,label_list:list):
        """
        Input:
            logit: List of [B,1]
            label: List of [B,1]
        Output:
            Accuracy
        """
        acc_list=[]
        for logit,label in zip(logit_list,label_list):
            pred=torch.sigmoid(logit) 
            pred_label=(pred>=0.5).float()   
            correct=(pred_label==label).float()
            step_acc=correct.mean()
            acc_list.append(step_acc)
        acc=torch.stack(acc_list).mean() 
        return acc.cpu().item()
    
    def compute_last_tR_acc(logit:torch.Tensor,label:torch.Tensor):
        """
        Input:
            logit: [N,1]
            label: [N,1]
        Output:
            Accuracy
        """
        pred=torch.sigmoid(logit) # [N,1]
        pred_label=(pred>=0.5).float() # [N,1]

        print(f"pred_label: {pred_label.squeeze(-1)}")
        print(f"label: {label.squeeze(-1)}")

        correct=(pred_label==label.float()).float() # [N,1]
        acc=correct.mean() # scalar []
        return acc.cpu().item()
