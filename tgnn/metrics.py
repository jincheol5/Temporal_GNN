import torch
import torch.nn.functional as F

class Metrics:
    @staticmethod
    def compute_step_tR_loss(logit:torch.Tensor,label:torch.Tensor):
        """
        Input:
            logit: [seq_len,batch_size,1]
            label: [seq_len,batch_size,1]
        Output:
            loss scalar tensor: [] (0차원)
        """
        loss_per_step=F.binary_cross_entropy_with_logits(
            logit,label.float(),reduction='none'
        ).mean(dim=(1,2))  # [seq_len]
        loss=loss_per_step.mean()
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
    def compute_step_tR_acc(logit:torch.Tensor,label:torch.Tensor):
        """
        Input:
            logit: [seq_len,B,1]
            label: [seq_len,B,1]
        Output:
            Accuracy
        """
        pred=torch.sigmoid(logit) # [seq_len,B,1]
        pred_label=(pred>=0.5).float() # [seq_len,B,1]

        correct=(pred_label==label.float()).float() # [seq_len,B,1]
        acc_per_step=correct.mean(dim=(1,2)) # [seq_len] 각 시점별 평균 정확도
        acc=acc_per_step.mean() # 전체 시퀀스 평균 정확도
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
        correct=(pred_label==label.float()).float() # [N,1]
        acc=correct.mean() # scalar []
        return acc.cpu().item()
