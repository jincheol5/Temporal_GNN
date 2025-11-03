import torch
import torch.nn.functional as F

class Metrics:
    @staticmethod
    def compute_tR_loss(logit:torch.Tensor,label:torch.Tensor):
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
    def compute_tR_acc(logit:torch.Tensor,label:torch.Tensor):
        """
        Input:
            logit: [batch_size,1]
            label: [batch_size,1]
        Output:
            Accuracy
        """
        pred=torch.sigmoid(logit)  # [seq_len,batch_size,1]
        pred_label=(pred>=0.5).float()  # [seq_len,batch_size,1]

        correct=(pred_label==label.float()).float()  # [seq_len,batch_size,1]
        acc_per_step=correct.mean(dim=(1,2))  # [seq_len] 각 시점별 평균 정확도
        acc=acc_per_step.mean()  # 전체 시퀀스 평균 정확도
        return acc.cpu().item()