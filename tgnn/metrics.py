import torch
import torch.nn.functional as F

class Metrics:
    @staticmethod
    def compute_tR_loss(logit:torch.Tensor,label:torch.Tensor):
        """
        Input:
            logit: [batch_size,1]
            label: [batch_size,1]
        Output:
            loss scalar tensor: [] (0차원)
        """
        label=label.float()
        loss=F.binary_cross_entropy_with_logits(logit,label)
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
        pred=torch.sigmoid(logit)
        pred_label=(pred>=0.5).float()
        correct=(pred_label==label.float()).sum()
        acc=correct/label.size(0)
        return acc