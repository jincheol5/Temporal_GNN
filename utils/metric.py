import torch

class Metric:
    @staticmethod
    def compute_accuracy(pred_logit:torch.Tensor,label:torch.Tensor):
        """
        Input:
            pred: [B,1]
            label: [B,1]
        """
        pred_logit=(torch.sigmoid(pred_logit)>=0.5).float()
        acc=(pred_logit==label.float()).float().mean().item()
        return acc