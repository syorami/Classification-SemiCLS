import argparse
import copy
import logging
import math
from operator import concat
import os
import pdb

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from loss import builder as loss_builder
from loss.soft_supconloss import SoftSupConLoss, SupConLoss

from .base_trainer import Trainer
from sklearn.mixture import GaussianMixture
from torchmetrics.functional.classification import accuracy


@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class FixMatch_SupCon(Trainer):
    """ FixMatch_SupCon """
    def __init__(self, cfg, device, all_cfg, **kwargs):
        super().__init__(cfg=cfg)

        self.all_cfg = all_cfg
        self.device = device
        if self.cfg.amp:
            from apex import amp
            self.amp = amp

        self.loss_x = loss_builder.build(cfg.loss_x)
        self.loss_u = loss_builder.build(cfg.loss_u)

        self.low_dim = all_cfg.model.low_dim

        self.loss_contrast = SupConLoss(temperature=self.cfg.temperature)

    def compute_loss(self,
                     data_x,
                     data_u,
                     model,
                     optimizer,
                     **kwargs):
        # make inputs
        inputs_x, targets_x = data_x
        inputs_x_w = inputs_x[0]

        inputs_u, targets_u = data_u
        inputs_u_w, inputs_u_s, inputs_u_s1 = inputs_u

        batch_size = inputs_x_w.shape[0]
        targets_x = targets_x.to(self.device)

        # inference once for all
        inputs = torch.cat([inputs_x_w, inputs_u_w, inputs_u_s, inputs_u_s1], dim=0).to(self.device)
        logits, features = model(inputs)

        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s, _ = logits[batch_size:].chunk(3)
        f_u_w, f_u_s1, f_u_s2 = features[batch_size:].chunk(3)
        feats_x = features[:batch_size]
        
        Lx = self.loss_x(logits_x.float(), targets_x, reduction='mean')
        probs_u_w = torch.softmax(logits_u_w.detach() / self.cfg.T, dim=-1)

        # pseudo label and scores for u_w
        max_probs, p_targets_u = torch.max(probs_u_w, dim=-1)
        mask = max_probs.ge(self.cfg.threshold).float()
        Lu = (self.loss_u(logits_u_s, p_targets_u, reduction='none') * mask).mean()

        # for supervised contrastive
        labels = p_targets_u
        features = torch.cat([f_u_s1.unsqueeze(1), f_u_s2.unsqueeze(1)], dim=1)
        
        # In case of early training stage, pseudo labels have low scores
        if labels.shape[0] != 0:
            Lcontrast = self.loss_contrast(features, labels=labels)
        else:
            Lcontrast = sum(features.view(-1, 1)) * 0

        loss = Lx + self.cfg.lambda_u * Lu + self.cfg.lambda_contrast * Lcontrast

        if hasattr(self, "amp"):
            with self.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        elif "SCALER" in kwargs and kwargs["SCALER"] is not None:
            kwargs['SCALER'].scale(loss).backward()
        else:
            loss.backward()
    
        # calculate pseudo label acc
        no_gts = (targets_u < 0).all()
        targets_u = targets_u.to(self.device)
        right_labels = (p_targets_u == targets_u).float() * mask
        pseudo_label_acc = right_labels.sum() / max(mask.sum(), 1.0)        

        loss_dict = {
            "loss": loss,
            "loss_x": Lx,
            "loss_u": Lu,
            "loss_contrast": Lcontrast,
            "mask_prob": mask.mean(),
            "pseudo_acc": pseudo_label_acc,
        }

        return loss_dict

