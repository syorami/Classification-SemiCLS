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
from loss.soft_supconloss import SoftSupConLoss

from .base_trainer import Trainer
from sklearn.mixture import GaussianMixture


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


class HyperMatch(Trainer):
    """ HyperMatch """
    def __init__(self, cfg, device, all_cfg, **kwargs):
        super().__init__(cfg=cfg)

        self.all_cfg = all_cfg
        self.device = device
        if self.cfg.amp:
            from apex import amp
            self.amp = amp

        self.loss_x = loss_builder.build(cfg.loss_x)
        self.loss_u = loss_builder.build(cfg.loss_u)

        self.gmm_thr = cfg.gmm_thr        
        self.low_dim = cfg.low_dim

        self.loss_contrast = SoftSupConLoss(temperature=self.cfg.temperature)
        self.feat_centroids = torch.zeros((self.num_classes, self.cfg.low_dim), device=self.device)

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
        
        cat_targets_x = concat_all_gather(targets_x)
        cat_feats_x = concat_all_gather(feats_x)
        for _cls in cat_feats_x.unique():
            cur_feat_centroid = cat_feats_x[cat_targets_x == _cls].mean(0).detach()
            self.feat_centroids[_cls] = self.feat_centroids[_cls] * 0.9 + 0.1 * cur_feat_centroid

        Lx = self.loss_x(logits_x, targets_x, reduction='mean')
        probs_u_w = torch.softmax(logits_u_w.detach() / self.cfg.T, dim=-1)

        # pseudo label and scores for u_w
        max_probs, p_targets_u = torch.max(probs_u_w, dim=-1)
        mask = max_probs.ge(self.cfg.threshold).float()
        Lu = (self.loss_u(logits_u_s, p_targets_u, reduction='none') * mask).mean()

        # for supervised contrastive
        labels = p_targets_u
        features = torch.cat([f_u_s1.unsqueeze(1), f_u_s2.unsqueeze(1)], dim=1)

        # compute cosine similarity
        cos_sim = lambda f, c: torch.cosine_similarity(f[..., None, :], c[None, ...], dim=2)
        feat_sims = cos_sim(f_u_w, self.feat_centroids.T)
        feat_labels = feat_sims.argmax(1)

        # compute contrastive loss
        epsilon = 1e-7
        labels = p_targets_u
        cat_feats = torch.cat([f_u_s1.unsqueeze(1), f_u_s2.unsqueeze(1)], dim=1)
        feat_sims = cos_sim(f_u_w, self.feat_centroids)
        dists = probs_u_w * feat_sims + epsilon
        dists = dists / dists.sum(1, keepdim=True)
        
        # gaussian mixture model
        gmm = GaussianMixture(n_components=2)
        dists_np = dists.cpu().detach().numpy()
        dists_np = np.sort(dists_np, axis=1)[:, ::-1]
        dists_np = dists_np[:, :20]
        
        gmm_labels = gmm.fit_predict(dists_np)
        gmm_probs = gmm.predict_proba(dists_np)
        
        # select sharp cluster
        _, top_var_idx = dists.var(1).topk(1)
        cls_idx = gmm_labels[top_var_idx]
        cls_probs = gmm_probs[:, cls_idx]
        
        sharp_idxs = np.where(cls_probs > self.cfg.gmm_threshold)[0]
        flat_idxs = np.where(cls_probs <= self.cfg.gmm_threshold)[0]
        
        topk_probs, topk_labels = dists.topk(self.cfg.topk, dim=1)
        extend_feats = torch.cat((cat_feats, cat_feats[flat_idxs].repeat(self.cfg.topk - 1, 1, 1)))
        extend_probs = torch.cat((topk_probs[:, 0], topk_probs[flat_idxs][:, 1:].permute(1, 0).reshape(-1)))
        extend_labels = torch.cat((topk_labels[:, 0], topk_labels[flat_idxs][:, 1:].permute(1, 0).reshape(-1)))
        
        # In case of early training stage, pseudo labels have low scores
        if labels.shape[0] != 0:
            Lcontrast = self.loss_contrast(extend_feats, extend_probs, extend_labels)
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

