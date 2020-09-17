#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 2020/09/17 | https://y-research.github.io

"""Description

"""

import torch
import torch.nn.functional as F

import batchminer
from .util import get_pairwise_stds, get_pairwise_similarity, dist

ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False


class Criterion(torch.nn.Module):

    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()

        self.name = 'listnet'
        # assert anchor_id in ANCHOR_ID

        self.opt = opt
        anchor_id = 'Anchor'
        self.anchor_id = anchor_id
        self.use_similarity = False #use_similarity

        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

        if 'Class' == anchor_id:
            assert 0 == self.opt.bs % self.opt.samples_per_class
            self.num_distinct_cls = int(self.opt.bs / self.opt.samples_per_class)


    def get_para_str(self):
        para_str = '_'.join(
            [self.name, self.anchor_id, 'Batch', str(self.opt.bs), 'Scls', str(self.opt.samples_per_class)])

        if self.use_similarity:
            para_str = '_'.join([para_str, 'Sim'])

        # else:
        #    if self.squared_dist:
        #        para_str = '_'.join([para_str, 'SqEuDist'])
        #    else:
        #        para_str = '_'.join([para_str, 'EuDist'])

        return para_str


    def forward(self, batch, labels, **kwargs):
        '''
        :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
        :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        :return:
        '''

        cls_match_mat = get_pairwise_stds(batch_labels=labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise

        if self.use_similarity:
            sim_mat = get_pairwise_similarity(batch_reprs=batch)
        else:
            dist_mat = dist(batch_reprs=batch, squared=False)  # [batch_size, batch_size], pairwise distances
            sim_mat = -dist_mat

        if 'Class' == self.anchor_id:  # vs. anchor wise sorting
            #cls_match_mat = cls_match_mat.view(self.num_distinct_cls, -1)
            #sim_mat = sim_mat.view(self.num_distinct_cls, -1)
            raise NotImplementedError


        # convert to one-dimension vector
        batch_size = batch.size(0)
        index_mat = torch.triu(torch.ones(batch_size, batch_size), diagonal=1) == 1
        sim_vec = sim_mat[index_mat]
        cls_vec = cls_match_mat[index_mat]

        # cross-entropy between two softmaxed vectors
        batch_loss = -torch.sum(F.softmax(sim_vec) * F.log_softmax(cls_vec))

        return batch_loss