#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import time

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest

        # a = xyz[batch_indices,:,:]       ###########################
        # b = xyz[batch_indices,farthest,:]   #######################
        # centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)   ################%%%%%%%%%%%%%%%%%%
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)   ######################################
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    dist, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return dist, group_idx

def convert_polar(neighbours, center):

    neighbours = neighbours.permute(0,2,3,1).contiguous()
    center = center.permute(0,2,3,1).contiguous()

    rel_x = (neighbours - center)[:,:,:,0]
    rel_y = (neighbours - center)[:,:,:,1]
    rel_z = (neighbours - center)[:,:,:,2]

    r_xy = torch.sqrt(rel_x**2 + rel_y**2)
    r_zx = torch.sqrt(rel_z**2 + rel_x**2)
    r_yz = torch.sqrt(rel_y**2 + rel_y**2)

    ### Z_axis
    z_beta = torch.atan2(rel_z, r_xy).unsqueeze(-3)
    z_alpha = torch.atan2(rel_y, rel_x).unsqueeze(-3)

    ### Y_axis
    y_beta = torch.atan2(rel_y, r_zx).unsqueeze(-3)
    y_alpha = torch.atan2(rel_x, rel_z).unsqueeze(-3)

    ### X_axis
    x_beta = torch.atan2(rel_x, r_yz).unsqueeze(-3)
    x_alpha = torch.atan2(rel_z, rel_y).unsqueeze(-3)

    return x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class LocalAggregation(nn.Module):
    def __init__(self, in_c, out_c, sample_num_list):
        super(LocalAggregation, self).__init__()


        self.sample_num_list = sample_num_list
        # self.feature_mlp = SharedMLP(in_c *2, in_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        # self.final_feature_mlp = SharedMLP(in_c*3, in_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        # self.coord_mlp = SharedMLP(10, in_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.coord_pool = AttentivePooling(in_c, in_c, in_c//4)

        # self.coord_mlp2 = SharedMLP(10, in_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.coord_pool2 = AttentivePooling(in_c, in_c, in_c//4)

        # self.feature_pool = AttentivePooling(in_c, in_c, in_c//4)
        # self.feature_pool2 = AttentivePooling(d1, d1, d1 // 4)
        self.mlp = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.mlp1 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.mlp2 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.mlp3 = SharedMLP(13, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.mlp4 = SharedMLP(in_c + 13, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.short_cut = SharedMLP(in_c, in_c, bn=True)

        # self.local1 = BasicBlock(in_c*2+1, in_c)
        # self.local2 = BasicBlock(in_c*2+1, in_c)



        # self.conv_blocks = nn.ModuleList()
        # self.bn_blocks = nn.ModuleList()
        # for i in range(len(sample_num_list)):
        #     convs = nn.ModuleList()
        #     bns = nn.ModuleList()
        #     # last_channel = in_channel + 3
        #     # for out_channel in sample_num_list[i]:
        #     convs.append(nn.Conv2d(in_c * 2, in_c, 1))
        #     bns.append(nn.BatchNorm2d(in_c))
        #     # last_channel = out_channel
        #     self.conv_blocks.append(convs)
        #     self.bn_blocks.append(bns)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, features, xyz=None):

        # idx = knn_idx
        # dist = knn_dist
        # residual = features

        features = features.permute(0, 2, 1).contiguous() ## B N C
        # upper_coords = upper_coords.permute(0, 2, 1)  ## B N C
        # coords = coords.permute(0, 2, 1)  ## B N C


        S = features.size(2)
        # features = features.unsqueeze(-1)
        # features = features.unsqueeze(-1)
        # if final == True:
        #     upper_features = upper_features.permute(0, 2, 1, 3)
        # else:
        # features = features.permute(0, 2, 1)  ### B N C

        # features = features.permute(0, 2, 1)  ### B N C

        # final_feature = []




        # if coords.size(1) != 1024:
        #     features = index_points(upper_features, coords_idx)
        # else:
        #     features = coords_feature.permute(0, 2, 1)

        # dist, idx = knn_point(self.sample_num_list[0], coords, coords)  ### B N K

        # coords_concat_list = []
        # features_concat_list = []

        for i, sample_num in enumerate(self.sample_num_list):

            dist, idx = knn_point(sample_num, features, features)  ### B N K

            # a = knn(features, sample_num)
            B, N, K = idx.size()

            ## neighbor coords aggregation
            # neighbors_coord = index_points(coords, idx)  ### use upper branch xyz to aggregate  (B N K C)
            # neighbors_coord = neighbors_coord.permute(0, 3, 1, 2)  ### B C N K
            # extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)  ## B 3 N K
            # coords_concat = torch.cat((extended_coords, neighbors_coord, extended_coords-neighbors_coord, dist.unsqueeze(-3)),dim=-3)  # B 10 N K
            #
            # coords_concat_list.append(coords_concat)
            # neighbors_coord = index_points(features, idx)
            # neighbors_coord = neighbors_coord.permute(0, 3, 1, 2)  ### B C N K
            # features_centre = features.transpose(-2, -1).unsqueeze(-1).expand(B, S, N, K)
            # coords_concat = torch.cat((neighbors_coord - features_centre, features_centre), dim=1)

            # coords_concat = self.coord_mlp(coords_concat)

            if S == 3:
                features_centre = features.transpose(-2, -1).unsqueeze(-1).expand(B, S, N, K).contiguous()

                neighbors_features = index_points(features, idx)
                neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()

                x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta = convert_polar(neighbors_features, features_centre)
                features = torch.cat((neighbors_features - features_centre, features_centre, x_alpha, x_beta, y_alpha,
                                      y_beta, z_alpha, z_beta, dist.unsqueeze(-3).contiguous()), dim=1)
                # features_concat = torch.cat((neighbors_features - features_centre, features_centre, dist.unsqueeze(-3)), dim=1)
                # features_concat_list.append(features_concat)

                features = self.mlp(features)
                features = torch.max(features, 3)[0]
            elif xyz is not None:
                ###################  xyz
                xyz_center = xyz.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K).contiguous()

                neighbors_xyz = index_points(xyz, idx)
                neighbors_xyz = neighbors_xyz.permute(0, 3, 1, 2).contiguous()

                dist = torch.sum(torch.sqrt((neighbors_xyz.permute(0, 2, 3, 1).contiguous() - xyz_center.permute(0, 2, 3, 1).contiguous())**2), dim=-1)

                x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta = convert_polar(neighbors_xyz, xyz_center)
                features_xyz = torch.cat((neighbors_xyz - xyz_center, xyz_center, x_alpha, x_beta, y_alpha,
                                      y_beta, z_alpha, z_beta, dist.unsqueeze(-3).contiguous()), dim=1)


                # features_xyz = self.mlp3(features_xyz)
                # features_xyz = torch.max(features_xyz, 3)[0]


                #################### feature
                features_centre = self.mlp1(features.transpose(-2, -1).unsqueeze(-1).contiguous())
                neighbors_features = index_points(features, idx)
                neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()
                neighbors_features = self.mlp4(torch.cat((neighbors_features, features_xyz), dim=1))
                neighbors_features = torch.max(neighbors_features, 3)[0].unsqueeze(-1).contiguous()


                # features_centre = self.mlp2(features.transpose(-2,-1).unsqueeze(-1))
                # neighbors_features = index_points(features, idx)
                # neighbors_features = neighbors_features.permute(0, 3, 1, 2)
                # neighbors_features = self.mlp1(neighbors_features)
                # neighbors_features = torch.max(neighbors_features, 3)[0].unsqueeze(-1)
                #
                features = (neighbors_features - features_centre).squeeze(-1).contiguous()
                #
                # features = self.mlp4(torch.cat((features, features_xyz), dim=1))



        # temp1 = self.local1(features_concat_list[0])
        # temp1 = self.mlp(torch.cat((temp1, features.expand(B, S, N, self.sample_num_list[0])), dim=1))
        # features = torch.max(features_concat_list[0], 3)[0].unsqueeze(-1)

        # temp2 = self.local2(features_concat_list[1])
        # temp2 = self.mlp2(torch.cat((temp2, features.expand(B, S, N, self.sample_num_list[1])), dim=1))
        # temp2 = torch.max(features_concat_list[1], 3)[0].unsqueeze(-1)

        # temp3 = self.local3(torch.cat((coords_concat_list[2], features.expand(B, S, N, self.sample_num_list[2])), dim=1))
        # temp3 = torch.max(temp3, 3)[0].unsqueeze(-1)

        # features = self.mlp(torch.cat(temp1, temp2, temp3), dim=1)
        # features = self.mlp3(torch.cat((temp1, temp2), dim=1))

            # coords_concat = self.mlp(torch.cat((coords_concat, features.expand(B, S, N, K)), dim=1))
            #
            # coords_concat = self.coord_pool(coords_concat)
            #
            #
            # # distF, idxF = knn_point(self.sample_num_list[0], features, features)  ### B N K
            #
            # # neighbors_features = index_points(features, idx)
            # # neighbors_features = neighbors_features.permute(0, 3, 1, 2)
            # # features_centre = features.transpose(-2, -1).unsqueeze(-1).expand(B, S, N, K)
            # #
            # # features_concat = torch.cat((neighbors_features - features_centre, features_centre), dim=1)
            # #
            # # features_concat = self.feature_mlp(features_concat)
            # # features_concat = self.feature_pool(features_concat)
            # # features_concat = torch.max(features_concat, 3)[0]
            #
            # features = self.mlp(torch.cat((coords_concat, features), dim=1))





            # final_feature.append(coords_concat)




        #
        # if len(self.sample_num_list) > 1:
        #     final_feature_concat = torch.cat(final_feature, dim=1)
        #     final_feature_concat = self.final_feature_mlp(final_feature_concat)  # .squeeze(-1)
        #     # final_feature_concat = torch.max(final_feature_concat, 2)[0].unsqueeze(-1)
        # else:
        #     final_feature_concat = final_feature[0]



        # if coords_feature is not None:
        #
        #     final_feature_concat = final_feature_concat + coords_feature.unsqueeze(-1)



        # return self.lrelu(self.mlp2(final_feature_concat) + self.short_cut(features.unsqueeze(-1))).squeeze(-1)
        return features
        # return final_feature_concat.squeeze(-1)


class Fuse_Stage(nn.Module):
    def __init__(self, b1_C, b2_C, b3_C, b4_C):
        super(Fuse_Stage, self).__init__()

        self.stage_12 = SharedMLP(b1_C, b2_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.stage_13 = SharedMLP(b1_C, b3_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.stage_23 = SharedMLP(b2_C, b3_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.stage_14 = SharedMLP(b1_C, b4_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.stage_24 = SharedMLP(b2_C, b4_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.stage_34 = SharedMLP(b3_C, b4_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.stage_conv1 = SharedMLP(b1_C, b1_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.stage_conv2 = SharedMLP(b2_C, b2_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.stage_conv3 = SharedMLP(b3_C, b3_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.stage_conv4 = SharedMLP(b4_C, b4_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.stage_fuse21 = PointNetFeaturePropagation(b1_C+b1_C, [b1_C, ], b2_C)
        self.stage_fuse31 = PointNetFeaturePropagation(b1_C+b1_C, [b1_C, ], b3_C)
        self.stage_fuse32 = PointNetFeaturePropagation(b2_C+b2_C, [b2_C, ], b3_C)
        self.stage_fuse41 = PointNetFeaturePropagation(b1_C+b1_C, [b1_C, ], b4_C)
        self.stage_fuse42 = PointNetFeaturePropagation(b2_C+b2_C, [b2_C, ], b4_C)
        self.stage_fuse43 = PointNetFeaturePropagation(b3_C+b3_C, [b3_C, ], b4_C)

    def forward(self, b1_f, b2_f, b3_f, b4_f, b2_idx, b3_idx, b4_idx):
        xyz = 0

        branch4_points_random1 = self.stage_14(index_points(b1_f.permute(0,2,1).contiguous(), b4_idx).permute(0,2,1).unsqueeze(-1).contiguous()).squeeze(-1)
        branch4_points_random2 = self.stage_24(index_points(b2_f.permute(0,2,1).contiguous(), b4_idx).permute(0,2,1).unsqueeze(-1).contiguous()).squeeze(-1)
        branch4_points_random3 = self.stage_34(index_points(b3_f.permute(0,2,1).contiguous(), b4_idx).permute(0,2,1).unsqueeze(-1).contiguous()).squeeze(-1)
        branch4_points_FP = self.stage_conv4((b4_f + branch4_points_random1 + branch4_points_random2 + branch4_points_random3).unsqueeze(-1)).squeeze(-1)


        temp43_point_stage4 = self.stage_fuse43(xyz, xyz, b3_f, b4_f)
        branch3_points_random1 = self.stage_13(index_points(b1_f.permute(0,2,1).contiguous(), b3_idx).permute(0,2,1).unsqueeze(-1).contiguous()).squeeze(-1)
        branch3_points_random2 = self.stage_23(index_points(b2_f.permute(0,2,1).contiguous(), b3_idx).permute(0,2,1).unsqueeze(-1).contiguous()).squeeze(-1)
        branch3_points_FP = self.stage_conv3((b3_f + branch3_points_random1 + branch3_points_random2 + temp43_point_stage4).unsqueeze(-1)).squeeze(-1)


        temp42_points_stage4 = self.stage_fuse42(xyz, xyz, b2_f, b4_f)
        temp32_points_stage4 = self.stage_fuse32(xyz, xyz, b2_f, b3_f)
        branch2_points_random = self.stage_12(index_points(b1_f.permute(0,2,1).contiguous(), b2_idx).permute(0,2,1).unsqueeze(-1).contiguous()).squeeze(-1)
        branch2_points_FP = self.stage_conv2((b2_f + branch2_points_random + temp32_points_stage4 + temp42_points_stage4).unsqueeze(-1)).squeeze(-1)


        temp41_points_stage4 = self.stage_fuse41(xyz, xyz, b1_f, b4_f)
        temp31_points_stage4 = self.stage_fuse31(xyz, xyz, b1_f, b3_f)
        temp21_points_stage4 = self.stage_fuse21(xyz, xyz, b1_f, b2_f)
        branch1_points = self.stage_conv1((b1_f + temp21_points_stage4 + temp31_points_stage4 + temp41_points_stage4).unsqueeze(-1).contiguous()).squeeze(-1)

        return branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature



class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x

class AdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, feat_channels):
        super(AdaptiveConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels

        self.conv0 = nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(out_channels, out_channels*in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y):
        # x: (bs, in_channels, num_points, k), y: (bs, feat_channels, num_points, k)
        batch_size, n_dims, num_points, k = x.size()

        y = self.conv0(y) # (bs, out, num_points, k)
        y = self.leaky_relu(self.bn0(y))
        y = self.conv1(y) # (bs, in*out, num_points, k)
        y = y.permute(0, 2, 3, 1).view(batch_size, num_points, k, self.out_channels, self.in_channels) # (bs, num_points, k, out, in)

        x = x.permute(0, 2, 3, 1).unsqueeze(4) # (bs, num_points, k, in_channels, 1)
        x = torch.matmul(y, x).squeeze(4) # (bs, num_points, k, out_channels)
        x = x.permute(0, 3, 1, 2).contiguous() # (bs, out_channels, num_points, k)

        x = self.bn1(x)
        x = self.leaky_relu(x)

        return x

class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        #
        ######################## adaptive conv
        # self.adapt_conv1 = AdaptiveConv(6, 64, 6)
        # self.adapt_conv2 = AdaptiveConv(6, 64, 64*2)
        ######################## adaptive conv




    def forward(self, x):
        batch_size = x.size(0)

        # # ######################## adaptive conv
        # points = x
        #
        # x, idx = get_graph_feature(x, k=self.k)
        # p, _ = get_graph_feature(points, k=self.k, idx=idx)
        # x = self.adapt_conv1(p, x)
        # x1 = x.max(dim=-1, keepdim=False)[0]
        #
        # x, idx = get_graph_feature(x1, k=self.k)
        # p, _ = get_graph_feature(points, k=self.k, idx=idx)
        # x = self.adapt_conv2(p, x)
        # x2 = x.max(dim=-1, keepdim=False)[0]
        #
        # x, _ = get_graph_feature(x2, k=self.k)
        # x = self.conv3(x)
        # x3 = x.max(dim=-1, keepdim=False)[0]
        #
        # x, _ = get_graph_feature(x3, k=self.k)
        # x = self.conv4(x)
        # x4 = x.max(dim=-1, keepdim=False)[0]
        #
        # x = torch.cat((x1, x2, x3, x4), dim=1)
        # # ######################## adaptive conv

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
        #
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, trans_channel):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)    ##########################################
        self.trans_conv = SharedMLP(trans_channel, mlp[0], bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))  #######################
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)
        #
        # points2 = points2.permute(0, 2, 1)
        # B, N, C = xyz1.shape
        # _, S, _ = xyz2.shape
        #
        # if S == 1:
        #     interpolated_points = points2.repeat(1, N, 1)
        # else:
        #     dists = square_distance(xyz1, xyz2)
        #     dists, idx = dists.sort(dim=-1)
        #     dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
        #
        #     dist_recip = 1.0 / (dists + 1e-8)
        #     norm = torch.sum(dist_recip, dim=2, keepdim=True)
        #     weight = dist_recip / norm
        #     # a = index_points(points2, idx)   #########################
        #     interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        #
        # if points1 is not None:
        #     points1 = points1.permute(0, 2, 1)
        #     new_points = torch.cat([points1, interpolated_points], dim=-1)
        # else:
        #     new_points = interpolated_points
        #
        # new_points = new_points.permute(0, 2, 1)
        # for i, conv in enumerate(self.mlp_convs):
        #     bn = self.mlp_bns[i]
        #     # new_points = F.relu(bn(conv(new_points)))
        #     new_points = self.lrelu(bn(conv(new_points)))
        # return new_points

        # points2 = points2.permute(0, 2, 1)


        # d_norm_2 = square_distance(xyz1.permute(0,2,1), xyz2.permute(0,2,1))  ###################


        # inner = torch.bmm(xyz1.transpose(1, 2), xyz2)  # (bs, v1, v2)

        points2 = self.trans_conv(points2.unsqueeze(-1).contiguous()).squeeze(-1)
        inner = torch.matmul(points1.transpose(1, 2).contiguous(), points2)
        s_norm_2 = torch.sum(points2 ** 2, dim=1)  # (bs, v2)
        t_norm_2 = torch.sum(points1 ** 2, dim=1)  # (bs, v1)
        d_norm_2 = s_norm_2.unsqueeze(1).contiguous() + t_norm_2.unsqueeze(2).contiguous() - 2 * inner

        nearest_index = torch.topk(d_norm_2, k=1, dim=-1, largest=False)[1]

        batch_size, num_points, k = nearest_index.size()

        id_0 = torch.arange(batch_size).view(-1, 1, 1)

        # points2 = points2.transpose(2, 1).contiguous()  # (bs, num_points, num_dims)
        points2 = points2.permute(0,2,1).contiguous()
        feature = points2[id_0, nearest_index]  # (bs, num_points, k, num_dims)
        feature = feature.permute(0, 1, 3, 2).squeeze(-1).contiguous()



        if points1 is not None:
            points1 = points1.permute(0, 2, 1).contiguous()
            feature = torch.cat([points1, feature], dim=-1)



        feature = feature.permute(0, 2, 1).contiguous()
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feature = self.lrelu(bn(conv(feature)))
        return feature


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all):
        super(DGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.transform_net = Transform_Net(args)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x


class DGCNN_semseg(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)
        

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        
        return x