import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


from collections import OrderedDict


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

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

# def square_distance(src, dst):
#     """
#     Calculate Euclid distance between each two points.
#
#     src^T * dst = xn * xm + yn * ym + zn * zm；
#     sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
#     sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
#     dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
#          = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
#
#     Input:
#         src: source points, [B, N, C]
#         dst: target points, [B, M, C]
#     Output:
#         dist: per-point square distance, [B, N, M]
#     """
#     B, N, _ = src.shape
#     _, M, _ = dst.shape
#     dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
#     dist += torch.sum(src ** 2, -1).view(B, N, 1)
#     dist += torch.sum(dst ** 2, -1).view(B, 1, M)
#     return dist


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


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    # dist = torch.gather(sqrdists, -1, group_idx)   #################
    return group_idx#, dist


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

##############################################################

def conv_bn(inp, oup, kernel, stride=1, activation='relu'):
    seq = nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride),
        nn.BatchNorm2d(oup)
    )
    if activation == 'relu':
        seq.add_module('2', nn.ReLU())
    return seq

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


def knn(x, k):
    # x = x.transpose(2,1)    ################################
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def pointsift_select(radius, xyz):
        """
        code by python matrix logic
        :param radius:
        :param xyz:
        :return: idx
        """
        dev = xyz.device
        B, N, _ = xyz.shape
        judge_dist = radius ** 2
        idx = torch.arange(N).repeat(8, 1).permute(1, 0).contiguous().repeat(B, 1, 1).to(dev)


        distenceNN = square_distance(xyz, xyz)  ########################



        for n in range(N):
            distance = torch.ones(B, N, 8).to(dev) * 1e10
            distance[:, n, :] = judge_dist
            centroid = xyz[:, n, :].view(B, 1, 3).to(dev)
            dist = torch.sum((xyz - centroid) ** 2, -1)  # shape: (B, N)

            # subspace_idx = torch.sum((xyz - centroid + 1).int() * torch.tensor([4, 2, 1], dtype=torch.int, device=dev), -1)   ##################%%%%%%%%%%%%%%%%%%%%
            subspace_idx = torch.sum((xyz > centroid).int() * torch.tensor([4, 2, 1], dtype=torch.int, device=dev), -1)  #################################


            for i in range(8):
                mask = (subspace_idx == i) & (dist > 1e-10) & (dist < judge_dist)  # shape: (B, N)
                distance[..., i][mask] = dist[mask]
                c = torch.min(distance[..., i], dim=-1)[1]
                idx[:, n, i] = torch.min(distance[..., i], dim=-1)[1]
        return idx

def pointsift_group(radius, xyz, points, use_xyz=True):

        B, N, C = xyz.shape
        assert C == 3
        idx = pointsift_select(radius, xyz)  # B, N, 8

        grouped_xyz = index_points(xyz, idx)  # B, N, 8, 3

        # grouped_xyz -= xyz.view(B, N, 1, 3)   ######################%%%%%%%%%%%%%%%%%%%%%%%

        xyz = xyz.view(B, N, 1, 3).repeat(1, 1, 8, 1)   #####################################
        grouped_edge = torch.cat((grouped_xyz-xyz, xyz), dim=3).contiguous()  ##################################
        # grouped_points = grouped_edge   ######################################


        if points is not None:
            grouped_points = index_points(points, idx)
            if use_xyz:
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            # grouped_points = grouped_xyz  ################%%%%%%%%%%%%%%%%%%%%%%
            grouped_points = grouped_xyz - xyz  ######################################

        return grouped_xyz, grouped_points, grouped_edge, idx




def conv1d(inplanes, outplanes, stride=1):

    return nn.Sequential(
        nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm1d(outplanes),
        nn.LeakyReLU(inplace=True, negative_slope=0.2)
    )

def convFuse(inplanes, outplanes, stride=1):

    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes))

def fc(inplanes, outplanes):

    return nn.Sequential(
        nn.Linear(inplanes, outplanes, bias=False),
        nn.BatchNorm1d(outplanes))

# class KeepHighResolutionModule(nn.Module):
#     def __init__(self, num_branches, blocks, num_block, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
#         super(KeepHighResolutionModule, self).__init__()
#
#         self.num_inchannels = num_inchannels
#         self.fuse_method = fuse_method
#         self.num_branches = num_branches
#         self.multi_scale_output = multi_scale_output
#
#         self.branches = self.make_branches(num_branches, blocks, num_block, num_channels)
#
#         self.fuse_layers = self.make_fuse()
#         self.relu = nn.ReLU(inplace=True)
#
#     def make_one_branch(self, branch_index, block, num_blocks, num_channels):
#         layers = []
#         layers.append(
#             block(self.num_inchannels[branch_index],
#                   num_channels[branch_index])
#         )
#
#         for i in range(1, num_blocks[branch_index]):
#             layers.append(
#                 block(self.num_inchannels[branch_index],
#                       num_channels[branch_index])
#             )
#
#         return nn.Sequential(*layers)
#
#     def make_branches(self, num_branches, block, num_blocks, num_channels):
#         branches = []
#
#         for i in range(num_branches):
#             branches.append(
#                 self.make_one_branch(i, block, num_blocks, num_channels)
#             )
#
#         return nn.ModuleList(branches)
#
#     def forward(self, x):
#         for i in range(self.num_branches):
#             x[i] = self.branches[i](x[i])
#
#         x_fuse = []
#
#
#         return x_fuse



def make_fuse(npoint_list):
    for i in range(len(npoint_list)):
        branch1 = npoint_list[0]
        branch2 = npoint_list[1]


def random_sample(xyz, sample_num):

    B, N, _ = xyz.size()
    permutation = torch.randperm(N)
    temp_sample = xyz[:, permutation]
    sampled_xyz = temp_sample[:, :sample_num, :]

    idx = permutation[:sample_num].unsqueeze(0).expand(B, sample_num)

    return sampled_xyz, idx

def sample_anchors(x, s):

    idx = torch.randperm(x.size(3))[:s]
    x = x[:, :, :, idx]

    return x

def make_head(pre_stage_channels):
    head_channels = [32, 64, 128, 256]

    incre_modules = []
    for i, channels in enumerate(pre_stage_channels):
        incre_module = SharedMLP(channels, head_channels[i]*4, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        incre_modules.append(incre_module)

    incre_modules = nn.ModuleList(incre_modules)

    downsamp_modules = []
    for i in range(len(pre_stage_channels)-1):
        in_channel = head_channels[i+1]
        out_channel = head_channels[i+1] * 2

        downsamp_module = SharedMLP(in_channel, out_channel, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        downsamp_modules.append(downsamp_module)

    downsamp_modules = nn.ModuleList(downsamp_modules)

    fuse_modules = []
    for i in range(len(pre_stage_channels)-1):
        in_channel = head_channels[i] * 4
        out_channel = head_channels[i+1] * 4

        fuse_module = SharedMLP(in_channel, out_channel, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        fuse_modules.append(fuse_module)

    fuse_modules = nn.ModuleList(fuse_modules)

    return incre_modules, downsamp_modules, fuse_modules

def make_stage(stage_channels):
    stage_modules = []

    for i in range(len(stage_channels)):
        in_channel = stage_channels[i]
        out_channel = stage_channels[i]

        stage_module = BasicBlock(in_channel, out_channel)
        stage_modules.append(stage_module)

    stage_modules = nn.Sequential(stage_modules[0], stage_modules[1])

    return stage_modules

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
    z_beta = torch.atan2(rel_z, r_xy).unsqueeze(-3).contiguous()
    z_alpha = torch.atan2(rel_y, rel_x).unsqueeze(-3).contiguous()

    ### Y_axis
    y_beta = torch.atan2(rel_y, r_zx).unsqueeze(-3).contiguous()
    y_alpha = torch.atan2(rel_x, rel_z).unsqueeze(-3).contiguous()

    ### X_axis
    x_beta = torch.atan2(rel_x, r_yz).unsqueeze(-3).contiguous()
    x_alpha = torch.atan2(rel_z, rel_y).unsqueeze(-3).contiguous()

    return x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(BasicBlock, self).__init__()

        # self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1)
        # self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-6, momentum=0.99)
        # self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1)
        # self.bn2 = nn.BatchNorm2d(inplanes, eps=1e-6, momentum=0.99)
        self.conv1 = SharedMLP(inplanes, outplanes, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = SharedMLP(outplanes, outplanes, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = SharedMLP(outplanes, outplanes, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)



    def forward(self, x):

        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.lrelu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        out = self.conv3(out)

        # out = self.lrelu(out + residual)

        return out



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


class SpatialAttention(nn.Module):
    def __init__(self, in_c, out_c):
        super(SpatialAttention, self).__init__()

        # self.conv = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.conv = SharedMLP(in_c, out_c //4, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, K, _, _ = x.size()
        context = self.conv(x)
        context = context.repeat(1,K//2,1,1)

        context = self.sigmoid(context)

        x = x + torch.mul(context, x)

        # avgout = torch.mean(x, dim=1, keepdim=True)
        # maxout, _ = torch.max(x, dim=1, keepdim=True)
        # out = torch.cat([avgout, maxout], dim=1)
        # out = self.conv(out)
        # out = self.sigmoid(out)
        #
        # out = torch.sum(out * x, dim=-1, keepdim=True)

        return x

class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels, reduction):
        super(AttentivePooling, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # self.avg_pool1 = nn.AdaptiveAvgPool1d(1)

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            # nn.Conv2d(in_channels, reduction, kernel_size=1, bias=False),
            # nn.Linear(channel, channel // reduction, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(reduction, in_channels, kernel_size=1, bias=False),
            nn.Softmax(dim=-2)   ################%%%%%%%%%%%%
            # nn.Sigmoid()           ###########################
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """

        # d = torch.rand(16, 512,48)
        # b = self.avg_pool1(d)

        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)  #########%%%%%%%%%%%%%%%%

        # a = self.avg_pool(x.permute(0,3,1,2))
        # scores = self.score_fn(a)
        # scores = scores.permute(0,2,3,1)

        # scores = self.score_fn(self.avg_pool(x))   ############################

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)
        features = self.mlp(features)
        # features = scores * x  ##traditional se

        return features
        # return self.mlp(features)  ############%%%%%%%%%%%%%


# class LocalSelfAttention2(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(LocalSelfAttention2, self).__init__()
#         self.q_conv = SharedMLP(in_c, out_c, bn=False, activation_fn=False)
#         self.k_conv = SharedMLP(in_c, out_c, bn=False, activation_fn=False)
#         self.v_conv = SharedMLP(in_c, out_c, bn=False, activation_fn=False)
#
#         self.trans_conv = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, feature):
#         q = self.q_conv(feature).permute(0, 2, 3, 1)
#         k = self.k_conv(feature).permute(0, 2, 1, 3)
#         v = self.v_conv(feature).permute(0, 2, 3, 1)
#
#         energy = torch.matmul(q, k)
#         attention = self.softmax(energy)
#         attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
#
#         context = torch.matmul(attention, v).permute(0, 3, 1, 2)
#         context = self.trans_conv(feature - context)
#         feature = feature + context
#
#         return feature

class LocalSelfAttention2(nn.Module):
    def __init__(self, in_c, out_c):
        super(LocalSelfAttention2, self).__init__()
        self.q_conv = SharedMLP(in_c, 16, bn=False, activation_fn=False)
        self.k_conv = SharedMLP(in_c, 16, bn=False, activation_fn=False)
        self.v_conv = SharedMLP(in_c, 16, bn=False, activation_fn=False)

        self.trans_conv = SharedMLP(16, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, neighbour, center):

        q = self.q_conv(neighbour).permute(0, 2, 3, 1)
        k = self.k_conv(center).permute(0, 2, 3, 1)
        v = self.v_conv(neighbour).permute(0,2,3,1)

        energy = torch.mul(q, k)
        attention = self.sigmoid(energy)

        context = torch.mul(attention, v)
        context = self.trans_conv(context.permute(0, 3, 1, 2))

        context = neighbour + context


        # q = self.q_conv(feature).permute(0, 2, 3, 1)
        # k = self.k_conv(feature).permute(0, 2, 1, 3)
        # v = self.v_conv(feature).permute(0, 2, 3, 1)
        #
        # energy = torch.matmul(q, k)
        # attention = self.softmax(energy)
        # attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        #
        # context = torch.matmul(attention, v).permute(0, 3, 1, 2)
        # context = self.trans_conv(feature - context)
        # feature = feature + context

        return context

class LocalSelfAttention(nn.Module):
    def __init__(self, in_c, out_c):
        super(LocalSelfAttention, self).__init__()

        self.k = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.q = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.v = SharedMLP(in_c, out_c, bn=False, activation_fn=False)

        self.bn = nn.BatchNorm2d(out_c, eps=1e-6, momentum=0.99)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature):

        local_key = self.k(feature).permute(0,2,3,1)
        local_query = self.q(feature).permute(0,2,1,3)

        local_value = self.v(feature).permute(0,2,3,1)

        sim_map = torch.matmul(local_key, local_query)
        sim_map = self.softmax(sim_map)

        context = self.lrelu(self.bn(torch.matmul(sim_map, local_value).permute(0,3,1,2)))

        return context



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
        # self.mlp2 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.mlp3 = SharedMLP(13, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.mlp4 = SharedMLP(in_c + 12, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.short_cut = SharedMLP(in_c, in_c, bn=True)

        # self.local1 = BasicBlock(in_c*2+1, in_c)
        # self.local2 = BasicBlock(in_c*2+1, in_c)

        # self.coord_pool = AttentivePooling(out_c, out_c, 4)
        # self.feature_pool = AttentivePooling(in_c, out_c, 4)
        self.local_attention1 = LocalSelfAttention2(3, 3)
        self.local_attention2 = LocalSelfAttention2(3, 3)
        self.spatial_attention1 = SpatialAttention(out_c, out_c)
        self.spatial_attention2 = SpatialAttention(out_c, out_c)
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

    def forward(self, features, xyz=None, norm=None):


        features = features.permute(0, 2, 1).contiguous()  ## B N C
        if norm is not None:
            norm = norm.permute(0, 2, 1).contiguous()  ## B N C
        # upper_coords = upper_coords.permute(0, 2, 1)  ## B N C
        # coords = coords.permute(0, 2, 1)  ## B N C


        S = features.size(2)

        for i, sample_num in enumerate(self.sample_num_list):

            dist, idx = knn_point(sample_num, features, features)  ### B N K

            # a = knn(features, sample_num)
            B, N, K = idx.size()



            if S == 3:
                features_centre = features.transpose(-2, -1).unsqueeze(-1).expand(B, S, N, K).contiguous()

                neighbors_features = index_points(features, idx)
                neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()

                # norm_center = norm.transpose(-2, -1).unsqueeze(-1).expand(B, S, N, K).contiguous()
                #
                # neighbors_norm = index_points(norm, idx)
                # neighbors_norm = neighbors_norm.permute(0, 3, 1, 2).contiguous()


                dist = torch.sqrt(torch.sum((neighbors_features.permute(0, 2, 3, 1).contiguous() - features_centre.permute(0, 2, 3, 1).contiguous()) ** 2, dim=-1))
                # dist_norm = torch.sqrt(torch.sum((neighbors_norm.permute(0, 2, 3, 1).contiguous() - norm_center.permute(0, 2, 3, 1).contiguous()) ** 2, dim=-1))
                # dist = torch.sum(torch.sqrt((neighbors_features.permute(0, 2, 3, 1).contiguous() - features_centre.permute(0, 2, 3, 1).contiguous())**2), dim=-1)

                x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta = convert_polar(neighbors_features, features_centre)

                # if norm is not None:
                #     neighbours_norm = index_points(norm, idx).permute(0, 3, 1, 2).contiguous()
                #     features = torch.cat((neighbors_features - features_centre, neighbors_features, x_alpha, x_beta, y_alpha,
                #          y_beta, z_alpha, z_beta, dist.unsqueeze(-3).contiguous(), neighbours_norm), dim=1)
                # else:

                # neighbors_features = self.local_attention1(neighbors_features, features_centre)
                features = torch.cat((neighbors_features - features_centre, neighbors_features, x_alpha, x_beta, y_alpha,
                                      y_beta, z_alpha, z_beta, dist.unsqueeze(-3).contiguous()), dim=1)
                # features = torch.cat((neighbors_features - features_centre, neighbors_features, dist.unsqueeze(-3).contiguous()), dim=1)
                # features_concat = torch.cat((neighbors_features - features_centre, features_centre, dist.unsqueeze(-3)), dim=1)
                # features_concat_list.append(features_concat)

                # features = self.local_attention1(features)
                features = self.mlp(features)
                # features = self.spatial_attention1(features)

                features = torch.max(features, 3)[0]
                # features = torch.sum(features, dim=-1, keepdim=False)
                # features = self.coord_pool(features).squeeze(-1)
            elif xyz is not None:
                ###################  xyz
                xyz_center = xyz.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K).contiguous()

                neighbors_xyz = index_points(xyz, idx)
                neighbors_xyz = neighbors_xyz.permute(0, 3, 1, 2).contiguous()


                # norm_center = norm.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K).contiguous()
                #
                # neighbors_norm = index_points(norm, idx)
                # neighbors_norm = neighbors_norm.permute(0, 3, 1, 2).contiguous()

                # dist = torch.sqrt(torch.sum((neighbors_xyz.permute(0, 2, 3, 1).contiguous() - xyz_center.permute(0, 2, 3, 1).contiguous()) ** 2, dim=-1))

                x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta = convert_polar(neighbors_xyz, xyz_center)

                # if norm is not None:
                #     neighbours_norm = index_points(norm, idx).permute(0, 3, 1, 2).contiguous()
                #     features_xyz = torch.cat((neighbors_xyz - xyz_center, xyz_center, x_alpha, x_beta, y_alpha,
                #                               y_beta, z_alpha, z_beta, neighbours_norm), dim=1)
                # else:
                # neighbors_xyz = self.local_attention2(neighbors_xyz, xyz_center)
                features_xyz = torch.cat((neighbors_xyz - xyz_center, xyz_center, x_alpha, x_beta, y_alpha,
                                      y_beta, z_alpha, z_beta), dim=1)
                # features_xyz = self.local_attention2(features_xyz)


                #################### feature
                # features_centre = features.transpose(-2, -1).unsqueeze(-1).expand(B, S, N, K).contiguous()
                #
                # neighbors_features = index_points(features, idx)
                # neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()
                #
                # features = torch.cat((neighbors_features - features_centre, features_centre, dist.unsqueeze(-3).contiguous()), dim=1)
                #
                # features = self.mlp4(features)
                # features = torch.max(features, 3)[0]


                features_centre = self.mlp1(features.transpose(-2, -1).unsqueeze(-1).contiguous())
                neighbors_features = index_points(features, idx)
                neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()
                neighbors_features = self.mlp4(torch.cat((neighbors_features, features_xyz), dim=1))
                # neighbors_features = self.mlp4(neighbors_features)
                # neighbors_features = self.spatial_attention2(neighbors_features)
                neighbors_features = torch.max(neighbors_features, 3)[0].unsqueeze(-1)
                # neighbors_features = torch.sum(neighbors_features, dim=-1, keepdim=True)


                # neighbors_features = self.feature_pool(neighbors_features)

                features = (neighbors_features - features_centre).squeeze(-1).contiguous()

        return features

class LocalAggregationPart(nn.Module):
    def __init__(self, in_c, out_c, sample_num_list):
        super(LocalAggregationPart, self).__init__()


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
        # self.mlp2 = SharedMLP(in_c, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.mlp3 = SharedMLP(13, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.mlp4 = SharedMLP(in_c + 12, out_c, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.short_cut = SharedMLP(in_c, in_c, bn=True)

        # self.local1 = BasicBlock(in_c*2+1, in_c)
        # self.local2 = BasicBlock(in_c*2+1, in_c)

        self.coord_pool = AttentivePooling(out_c, out_c, 4)
        self.feature_pool = AttentivePooling(in_c, out_c, 4)

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

    def forward(self, features, xyz=None, norm=None):


        features = features.permute(0, 2, 1).contiguous()  ## B N C
        if norm is not None:
            norm = norm.permute(0, 2, 1).contiguous()  ## B N C
        # upper_coords = upper_coords.permute(0, 2, 1)  ## B N C
        # coords = coords.permute(0, 2, 1)  ## B N C


        S = features.size(2)

        for i, sample_num in enumerate(self.sample_num_list):

            dist, idx = knn_point(sample_num, features, features)  ### B N K

            # a = knn(features, sample_num)
            B, N, K = idx.size()



            if S == 3:
                features_centre = features.transpose(-2, -1).unsqueeze(-1).expand(B, S, N, K).contiguous()

                neighbors_features = index_points(features, idx)
                neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()

                dist = torch.sqrt(torch.sum((neighbors_features.permute(0, 2, 3, 1).contiguous() - features_centre.permute(0, 2, 3, 1).contiguous()) ** 2, dim=-1))
                # dist = torch.sum(torch.sqrt((neighbors_features.permute(0, 2, 3, 1).contiguous() - features_centre.permute(0, 2, 3, 1).contiguous())**2), dim=-1)

                x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta = convert_polar(neighbors_features, features_centre)

                # if norm is not None:
                #     neighbours_norm = index_points(norm, idx).permute(0, 3, 1, 2).contiguous()
                #     features = torch.cat((neighbors_features - features_centre, neighbors_features, x_alpha, x_beta, y_alpha,
                #          y_beta, z_alpha, z_beta, dist.unsqueeze(-3).contiguous(), neighbours_norm), dim=1)
                # else:
                features = torch.cat((neighbors_features - features_centre, neighbors_features, x_alpha, x_beta, y_alpha,
                                      y_beta, z_alpha, z_beta, dist.unsqueeze(-3).contiguous()), dim=1)
                # features = torch.cat((neighbors_features - features_centre, neighbors_features, dist.unsqueeze(-3).contiguous()), dim=1)
                # features_concat = torch.cat((neighbors_features - features_centre, features_centre, dist.unsqueeze(-3)), dim=1)
                # features_concat_list.append(features_concat)

                features = self.mlp(features)
                features = torch.max(features, 3)[0]
                # features = self.coord_pool(features).squeeze(-1)
            elif xyz is not None:
                ###################  xyz
                xyz_center = xyz.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K).contiguous()

                neighbors_xyz = index_points(xyz, idx)
                neighbors_xyz = neighbors_xyz.permute(0, 3, 1, 2).contiguous()

                # dist_xyz = torch.sum(torch.sqrt((neighbors_xyz.permute(0, 2, 3, 1).contiguous() - xyz_center.permute(0, 2, 3, 1).contiguous())**2), dim=-1)
                x_alpha, x_beta, y_alpha, y_beta, z_alpha, z_beta = convert_polar(neighbors_xyz, xyz_center)

                # if norm is not None:
                #     neighbours_norm = index_points(norm, idx).permute(0, 3, 1, 2).contiguous()
                #     features_xyz = torch.cat((neighbors_xyz - xyz_center, xyz_center, x_alpha, x_beta, y_alpha,
                #                               y_beta, z_alpha, z_beta, neighbours_norm), dim=1)
                # else:
                features_xyz = torch.cat((neighbors_xyz - xyz_center, xyz_center, x_alpha, x_beta, y_alpha,
                                      y_beta, z_alpha, z_beta), dim=1)
                # features_xyz = torch.cat((neighbors_xyz - xyz_center, xyz_center), dim=1)




                #################### feature
                # features_centre = features.transpose(-2, -1).unsqueeze(-1).expand(B, S, N, K).contiguous()
                #
                # neighbors_features = index_points(features, idx)
                # neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()
                #
                # features = torch.cat((neighbors_features - features_centre, features_centre, dist.unsqueeze(-3).contiguous()), dim=1)
                #
                # features = self.mlp4(features)
                # features = torch.max(features, 3)[0]


                features_centre = self.mlp1(features.transpose(-2, -1).unsqueeze(-1).contiguous())
                neighbors_features = index_points(features, idx)
                neighbors_features = neighbors_features.permute(0, 3, 1, 2).contiguous()
                neighbors_features = self.mlp4(torch.cat((neighbors_features, features_xyz), dim=1))
                neighbors_features = torch.max(neighbors_features, 3)[0].unsqueeze(-1)
                # neighbors_features = self.feature_pool(neighbors_features)

                features = (neighbors_features - features_centre).squeeze(-1).contiguous()





        return features



class Mlp(nn.Module):

    def __init__(self, planes_tab=[], learnable=True):
        super(Mlp, self).__init__()

        self.layers = self.make_layer(planes_tab)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if not learnable:
            for p in self.parameters():
                p.requires_grad = False

    def make_layer(self, planes_tab):

        layers = nn.ModuleList()
        for i in range(len(planes_tab) - 1):
            layers.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(planes_tab[i], planes_tab[i + 1], kernel_size=(1, 1), bias=False)),
                ('bn', nn.BatchNorm2d(planes_tab[i + 1])),
                ('relu', nn.ReLU(inplace=True))])))

        return layers

    def forward(self, x):

        # layers in ModuleList
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        return x

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
        branch4_points_FP = self.stage_conv4((b4_f + branch4_points_random1 + branch4_points_random2 + branch4_points_random3).unsqueeze(-1).contiguous()).squeeze(-1)


        temp43_point_stage4 = self.stage_fuse43(xyz, xyz, b3_f, b4_f)
        branch3_points_random1 = self.stage_13(index_points(b1_f.permute(0,2,1).contiguous(), b3_idx).permute(0,2,1).unsqueeze(-1).contiguous()).squeeze(-1)
        branch3_points_random2 = self.stage_23(index_points(b2_f.permute(0,2,1).contiguous(), b3_idx).permute(0,2,1).unsqueeze(-1).contiguous()).squeeze(-1)
        branch3_points_FP = self.stage_conv3((b3_f + branch3_points_random1 + branch3_points_random2 + temp43_point_stage4).unsqueeze(-1).contiguous()).squeeze(-1)


        temp42_points_stage4 = self.stage_fuse42(xyz, xyz, b2_f, b4_f)
        temp32_points_stage4 = self.stage_fuse32(xyz, xyz, b2_f, b3_f)
        branch2_points_random = self.stage_12(index_points(b1_f.permute(0,2,1).contiguous(), b2_idx).permute(0,2,1).unsqueeze(-1).contiguous()).squeeze(-1)
        branch2_points_FP = self.stage_conv2((b2_f + branch2_points_random + temp32_points_stage4 + temp42_points_stage4).unsqueeze(-1).contiguous()).squeeze(-1)


        temp41_points_stage4 = self.stage_fuse41(xyz, xyz, b1_f, b4_f)
        temp31_points_stage4 = self.stage_fuse31(xyz, xyz, b1_f, b3_f)
        temp21_points_stage4 = self.stage_fuse21(xyz, xyz, b1_f, b2_f)
        branch1_points = self.stage_conv1((b1_f + temp21_points_stage4 + temp31_points_stage4 + temp41_points_stage4).unsqueeze(-1).contiguous()).squeeze(-1)

        return branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP

class KeepHighResolutionModule(nn.Module):

    def __init__(self, data_C, b1_C, b2_C, b3_C, b4_C):
        super(KeepHighResolutionModule, self).__init__()

        self.local_num_neighbors = [16,32]
        self.neighbour = 20


        # self.fc_start = nn.Linear(3, 8)
        # self.bn_start = nn.Sequential(
        #     nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
        #     nn.LeakyReLU(0.2)
        # )
        #
        # # encoding layers
        # self.encoder = nn.ModuleList([
        #     LocalFeatureAggregation(8, 16, num_neighbors=16),
        #     LocalFeatureAggregation(32, 64, num_neighbors=16),
        #     LocalFeatureAggregation(128, 128, num_neighbors=16),
        #     LocalFeatureAggregation(256, 256, num_neighbors=16)
        # ])
        #
        # self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())
        #
        # self.decimation = 4



        # self.start = SharedMLP(7, b1_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.start_la1 = LocalAggregation(3 * 2 + 7, b1_C, [self.neighbour, ])
        self.start_la2 = LocalAggregation(3 * 2 + 7, b2_C, [self.neighbour, ])
        self.start_la3 = LocalAggregation(3 * 2 + 7, b3_C, [self.neighbour, ])
        self.start_la4 = LocalAggregation(3 * 2 + 7, b4_C, [self.neighbour, ])

        self.start_fuse = Fuse_Stage(64, 64, 64, 64)


        ###stage2
        self.stage2_la1 = LocalAggregation(b1_C, b1_C, [self.neighbour, ])
        self.stage2_la2 = LocalAggregation(b2_C, b2_C, [self.neighbour, ])
        self.stage2_la3 = LocalAggregation(b3_C, b3_C, [self.neighbour, ])
        self.stage2_la4 = LocalAggregation(b4_C, b4_C, [self.neighbour, ])

        self.stage2_fuse = Fuse_Stage(64, 64, 64, 64)

        ###stage3
        self.stage3_la1 = LocalAggregation(b1_C, b1_C, [self.neighbour, ])
        self.stage3_la2 = LocalAggregation(b2_C, b2_C, [self.neighbour, ])
        self.stage3_la3 = LocalAggregation(b3_C, b3_C, [self.neighbour, ])
        self.stage3_la4 = LocalAggregation(b4_C, b4_C, [self.neighbour, ])

        self.stage3_fuse = Fuse_Stage(64, 64, 64, 64)


        ###stage4
        self.stage4_la1 = LocalAggregation(b1_C, b1_C, [self.neighbour, ])
        self.stage4_la2 = LocalAggregation(b2_C, b2_C, [self.neighbour, ])
        self.stage4_la3 = LocalAggregation(b3_C, b3_C, [self.neighbour, ])
        self.stage4_la4 = LocalAggregation(b4_C, b4_C, [self.neighbour, ])

        self.stage4_fuse = Fuse_Stage(64, 64, 64, 64)


        #
        # ###stage3
        #
        # self.stage3_1 = SharedMLP(b2_C, b3_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        #
        # self.stage3_12 = SharedMLP(b1_C, b2_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.stage3_13 = SharedMLP(b1_C, b3_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.stage3_23 = SharedMLP(b2_C, b3_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        #
        #
        # # self.stage3_1stage = make_stage([b1_C, b1_C])
        # # self.stage3_2stage = make_stage([b2_C, b2_C])
        # # self.stage3_3stage = make_stage([b3_C, b3_C])
        # self.stage3_la1 = LocalAggregation(b1_C, b1_C, [20,], [0.2, 0.4, 0.8])
        # self.stage3_la2 = LocalAggregation(b2_C, b2_C, [20,], [0.2, 0.4, 0.8])
        # self.stage3_la3 = LocalAggregation(b3_C, b3_C, [20,], [0.3, 0.6, 1.2])
        #
        #
        # # self.conv3_1 = conv1d(b1_C, b1_C)
        # self.stage3_conv1 = SharedMLP(b1_C, b1_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.stage3_conv2 = SharedMLP(b2_C, b2_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.stage3_conv3 = SharedMLP(b3_C, b3_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        #
        # self.stage3_fuse21 = PointNetFeaturePropagation(b1_C+b1_C, [b1_C, ], b2_C)
        # self.stage3_fuse31 = PointNetFeaturePropagation(b1_C+b1_C, [b1_C, ], b3_C)
        # self.stage3_fuse32 = PointNetFeaturePropagation(b2_C+b2_C, [b2_C, ], b3_C)
        #
        #
        # ###stage4
        #
        # self.stage4_1 = SharedMLP(b3_C, b4_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        #
        # self.stage4_12 = SharedMLP(b1_C, b2_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.stage4_13 = SharedMLP(b1_C, b3_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.stage4_23 = SharedMLP(b2_C, b3_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.stage4_14 = SharedMLP(b1_C, b4_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.stage4_24 = SharedMLP(b2_C, b4_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.stage4_34 = SharedMLP(b3_C, b4_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        #
        #
        # # self.stage4_1stage = make_stage([b1_C, b1_C])
        # # self.stage4_2stage = make_stage([b2_C, b2_C])
        # # self.stage4_3stage = make_stage([b3_C, b3_C])
        # # self.stage4_4stage = make_stage([b4_C, b4_C])
        # self.stage4_la1 = LocalAggregation(b1_C, b1_C, [20,], [0.2, 0.4, 0.8])
        # self.stage4_la2 = LocalAggregation(b2_C, b2_C, [20,], [0.2, 0.4, 0.8])
        # self.stage4_la3 = LocalAggregation(b3_C, b3_C, [20,], [0.3, 0.6, 1.2])
        # self.stage4_la4 = LocalAggregation(b4_C, b4_C, [20,], [0.4, 0.8, 1.6])
        #
        # self.stage4_conv1 = SharedMLP(b1_C, b1_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.stage4_conv2 = SharedMLP(b2_C, b2_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.stage4_conv3 = SharedMLP(b3_C, b3_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.stage4_conv4 = SharedMLP(b4_C, b4_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        #
        #
        # self.stage4_fuse21 = PointNetFeaturePropagation(b1_C+b1_C, [b1_C, ], b2_C)
        # self.stage4_fuse31 = PointNetFeaturePropagation(b1_C+b1_C, [b1_C, ], b3_C)
        # self.stage4_fuse32 = PointNetFeaturePropagation(b2_C+b2_C, [b2_C, ], b3_C)
        # self.stage4_fuse41 = PointNetFeaturePropagation(b1_C+b1_C, [b1_C, ], b4_C)
        # self.stage4_fuse42 = PointNetFeaturePropagation(b2_C+b2_C, [b2_C, ], b4_C)
        # self.stage4_fuse43 = PointNetFeaturePropagation(b3_C+b3_C, [b3_C, ], b4_C)



        ###final stage
        self.convFinal = conv1d(b1_C, b1_C)

        # self.final_fuse43 = PointNetFeaturePropagation(b3_C + b4_C, [b3_C, ])
        # self.final_fuse32 = PointNetFeaturePropagation(b2_C + b3_C, [b2_C, ])
        # self.final_fuse21 = PointNetFeaturePropagation(b1_C + b2_C, [b1_C, ])


        # self.final_fuse1 = LocalAggregation(b2_C * 2, b2_C * 2, self.local_num_neighbors, [0.2, 0.4, 0.8])
        # self.final_fuse2 = LocalAggregation(b3_C * 2, b3_C * 2, self.local_num_neighbors, [0.3, 0.6, 1.2])
        # self.final_fuse3 = LocalAggregation(b4_C * 2, b4_C * 2, self.local_num_neighbors, [0.3, 0.6, 1.2])
        # self.final_12 = SharedMLP(3, b2_C*2, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))




        # self.fc1 = fc(480, 256)
        # self.fc2 = fc(256, 40)
        # self.bn1 = nn.BatchNorm1d(256)
        # self.bn2 = nn.BatchNorm1d(40)
        #
        # self.drop1 = nn.Dropout(0.5)
        # self.drop2 = nn.Dropout(0.5)
        # self.relu = nn.ReLU(inplace=True)
        # self.lse1 = LocalSpatialEncoding(32, 16)
        # self.decimation = 4
        # # self.lfa = LocalFeatureAggregation(8, 16, num_neighbors, device)
        # self.fc_start = nn.Linear(3, 8)
        # self.bn_start = nn.Sequential(
        #     nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
        #     nn.LeakyReLU(0.2)
        # )

        # self.incre_modules, self.downsamp_modules, self.fuse_modules = make_head([32, 64, 128, 256])
        self.final_class = SharedMLP(1024, 1024, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.final_pool = AttentivePooling(1024, 1024, 256)

        self.conv_x1 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.conv_x2 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.conv_x2_temp = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.conv_x3 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.conv_x3_temp = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.conv_x4 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.conv_x4_temp = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.final = SharedMLP(512, 512, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

    def forward(self, xyz, points):

        # xyz = xyz.permute(0, 2, 1)    ### B N C
        # if points is not None:
        #     points = points.permute(0, 2, 1)

        # N = xyz.size(1)
        # d = self.decimation
        #
        # coords = xyz[...,:3]
        # x = self.fc_start(xyz).transpose(-2,-1).unsqueeze(-1)
        # x = self.bn_start(x) # shape (B, d, N, 1)
        #
        # decimation_ratio = 1
        #
        # # <<<<<<<<<< ENCODER
        # x_stack = []
        #
        # permutation = torch.randperm(N)
        # coords = coords[:,permutation]
        # x = x[:,:,permutation]
        #
        # for lfa in self.encoder:
        #     # at iteration i, x.shape = (B, N//(d**i), d_in)
        #     x = lfa(coords[:,:N//decimation_ratio], x)
        #     x_stack.append(x.clone())
        #     decimation_ratio *= d
        #     x = x[:,:,:N//decimation_ratio]
        #
        #
        # # # >>>>>>>>>> ENCODER
        # x = self.mlp(x)


        ###distribution stage2


        branch1_xyz = xyz   ### B C N

        if points is not None:
            branch1_norm = points
        else:
            branch1_norm = None


        # branch2_idx_FP = farthest_point_sample(branch1_xyz, 512)
        # branch2_xyz_FP = index_points(branch1_xyz, branch2_idx_FP)  # FPS generate branch2

        # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
        # branch2_xyz_FP = branch2_xyz_FP.permute(0, 2, 1)  ### B C N

        # branch1_points = self.stage2_1(branch1_xyz.unsqueeze(-1)).squeeze(-1)

        branch1_points = self.start_la1(branch1_xyz, norm=branch1_norm)

        # x1 = self.conv_x1(branch1_points.unsqueeze(-1))


        branch2_idx_FP = farthest_point_sample(branch1_xyz.permute(0,2,1).contiguous(), 512)
        branch2_xyz = index_points(branch1_xyz.permute(0,2,1).contiguous(), branch2_idx_FP)
        if points is not None:
            branch2_norm = index_points(branch1_norm.permute(0,2,1), branch2_idx_FP).permute(0,2,1).contiguous()
        else:
            branch2_norm = None
        branch2_points_FP = self.start_la2(branch2_xyz.permute(0, 2, 1).contiguous(), norm=branch2_norm)  # FPS generate branch2
        # branch2_points_FP = self.start_la2(branch2_xyz.permute(0,2,1).contiguous()) # FPS generate branch2


        branch3_idx_FP = farthest_point_sample(branch2_xyz, 256)
        branch3_xyz = index_points(branch2_xyz, branch3_idx_FP)
        if points is not None:
            branch3_norm = index_points(branch2_norm.permute(0,2,1), branch3_idx_FP).permute(0,2,1).contiguous()
        else:
            branch3_norm = None
        branch3_points_FP = self.start_la3(branch3_xyz.permute(0,2,1).contiguous(), norm=branch3_norm)


        branch4_idx_FP = farthest_point_sample(branch3_xyz, 128)
        branch4_xyz = index_points(branch3_xyz, branch4_idx_FP)
        if points is not None:
            branch4_norm = index_points(branch3_norm.permute(0,2,1), branch4_idx_FP).permute(0,2,1).contiguous()
        else:
            branch4_norm = None
        branch4_points_FP = self.start_la4(branch4_xyz.permute(0,2,1).contiguous(), norm=branch4_norm)


        ################ fuse
        branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.start_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
                                                                                             branch2_idx_FP, branch3_idx_FP, branch4_idx_FP)

        x1 = self.conv_x1(branch1_points.unsqueeze(-1).contiguous())

        branch1_xyz = branch1_xyz.permute(0,2,1).contiguous()

        ############## Local Aggregation  stage2
        branch1_points = self.stage2_la1(branch1_points, branch1_xyz, norm=branch1_norm)
        branch2_points_FP = self.stage2_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm) # FPS generate branch2
        branch3_points_FP = self.stage2_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
        branch4_points_FP = self.stage2_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)


        ##################  Fuse
        branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage2_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
                                                                                             branch2_idx_FP, branch3_idx_FP, branch4_idx_FP)

        x2 = self.conv_x2(branch1_points.unsqueeze(-1).contiguous())


        ############## Local Aggregation  stage3
        branch1_points = self.stage3_la1(branch1_points, branch1_xyz, norm=branch1_norm)
        branch2_points_FP = self.stage3_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm) # FPS generate branch2
        branch3_points_FP = self.stage3_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
        branch4_points_FP = self.stage3_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)


        ##################  Fuse
        branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage3_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
                                                                                             branch2_idx_FP, branch3_idx_FP, branch4_idx_FP)

        x3 = self.conv_x3(branch1_points.unsqueeze(-1).contiguous())



        ############## Local Aggregation  stage4
        branch1_points = self.stage4_la1(branch1_points, branch1_xyz, norm=branch1_norm)
        branch2_points_FP = self.stage4_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm) # FPS generate branch2
        branch3_points_FP = self.stage4_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
        branch4_points_FP = self.stage4_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)


        ##################  Fuse
        branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage4_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
                                                                                             branch2_idx_FP, branch3_idx_FP, branch4_idx_FP)

        x4 = self.conv_x4(branch1_points.unsqueeze(-1).contiguous())



        # # branch1_points = branch1_points.permute(0, 2, 1)  ### B C N
        # # branch2_points = self.stage2_1(index_points(branch1_points.permute(0,2,1), branch2_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        #
        #
        # # branch1_points = self.stage2_1stage(branch1_points.unsqueeze(-1)).squeeze(-1)
        # # branch2_points = self.stage2_2stage(branch2_points.unsqueeze(-1)).squeeze(-1)
        #
        # ### local feature aggregation
        # branch1_points = self.stage2_la1(xyz, branch1_points)
        # # branch2_points = self.stage2_1(index_points(branch1_points.permute(0,2,1), branch2_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        # branch2_points_FP = self.stage2_la2(xyz, branch2_points_FP)
        #
        #
        #
        # ###fuse stage2
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
        # # branch2_xyz, branch2_idx = random_sample(branch1_xyz, 512)  ## random sample 512
        # # branch2_xyz = index_points(branch1_xyz, farthest_point_sample(branch1_xyz, 512))  ### FPS 512
        #
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B C N
        # # branch2_xyz = branch2_xyz.permute(0, 2, 1)  ### B C N
        #
        # # branch2_points_random = self.stage2_12(branch2_xyz.unsqueeze(-1)).squeeze(-1)
        # branch2_points_random = self.stage2_12(index_points(branch1_points.permute(0,2,1), branch2_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        # # branch2_points_FP = self.stage2_conv2(torch.cat((branch2_points_FP, branch2_points_random), 1).unsqueeze(-1)).squeeze(-1)
        # branch2_points_FP = self.stage2_conv2((branch2_points_FP + branch2_points_random).unsqueeze(-1)).squeeze(-1)
        #
        #
        # temp21_points_stage2 = self.stage2_fuse21(xyz, xyz, branch1_points, branch2_points_FP)
        # # branch1_points = self.stage2_conv1(torch.cat((branch1_points, temp21_points_stage2), 1).unsqueeze(-1)).squeeze(-1)
        # branch1_points = self.stage2_conv1((branch1_points + temp21_points_stage2).unsqueeze(-1)).squeeze(-1)  # branh2 upsample
        #
        #
        # # branch1_points_temp = self.initial2_la(xyz, branch1_xyz)
        # # x2 = torch.cat((self.conv_x2(branch1_points.unsqueeze(-1)), self.conv_x2_temp(branch1_points_temp.unsqueeze(-1))),dim=1)
        # x2 = self.conv_x2(branch1_points.unsqueeze(-1))
        # # x2 = branch1_points.unsqueeze(-1)
        # #######################################################################################
        #
        #
        # ###distribution stage3
        # # branch2_xyz_FP = branch2_xyz_FP.permute(0, 2, 1)  ### B N C
        # # branch3_xyz = random_sample(branch2_xyz, 256)  ## random sample 256, generate branch3
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
        # branch3_idx_FP = farthest_point_sample(branch2_points_FP.permute(0,2,1), 256)
        # # branch3_xyz_FP = index_points(branch2_xyz_FP, branch3_idx_FP)  # generate branch3
        # # branch3_xyz_FP = branch3_xyz_FP.permute(0, 2, 1)  ### B C N
        # # branch2_xyz = branch2_xyz.permute(0, 2, 1)  ### B C N
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B C N
        # # branch2_xyz_FP = branch2_xyz_FP.permute(0, 2, 1)  ### B C N
        #
        #
        #
        # branch3_points_FP = self.stage3_1(index_points(branch2_points_FP.permute(0,2,1), branch3_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        #
        #
        # # branch1_points = self.stage3_1stage(branch1_points.unsqueeze(-1)).squeeze(-1)
        # # branch2_points = self.stage3_2stage(branch2_points.unsqueeze(-1)).squeeze(-1)
        # # branch3_points = self.stage3_3stage(branch3_points.unsqueeze(-1)).squeeze(-1)
        #
        #
        # ### local feature aggregation
        # branch1_points = self.stage3_la1(xyz, branch1_points)
        # branch2_points_FP = self.stage3_la2(xyz, branch2_points_FP)
        #
        # # branch3_points = self.stage3_1(index_points(branch2_points.permute(0,2,1), branch3_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        # branch3_points_FP = self.stage3_la3(xyz, branch3_points_FP)
        #
        #
        #
        # ###fuse stage3
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
        # # branch3_xyz, branch3_idx = random_sample(branch1_xyz, 256)  ### random sample 256
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B C N
        # # branch3_xyz = branch3_xyz.permute(0, 2, 1)  ### B C N
        # # branch3_points_random1 = self.stage3_13(branch3_xyz.unsqueeze(-1)).squeeze(-1)
        # branch3_points_random1 = self.stage3_13(index_points(branch1_points.permute(0,2,1), branch3_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        # # branch3_xyz = branch3_xyz.permute(0, 2, 1)  ### B C N
        # # branch2_xyz = branch2_xyz.permute(0, 2, 1)   ### B N C
        # # branch2_xyz_FP = branch2_xyz_FP.permute(0, 2, 1)  ### B N C
        # # branch3_xyz, branch3_idx = random_sample(branch2_xyz_FP, 256)  ### random sample 256
        # # branch3_xyz = branch3_xyz.permute(0, 2, 1)  ### B C N
        # # branch3_points_random2 = self.stage3_23(branch3_xyz.unsqueeze(-1)).squeeze(-1)
        # branch3_points_random2 = self.stage3_23(index_points(branch2_points_FP.permute(0,2,1), branch3_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        # # branch3_points_FP = self.stage3_conv3(torch.cat((branch3_points_FP, branch3_points_random1, branch3_points_random2), 1).unsqueeze(-1)).squeeze(-1)
        # branch3_points_FP = self.stage3_conv3((branch3_points_FP + branch3_points_random1 + branch3_points_random2).unsqueeze(-1)).squeeze(-1)
        #
        #
        #
        # temp32_points_stage3 = self.stage3_fuse32(xyz, xyz, branch2_points_FP, branch3_points_FP)
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
        # # branch2_xyz, branch2_idx = random_sample(branch1_xyz, 512)  ### random sample 512
        # # branch2_xyz = index_points(branch1_xyz, farthest_point_sample(branch1_xyz, 512))  ### FPS
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B C N
        # # branch2_xyz = branch2_xyz.permute(0, 2, 1) ### B C N
        # # branch2_points_random = self.stage3_12(branch2_xyz.unsqueeze(-1)).squeeze(-1)
        # branch2_points_random = self.stage3_12(index_points(branch1_points.permute(0,2,1), branch2_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        # # branch2_points_FP = self.stage3_conv2(torch.cat((branch2_points_FP, branch2_points_random, temp32_points_stage3), 1).unsqueeze(-1)).squeeze(-1)
        # branch2_points_FP = self.stage3_conv2((branch2_points_FP + branch2_points_random + temp32_points_stage3).unsqueeze(-1)).squeeze(-1)
        #
        #
        # temp31_points_stage3 = self.stage3_fuse31(xyz, xyz, branch1_points, branch3_points_FP)
        # temp21_points_stage3 = self.stage3_fuse21(xyz, xyz, branch1_points, branch2_points_FP)
        # # branch1_points = self.stage3_conv1(torch.cat(((branch1_points, temp21_points_stage3, temp31_points_stage3)), 1).unsqueeze(-1)).squeeze(-1)
        # branch1_points = self.stage3_conv1((branch1_points + temp21_points_stage3 + temp31_points_stage3).unsqueeze(-1)).squeeze(-1)
        #
        # # branch1_points_temp = self.initial3_la(xyz, branch1_xyz)
        # # x3 = torch.cat((self.conv_x3(branch1_points.unsqueeze(-1)), self.conv_x3_temp(branch1_points_temp.unsqueeze(-1))), dim=1)
        # x3 = self.conv_x3(branch1_points.unsqueeze(-1))
        # # x3 = branch1_points.unsqueeze(-1)
        # ######################################################################################
        #
        #
        # ###distribution stage4
        # # branch3_xyz_FP = branch3_xyz_FP.permute(0, 2, 1)  ### B N C
        # # branch4_xyz_FP_idx = farthest_point_sample(branch3_xyz_FP, 128)
        # # branch4_xyz = random_sample(branch3_xyz, 128)  ### random sample 128, generate branch4
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
        # branch4_idx_FP = farthest_point_sample(branch3_points_FP.permute(0,2,1), 128)
        # # branch4_xyz_FP = index_points(branch3_xyz_FP, branch4_idx_FP)  # generate branch4
        #
        # # branch4_xyz_FP = branch4_xyz_FP.permute(0, 2, 1)  ### B C N
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B C N
        # # branch2_xyz_FP = branch2_xyz_FP.permute(0, 2, 1)  ### B C N
        # # branch3_xyz_FP = branch3_xyz_FP.permute(0, 2, 1)  ### B C N
        #
        #
        # branch4_points_FP = self.stage4_1(index_points(branch3_points_FP.permute(0,2,1), branch4_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        #
        # # branch1_points = self.stage4_1stage(branch1_points.unsqueeze(-1)).squeeze(-1)
        # # branch2_points = self.stage4_2stage(branch2_points.unsqueeze(-1)).squeeze(-1)
        # # branch3_points = self.stage4_3stage(branch3_points.unsqueeze(-1)).squeeze(-1)
        # # branch4_points = self.stage4_4stage(branch4_points.unsqueeze(-1)).squeeze(-1)
        #
        # # ### local feature aggregation
        # # branch2_xyz = branch2_xyz.permute(0, 2, 1)  ### B C N
        # branch1_points = self.stage4_la1(xyz, branch1_points)
        # branch2_points_FP = self.stage4_la2(xyz, branch2_points_FP)
        # branch3_points_FP = self.stage4_la3(xyz, branch3_points_FP)
        #
        # # branch4_points = self.stage4_1(index_points(branch3_points.permute(0,2,1), branch4_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        # branch4_points_FP = self.stage4_la4(xyz, branch4_points_FP)
        #
        #
        # ###fuse stage4
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
        # # branch4_xyz, branch4_idx = random_sample(branch1_xyz, 128)  ### random sample 128
        # # branch4_xyz = branch4_xyz.permute(0, 2, 1)  ### B C N
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B C N
        # # branch4_points_random1 = self.stage4_14(branch4_xyz.unsqueeze(-1)).squeeze(-1)
        # branch4_points_random1 = self.stage4_14(index_points(branch1_points.permute(0,2,1), branch4_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        # # branch4_xyz, branch4_idx = random_sample(branch2_xyz_FP, 128)  ### random sample 128
        # # branch4_xyz = branch4_xyz.permute(0, 2, 1)  ### B C N
        # # branch4_points_random2 = self.stage4_24(branch4_xyz.unsqueeze(-1)).squeeze(-1)
        # branch4_points_random2 = self.stage4_24(index_points(branch2_points_FP.permute(0,2,1), branch4_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        # # branch3_xyz = branch3_xyz.permute(0, 2, 1)  ### B N C
        # # branch3_xyz_FP = branch3_xyz_FP.permute(0, 2, 1)  ### B N C
        # # branch4_xyz, branch4_idx = random_sample(branch3_xyz_FP, 128)  ### random sample 128
        # # branch4_xyz = branch4_xyz.permute(0, 2, 1)  ### B C N
        # # branch4_points_random3 = self.stage4_34(branch4_xyz.unsqueeze(-1)).squeeze(-1)
        # branch4_points_random3 = self.stage4_34(index_points(branch3_points_FP.permute(0,2,1), branch4_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        #
        # # branch4_points_FP = self.stage4_conv4(torch.cat((branch4_points_FP, branch4_points_random1, branch4_points_random2, branch4_points_random3), 1).unsqueeze(-1)).squeeze(-1)
        # branch4_points_FP = self.stage4_conv4((branch4_points_FP + branch4_points_random1 + branch4_points_random2 + branch4_points_random3).unsqueeze(-1)).squeeze(-1)
        #
        #
        #
        # temp43_point_stage4 = self.stage4_fuse43(xyz, xyz, branch3_points_FP, branch4_points_FP)
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
        # # branch3_xyz, branch3_idx = random_sample(branch1_xyz, 256)  ### random sample 256
        # # branch3_xyz = branch3_xyz.permute(0, 2, 1)  ### B C N
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B C N
        # # branch3_points_random1 = self.stage4_13(branch3_xyz.unsqueeze(-1)).squeeze(-1)
        # branch3_points_random1 = self.stage4_13(index_points(branch1_points.permute(0,2,1), branch3_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        # # branch2_xyz_FP = branch2_xyz_FP.permute(0, 2, 1)  ### B N C
        # # branch3_xyz, branch3_idx = random_sample(branch2_xyz_FP, 256)  ### random sample 256
        # # branch3_xyz = branch3_xyz.permute(0, 2, 1)  ### B C N
        # # branch3_points_random2 = self.stage4_23(branch3_xyz.unsqueeze(-1)).squeeze(-1)
        # branch3_points_random2 = self.stage4_23(index_points(branch2_points_FP.permute(0,2,1), branch3_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        #
        # # branch3_points_FP = self.stage4_conv3(torch.cat((branch3_points_FP, branch3_points_random1, branch3_points_random2, temp43_point_stage4), 1).unsqueeze(-1)).squeeze(-1)
        # branch3_points_FP = self.stage4_conv3((branch3_points_FP + branch3_points_random1 + branch3_points_random2 + temp43_point_stage4).unsqueeze(-1)).squeeze(-1)
        #
        #
        #
        #
        # temp42_points_stage4 = self.stage4_fuse42(xyz, xyz, branch2_points_FP, branch4_points_FP)
        # temp32_points_stage4 = self.stage4_fuse32(xyz, xyz, branch2_points_FP, branch3_points_FP)
        #
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
        # # branch2_xyz, branch2_idx = random_sample(branch1_xyz, 512)  ### random sample 512
        # # branch2_xyz = index_points(branch1_xyz, farthest_point_sample(branch1_xyz, 512))  ## FPS
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B C N
        # # branch2_xyz = branch2_xyz.permute(0, 2, 1)  ### B C N
        # # branch2_points_random = self.stage4_12(branch2_xyz.unsqueeze(-1)).squeeze(-1)
        # branch2_points_random = self.stage4_12(index_points(branch1_points.permute(0,2,1), branch2_idx_FP).permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        # # branch2_points_FP = self.stage4_conv2(torch.cat((branch2_points_FP, branch2_points_random, temp32_points_stage4, temp42_points_stage4), 1).unsqueeze(-1)).squeeze(-1)
        # branch2_points_FP = self.stage4_conv2((branch2_points_FP + branch2_points_random + temp32_points_stage4 + temp42_points_stage4).unsqueeze(-1)).squeeze(-1)
        #
        #
        #
        # temp41_points_stage4 = self.stage4_fuse41(xyz, xyz, branch1_points, branch4_points_FP)
        # temp31_points_stage4 = self.stage4_fuse31(xyz, xyz, branch1_points, branch3_points_FP)
        # temp21_points_stage4 = self.stage4_fuse21(xyz, xyz, branch1_points, branch2_points_FP)
        # # temp31_points_stage4 = temp21_points_stage4
        # # temp41_points_stage4 = temp21_points_stage4
        # # branch1_points = self.stage4_conv1(torch.cat((branch1_points, temp21_points_stage4, temp31_points_stage4, temp41_points_stage4), 1).unsqueeze(-1)).squeeze(-1)
        # branch1_points = self.stage4_conv1((branch1_points + temp21_points_stage4 + temp31_points_stage4 + temp41_points_stage4).unsqueeze(-1)).squeeze(-1)
        #
        # # branch1_points_temp = self.initial4_la(xyz, branch1_xyz)
        # # x4 = torch.cat((self.conv_x4(branch1_points.unsqueeze(-1)), self.conv_x4_temp(branch1_points_temp.unsqueeze(-1))), dim=1)
        # x4 = self.conv_x4(branch1_points.unsqueeze(-1))
        # # x4 = branch1_points.unsqueeze(-1)
        # #######################################################################################


        final = self.final(torch.cat((x1, x2, x3, x4), 1))

        x1 = F.adaptive_max_pool1d(final.squeeze(-1), 1)
        x2 = F.adaptive_avg_pool1d(final.squeeze(-1), 1)
        final_fuse = torch.cat((x1, x2), 1).unsqueeze(-1)

        final_fuse = self.final_class(final_fuse).squeeze(-1).squeeze(-1)


        # ###final stage classification
        # branch1_points = branch1_points.unsqueeze(-1)
        # branch1_points = self.incre_modules[0](branch1_points).squeeze(-1)
        #
        # # branch1_x1 = F.adaptive_max_pool1d(branch1_points, 1)
        # # branch1_x2 = F.adaptive_avg_pool1d(branch1_points, 1)
        # # branch1_points_final = self.fuse_modules[0](torch.cat((branch1_x1, branch1_x2), 1).unsqueeze(-1))
        # ###################################################
        #
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
        # # branch2_xyz, branch2_idx = random_sample(branch1_xyz, 512)  ### random sample 512
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B C N
        # # branch2_xyz = branch2_xyz.permute(0, 2, 1)  ### B C N
        # final_temp = index_points(branch1_points.permute(0, 2, 1), branch2_idx_FP).permute(0, 2, 1).unsqueeze(-1)
        # # final_temp = self.downsamp_modules[0](self.final_fuse1(branch1_xyz, branch1_points, branch2_xyz, branch2_idx).unsqueeze(-1))
        # # final_temp = self.downsamp_modules[0](index_points(branch1_points.permute(0,2,1), branch2_idx).permute(0,2,1).unsqueeze(-1))
        # # final_temp = index_points(branch1_points.permute(0,2,1), branch2_idx).permute(0,2,1).unsqueeze(-1)
        #
        # branch2_points_FP = branch2_points_FP.unsqueeze(-1)
        # branch2_points_FP = (self.incre_modules[1](branch2_points_FP) + self.fuse_modules[0](final_temp)).squeeze(-1)
        # # branch2_points = (final_temp + branch2_points).squeeze(-1)
        #
        # # branch2_x1 = F.adaptive_max_pool1d(branch2_points, 1)
        # # branch2_x2 = F.adaptive_avg_pool1d(branch2_points, 1)
        # # branch2_points = self.fuse_modules[1](torch.cat((branch2_points_FP, branch2_points_FP), 1).unsqueeze(-1))
        # ############################################################# 1-2
        #
        # # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
        # # branch3_xyz, branch3_idx = random_sample(branch2_xyz_FP, 256)  ### random sample 256
        # # branch2_xyz = branch2_xyz.permute(0, 2, 1)  ### B C N
        # # branch3_xyz = branch3_xyz.permute(0, 2, 1)  ### B C N
        #
        # # final_fuse = self.downsamp_modules[1](self.final_fuse2(branch2_xyz, final_fuse, branch3_xyz, branch3_idx).unsqueeze(-1))
        # final_temp = index_points(branch2_points_FP.permute(0, 2, 1), branch3_idx_FP).permute(0, 2, 1).unsqueeze(-1)
        # # final_temp = index_points(branch2_points.permute(0,2,1), branch3_idx).permute(0,2,1).unsqueeze(-1)
        #
        # branch3_points_FP = branch3_points_FP.unsqueeze(-1)
        # branch3_points_FP = (self.incre_modules[2](branch3_points_FP) + self.fuse_modules[1](final_temp)).squeeze(-1)
        # # branch3_points = (final_temp + branch3_points).squeeze(-1)
        #
        # # branch3_x1 = F.adaptive_max_pool1d(branch3_points, 1)
        # # branch3_x2 = F.adaptive_avg_pool1d(branch3_points, 1)
        # # branch3_points = self.fuse_modules[2](torch.cat((branch3_x1, branch3_x2), 1).unsqueeze(-1))
        # ############################################################# 2-3
        #
        # # branch3_xyz = branch3_xyz.permute(0, 2, 1)  ### B N C
        # # branch4_xyz, branch4_idx = random_sample(branch1_xyz, 128)  ### random sample 256
        # # branch3_xyz = branch3_xyz.permute(0, 2, 1)  ### B C N
        # # branch4_xyz = branch4_xyz.permute(0, 2, 1)  ### B C N
        #
        # final_temp = index_points(branch3_points_FP.permute(0, 2, 1), branch4_idx_FP).permute(0, 2, 1).unsqueeze(-1)
        # # final_temp = index_points(branch1_points.permute(0, 2, 1), branch4_idx).permute(0, 2, 1).unsqueeze(-1)
        #
        # branch4_points_FP = branch4_points_FP.unsqueeze(-1)
        # branch4_points_FP = (self.incre_modules[3](branch4_points_FP) + self.fuse_modules[2](final_temp)).squeeze(-1)
        # # branch4_points = (final_temp + branch4_points).squeeze(-1)
        #
        # # branch4_x1 = F.adaptive_max_pool1d(branch4_points, 1)
        # # branch4_x2 = F.adaptive_avg_pool1d(branch4_points, 1)
        # # branch4_points = self.fuse_modules[3](torch.cat((branch4_x1, branch4_x2), 1).unsqueeze(-1))
        # ############################################################# 3-4
        #
        # x1 = F.adaptive_max_pool1d(branch4_points_FP, 1)
        # x2 = F.adaptive_avg_pool1d(branch4_points_FP, 1)
        # final_fuse = torch.cat((x1, x2), 1).unsqueeze(-1)
        #
        # # final_fuse = self.final_class(branch1_points_final + branch2_points + branch3_points + branch4_points).squeeze(-1).squeeze(-1)
        # # final_fuse = self.final_class(torch.cat((branch1_points_final, branch2_points, branch3_points, branch4_points), 1)).squeeze(-1).squeeze(-1)
        # final_fuse = self.final_class(final_fuse).squeeze(-1).squeeze(-1)

        return xyz, final_fuse

class KeepHighResolutionModulePartSeg(nn.Module):

    def __init__(self, data_C, b1_C, b2_C, b3_C, b4_C):
        super(KeepHighResolutionModulePartSeg, self).__init__()

        self.local_num_neighbors = [16,32]
        self.neighbour = 40
        # self.transform_net = Transform_Net()
        # self.transform_la = LocalAggregation(3 * 2 + 7, b1_C, [self.neighbour, ])


        # self.fc_start = nn.Linear(3, 8)
        # self.bn_start = nn.Sequential(
        #     nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
        #     nn.LeakyReLU(0.2)
        # )
        #
        # # encoding layers
        # self.encoder = nn.ModuleList([
        #     LocalFeatureAggregation(8, 16, num_neighbors=16),
        #     LocalFeatureAggregation(32, 64, num_neighbors=16),
        #     LocalFeatureAggregation(128, 128, num_neighbors=16),
        #     LocalFeatureAggregation(256, 256, num_neighbors=16)
        # ])
        #
        # self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())
        #
        # self.decimation = 4



        # self.start = SharedMLP(7, b1_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.start_la1 = LocalAggregationPart(3 * 2 + 7, b1_C, [self.neighbour, ])
        self.start_la2 = LocalAggregationPart(3 * 2 + 7, b2_C, [self.neighbour, ])
        self.start_la3 = LocalAggregationPart(3 * 2 + 7, b3_C, [self.neighbour, ])
        self.start_la4 = LocalAggregationPart(3 * 2 + 7, b4_C, [self.neighbour, ])

        self.start_fuse = Fuse_Stage(64, 64, 64, 64)


        ###stage2
        self.stage2_la1 = LocalAggregationPart(b1_C, b1_C, [self.neighbour, ])
        self.stage2_la2 = LocalAggregationPart(b2_C, b2_C, [self.neighbour, ])
        self.stage2_la3 = LocalAggregationPart(b3_C, b3_C, [self.neighbour, ])
        self.stage2_la4 = LocalAggregationPart(b4_C, b4_C, [self.neighbour, ])

        self.stage2_fuse = Fuse_Stage(64, 64, 64, 64)

        ###stage3
        self.stage3_la1 = LocalAggregationPart(b1_C, b1_C, [self.neighbour, ])
        self.stage3_la2 = LocalAggregationPart(b2_C, b2_C, [self.neighbour, ])
        self.stage3_la3 = LocalAggregationPart(b3_C, b3_C, [self.neighbour, ])
        self.stage3_la4 = LocalAggregationPart(b4_C, b4_C, [self.neighbour, ])

        self.stage3_fuse = Fuse_Stage(64, 64, 64, 64)


        ###stage4
        self.stage4_la1 = LocalAggregationPart(b1_C, b1_C, [self.neighbour, ])
        self.stage4_la2 = LocalAggregationPart(b2_C, b2_C, [self.neighbour, ])
        self.stage4_la3 = LocalAggregationPart(b3_C, b3_C, [self.neighbour, ])
        self.stage4_la4 = LocalAggregationPart(b4_C, b4_C, [self.neighbour, ])

        self.stage4_fuse = Fuse_Stage(64, 64, 64, 64)



        self.conv6 = SharedMLP(512, 1024, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = SharedMLP(16, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))




        self.final_class = SharedMLP(1024, 1024, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.final_pool = AttentivePooling(1024, 1024, 256)

        self.conv_x1 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.conv_x2 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.conv_x2_temp = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.conv_x3 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.conv_x3_temp = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.conv_x4 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.conv_x4_temp = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.final = SharedMLP(512, 512, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))




    def forward(self, xyz, points=None, label=None):



        batch_size = xyz.size(0)
        num_points = xyz.size(2)


        branch1_xyz = xyz   ### B C N

        if points is not None:
            branch1_norm = points
        else:
            branch1_norm = None

        # x0 = self.transform_la(branch1_xyz, trans=True)
        # trans = self.transform_net(x0)
        # branch1_xyz = torch.bmm(branch1_xyz.transpose(2,1), trans).transpose(2,1)


        # branch2_idx_FP = farthest_point_sample(branch1_xyz, 512)
        # branch2_xyz_FP = index_points(branch1_xyz, branch2_idx_FP)  # FPS generate branch2

        # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
        # branch2_xyz_FP = branch2_xyz_FP.permute(0, 2, 1)  ### B C N

        # branch1_points = self.stage2_1(branch1_xyz.unsqueeze(-1)).squeeze(-1)

        branch1_points = self.start_la1(branch1_xyz, norm=branch1_norm)

        # x1 = self.conv_x1(branch1_points.unsqueeze(-1))


        branch2_idx_FP = farthest_point_sample(branch1_xyz.permute(0,2,1).contiguous(), 512)
        branch2_xyz = index_points(branch1_xyz.permute(0,2,1).contiguous(), branch2_idx_FP)
        if points is not None:
            branch2_norm = index_points(branch1_norm.permute(0,2,1), branch2_idx_FP).permute(0,2,1).contiguous()
        else:
            branch2_norm = None
        branch2_points_FP = self.start_la2(branch2_xyz.permute(0, 2, 1).contiguous(), norm=branch2_norm)  # FPS generate branch2
        # branch2_points_FP = self.start_la2(branch2_xyz.permute(0,2,1).contiguous()) # FPS generate branch2


        branch3_idx_FP = farthest_point_sample(branch2_xyz, 256)
        branch3_xyz = index_points(branch2_xyz, branch3_idx_FP)
        if points is not None:
            branch3_norm = index_points(branch2_norm.permute(0,2,1), branch3_idx_FP).permute(0,2,1).contiguous()
        else:
            branch3_norm = None
        branch3_points_FP = self.start_la3(branch3_xyz.permute(0,2,1).contiguous(), norm=branch3_norm)


        branch4_idx_FP = farthest_point_sample(branch3_xyz, 128)
        branch4_xyz = index_points(branch3_xyz, branch4_idx_FP)
        if points is not None:
            branch4_norm = index_points(branch3_norm.permute(0,2,1), branch4_idx_FP).permute(0,2,1).contiguous()
        else:
            branch4_norm = None
        branch4_points_FP = self.start_la4(branch4_xyz.permute(0,2,1).contiguous(), norm=branch4_norm)


        ################ fuse
        branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.start_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
                                                                                             branch2_idx_FP, branch3_idx_FP, branch4_idx_FP)

        x1 = self.conv_x1(branch1_points.unsqueeze(-1).contiguous())

        branch1_xyz = branch1_xyz.permute(0,2,1).contiguous()

        ############## Local Aggregation  stage2
        branch1_points = self.stage2_la1(branch1_points, branch1_xyz, norm=branch1_norm)
        branch2_points_FP = self.stage2_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm) # FPS generate branch2
        branch3_points_FP = self.stage2_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
        branch4_points_FP = self.stage2_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)


        ##################  Fuse
        branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage2_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
                                                                                             branch2_idx_FP, branch3_idx_FP, branch4_idx_FP)

        x2 = self.conv_x2(branch1_points.unsqueeze(-1).contiguous())


        ############## Local Aggregation  stage3
        branch1_points = self.stage3_la1(branch1_points, branch1_xyz, norm=branch1_norm)
        branch2_points_FP = self.stage3_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm) # FPS generate branch2
        branch3_points_FP = self.stage3_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
        branch4_points_FP = self.stage3_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)


        ##################  Fuse
        branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage3_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
                                                                                             branch2_idx_FP, branch3_idx_FP, branch4_idx_FP)

        x3 = self.conv_x3(branch1_points.unsqueeze(-1).contiguous())



        ############## Local Aggregation  stage4
        branch1_points = self.stage4_la1(branch1_points, branch1_xyz, norm=branch1_norm)
        branch2_points_FP = self.stage4_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm) # FPS generate branch2
        branch3_points_FP = self.stage4_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
        branch4_points_FP = self.stage4_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)


        ##################  Fuse
        branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage4_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
                                                                                             branch2_idx_FP, branch3_idx_FP, branch4_idx_FP)

        x4 = self.conv_x4(branch1_points.unsqueeze(-1).contiguous())



        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv6(x).squeeze(-1)
        x = x.max(dim=-1, keepdim=True)[0]

        label = label.view(batch_size, -1, 1, 1)
        label = self.conv7(label).squeeze(-1)

        x = torch.cat((x, label), dim=1)
        x = x.repeat(1, 1, num_points).unsqueeze(-1)




        final = torch.cat((x, x1, x2, x3, x4), 1)


        return xyz, final

class KeepHighResolutionModuleSemiSeg(nn.Module):

    def __init__(self, data_C, b1_C, b2_C, b3_C, b4_C):
        super(KeepHighResolutionModulePartSeg, self).__init__()

        self.local_num_neighbors = [16,32]
        self.neighbour = 40
        # self.transform_net = Transform_Net()
        # self.transform_la = LocalAggregation(3 * 2 + 7, b1_C, [self.neighbour, ])


        # self.fc_start = nn.Linear(3, 8)
        # self.bn_start = nn.Sequential(
        #     nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
        #     nn.LeakyReLU(0.2)
        # )
        #
        # # encoding layers
        # self.encoder = nn.ModuleList([
        #     LocalFeatureAggregation(8, 16, num_neighbors=16),
        #     LocalFeatureAggregation(32, 64, num_neighbors=16),
        #     LocalFeatureAggregation(128, 128, num_neighbors=16),
        #     LocalFeatureAggregation(256, 256, num_neighbors=16)
        # ])
        #
        # self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())
        #
        # self.decimation = 4



        # self.start = SharedMLP(7, b1_C, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.start_la1 = LocalAggregation(3 * 2 + 7, b1_C, [self.neighbour, ])
        self.start_la2 = LocalAggregation(3 * 2 + 7, b2_C, [self.neighbour, ])
        self.start_la3 = LocalAggregation(3 * 2 + 7, b3_C, [self.neighbour, ])
        self.start_la4 = LocalAggregation(3 * 2 + 7, b4_C, [self.neighbour, ])

        self.start_fuse = Fuse_Stage(64, 64, 64, 64)


        ###stage2
        self.stage2_la1 = LocalAggregation(b1_C, b1_C, [self.neighbour, ])
        self.stage2_la2 = LocalAggregation(b2_C, b2_C, [self.neighbour, ])
        self.stage2_la3 = LocalAggregation(b3_C, b3_C, [self.neighbour, ])
        self.stage2_la4 = LocalAggregation(b4_C, b4_C, [self.neighbour, ])

        self.stage2_fuse = Fuse_Stage(64, 64, 64, 64)

        ###stage3
        self.stage3_la1 = LocalAggregation(b1_C, b1_C, [self.neighbour, ])
        self.stage3_la2 = LocalAggregation(b2_C, b2_C, [self.neighbour, ])
        self.stage3_la3 = LocalAggregation(b3_C, b3_C, [self.neighbour, ])
        self.stage3_la4 = LocalAggregation(b4_C, b4_C, [self.neighbour, ])

        self.stage3_fuse = Fuse_Stage(64, 64, 64, 64)


        ###stage4
        self.stage4_la1 = LocalAggregation(b1_C, b1_C, [self.neighbour, ])
        self.stage4_la2 = LocalAggregation(b2_C, b2_C, [self.neighbour, ])
        self.stage4_la3 = LocalAggregation(b3_C, b3_C, [self.neighbour, ])
        self.stage4_la4 = LocalAggregation(b4_C, b4_C, [self.neighbour, ])

        self.stage4_fuse = Fuse_Stage(64, 64, 64, 64)



        self.conv6 = SharedMLP(512, 1024, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = SharedMLP(16, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))




        self.final_class = SharedMLP(1024, 1024, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.final_pool = AttentivePooling(1024, 1024, 256)

        self.conv_x1 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.conv_x2 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.conv_x2_temp = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.conv_x3 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.conv_x3_temp = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.conv_x4 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))
        # self.conv_x4_temp = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))

        self.final = SharedMLP(512, 512, bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2))




    def forward(self, xyz, points=None, label=None):



        batch_size = xyz.size(0)
        num_points = xyz.size(2)


        branch1_xyz = xyz   ### B C N

        if points is not None:
            branch1_norm = points
        else:
            branch1_norm = None

        # x0 = self.transform_la(branch1_xyz, trans=True)
        # trans = self.transform_net(x0)
        # branch1_xyz = torch.bmm(branch1_xyz.transpose(2,1), trans).transpose(2,1)


        # branch2_idx_FP = farthest_point_sample(branch1_xyz, 512)
        # branch2_xyz_FP = index_points(branch1_xyz, branch2_idx_FP)  # FPS generate branch2

        # branch1_xyz = branch1_xyz.permute(0, 2, 1)  ### B N C
        # branch2_xyz_FP = branch2_xyz_FP.permute(0, 2, 1)  ### B C N

        # branch1_points = self.stage2_1(branch1_xyz.unsqueeze(-1)).squeeze(-1)

        branch1_points = self.start_la1(branch1_xyz, norm=branch1_norm)

        # x1 = self.conv_x1(branch1_points.unsqueeze(-1))


        branch2_idx_FP = farthest_point_sample(branch1_xyz.permute(0,2,1).contiguous(), 512)
        branch2_xyz = index_points(branch1_xyz.permute(0,2,1).contiguous(), branch2_idx_FP)
        if points is not None:
            branch2_norm = index_points(branch1_norm.permute(0,2,1), branch2_idx_FP).permute(0,2,1).contiguous()
        else:
            branch2_norm = None
        branch2_points_FP = self.start_la2(branch2_xyz.permute(0, 2, 1).contiguous(), norm=branch2_norm)  # FPS generate branch2
        # branch2_points_FP = self.start_la2(branch2_xyz.permute(0,2,1).contiguous()) # FPS generate branch2


        branch3_idx_FP = farthest_point_sample(branch2_xyz, 256)
        branch3_xyz = index_points(branch2_xyz, branch3_idx_FP)
        if points is not None:
            branch3_norm = index_points(branch2_norm.permute(0,2,1), branch3_idx_FP).permute(0,2,1).contiguous()
        else:
            branch3_norm = None
        branch3_points_FP = self.start_la3(branch3_xyz.permute(0,2,1).contiguous(), norm=branch3_norm)


        branch4_idx_FP = farthest_point_sample(branch3_xyz, 128)
        branch4_xyz = index_points(branch3_xyz, branch4_idx_FP)
        if points is not None:
            branch4_norm = index_points(branch3_norm.permute(0,2,1), branch4_idx_FP).permute(0,2,1).contiguous()
        else:
            branch4_norm = None
        branch4_points_FP = self.start_la4(branch4_xyz.permute(0,2,1).contiguous(), norm=branch4_norm)


        ################ fuse
        branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.start_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
                                                                                             branch2_idx_FP, branch3_idx_FP, branch4_idx_FP)

        x1 = self.conv_x1(branch1_points.unsqueeze(-1).contiguous())

        branch1_xyz = branch1_xyz.permute(0,2,1).contiguous()

        ############## Local Aggregation  stage2
        branch1_points = self.stage2_la1(branch1_points, branch1_xyz, norm=branch1_norm)
        branch2_points_FP = self.stage2_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm) # FPS generate branch2
        branch3_points_FP = self.stage2_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
        branch4_points_FP = self.stage2_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)


        ##################  Fuse
        branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage2_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
                                                                                             branch2_idx_FP, branch3_idx_FP, branch4_idx_FP)

        x2 = self.conv_x2(branch1_points.unsqueeze(-1).contiguous())


        ############## Local Aggregation  stage3
        branch1_points = self.stage3_la1(branch1_points, branch1_xyz, norm=branch1_norm)
        branch2_points_FP = self.stage3_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm) # FPS generate branch2
        branch3_points_FP = self.stage3_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
        branch4_points_FP = self.stage3_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)


        ##################  Fuse
        branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage3_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
                                                                                             branch2_idx_FP, branch3_idx_FP, branch4_idx_FP)

        x3 = self.conv_x3(branch1_points.unsqueeze(-1).contiguous())



        ############## Local Aggregation  stage4
        branch1_points = self.stage4_la1(branch1_points, branch1_xyz, norm=branch1_norm)
        branch2_points_FP = self.stage4_la2(branch2_points_FP, branch2_xyz, norm=branch2_norm) # FPS generate branch2
        branch3_points_FP = self.stage4_la3(branch3_points_FP, branch3_xyz, norm=branch3_norm)
        branch4_points_FP = self.stage4_la4(branch4_points_FP, branch4_xyz, norm=branch4_norm)


        ##################  Fuse
        branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP = self.stage4_fuse(branch1_points, branch2_points_FP, branch3_points_FP, branch4_points_FP,
                                                                                             branch2_idx_FP, branch3_idx_FP, branch4_idx_FP)

        x4 = self.conv_x4(branch1_points.unsqueeze(-1).contiguous())



        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv6(x).squeeze(-1)
        x = x.max(dim=-1, keepdim=True)[0]

        # label = label.view(batch_size, -1, 1, 1)
        # label = self.conv7(label).squeeze(-1)

        # x = torch.cat((x, label), dim=1)
        x = x.repeat(1, 1, num_points).unsqueeze(-1)




        final = torch.cat((x, x1, x2, x3, x4), 1)


        return xyz, final

class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = 20

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
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

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
        #
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x

#########################################################


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetSetAbstractionMsgTest(nn.Module):    ##################################
    def __init__(self, radiusQuad, inputQuad, outputQuad, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgTest, self).__init__()
        self.npoint = npoint

        self.radiusQuad = radiusQuad  #######################
        # self.outputQuad = outputQuad  ###################

        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 32  #############################
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

        self.conv1 = nn.Sequential(
            conv_bn(inputQuad, outputQuad, [1, 2], [1, 2]),
            conv_bn(outputQuad, outputQuad, [1, 2], [1, 2]),
            conv_bn(outputQuad, outputQuad, [1, 2], [1, 2])
        )

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        # """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape



        a = torch.rand([1, 5, 3])
        sqrdists = square_distance(a, a)

        maskT = sqrdists < 0.5

        idxtemp = torch.arange(5).repeat(5, 1).contiguous().repeat(1, 1, 1)

        idx = torch.tensor(999).repeat(1,5,5)

        idx[maskT] = idxtemp[maskT]



        _, group_idx = torch.topk(sqrdists, 2, dim=-1, largest=False, sorted=False)


        grouped_xyz = index_points(a, group_idx)



        ###################################################################### 8 quadrant
        radiusQuad = self.radiusQuad
        _, grouped_pointsQuad, grouped_edgeQuad, idx = pointsift_group(radiusQuad, xyz, points)

        grouped_pointsQuad = grouped_pointsQuad.permute(0, 3, 1, 2).contiguous()
        new_pointsQuad = self.conv1(grouped_pointsQuad)
        new_pointsQuad = new_pointsQuad.squeeze(-1).permute(0, 2, 1).contiguous()

        ######################################################################
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))

        # group_idx = query_ball_point(0.1, 32, xyz, new_xyz)  ############################
        # grouped_points = index_points(new_points, group_idx)  #############################
        # new_points = torch.max(grouped_points, 2)[0]   ############################

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)

            grouped_points = index_points(new_pointsQuad, group_idx)  #############################
            # grouped_xyz = index_points(xyz, group_idx)   ###################%%%%%%%%%%%%
            # grouped_xyz -= new_xyz.view(B, S, 1, C)    ##############%%%%%%%%%%%%%%%%%
            #
            # if points is not None:
            #     grouped_points = index_points(points, group_idx)
            #     grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            # else:
            #     grouped_points = grouped_xyz



            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()



        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        # """



        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


# class PointNetFeaturePropagation(nn.Module):
#     def __init__(self, in_channel, mlp):
#         super(PointNetFeaturePropagation, self).__init__()
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm1d(out_channel))
#             last_channel = out_channel
#
#     def forward(self, xyz1, xyz2, points1, points2):
#         """
#         Input:
#             xyz1: input points position data, [B, C, N]
#             xyz2: sampled input points position data, [B, C, S]
#             points1: input points data, [B, D, N]
#             points2: input points data, [B, D, S]
#         Return:
#             new_points: upsampled points data, [B, D', N]
#         """
#         xyz1 = xyz1.permute(0, 2, 1)
#         xyz2 = xyz2.permute(0, 2, 1)
#
#         points2 = points2.permute(0, 2, 1)
#         B, N, C = xyz1.shape
#         _, S, _ = xyz2.shape
#
#         if S == 1:
#             interpolated_points = points2.repeat(1, N, 1)
#         else:
#             dists = square_distance(xyz1, xyz2)
#             dists, idx = dists.sort(dim=-1)
#             dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
#
#             dist_recip = 1.0 / (dists + 1e-8)
#             norm = torch.sum(dist_recip, dim=2, keepdim=True)
#             weight = dist_recip / norm
#             interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
#
#         if points1 is not None:
#             points1 = points1.permute(0, 2, 1)
#             new_points = torch.cat([points1, interpolated_points], dim=-1)
#         else:
#             new_points = interpolated_points
#
#         new_points = new_points.permute(0, 2, 1)
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             new_points = F.relu(bn(conv(new_points)))
#         return new_points



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


        #############################################################################################


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

