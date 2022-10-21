import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation, KeepHighResolutionModuleSemiSeg


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)


        ##########################################################################

        # self.keepHigh = KeepHighResolutionModuleSemiSeg(3, 64, 64, 64, 64)  ########################
        #
        # self.bn7 = nn.BatchNorm2d(512)
        # self.bn8 = nn.BatchNorm2d(256)
        # self.bn9 = nn.BatchNorm2d(128)
        #
        #
        # self.conv7 = nn.Sequential(nn.Conv2d(1563, 512, kernel_size=1, bias=False), self.bn7, nn.LeakyReLU(negative_slope=0.2))
        # # self.drop1 = nn.Dropout(0.5)     ########################
        # self.conv8 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False), self.bn8, nn.LeakyReLU(negative_slope=0.2))
        # self.drop1 = nn.Dropout(0.5)     ########################
        # self.conv9 = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1, bias=False), self.bn9, nn.LeakyReLU(negative_slope=0.2))
        # self.conv10 = nn.Conv2d(128, num_classes, kernel_size=1, bias=False)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        ########################################################################


        # branch1_xyz, final_points = self.keepHigh(xyz, label=None)  ########################
        # x = self.conv7(final_points)   ########################
        # x = self.conv8(x)    ########################
        # x = self.drop1(self.conv9(x)).squeeze(-1)                     ########################
        # # x = self.conv11(x).squeeze(-1)  ########################
        #
        # # x = F.log_softmax(x, dim=1)         ########################
        # x = x.permute(0, 2, 1)


        return x, xyz


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target)

        ########################################################################
        # target = target.contiguous().view(-1)
        #
        # eps = 0.2
        # n_class = pred.size(1)
        #
        # one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        # one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        # log_prb = F.log_softmax(pred, dim=1)
        #
        # total_loss = -(one_hot * log_prb).sum(dim=1).mean()
        ###########################################################################
        # print('loss: ', total_loss)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))