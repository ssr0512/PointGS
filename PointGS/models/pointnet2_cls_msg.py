import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetSetAbstractionMsgTest, KeepHighResolutionModule, DGCNN_cls  ###############
# from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetSetAbstractionMsgTest  ##############%%%%%%%%%%%%
import torch

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel


        # self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])    #############%%%%%%%%%%%
        # self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])   ###########%%%%%%%%%%%%%%
        # self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        ####################################################################
        self.dgcnn = DGCNN_cls(50)    ##############################

        self.keepHigh = KeepHighResolutionModule(3, 64, 64, 64, 64)  ########################
        self.finalGroup = PointNetSetAbstraction(None, None, None, 32 + 3, [64, 128, 256], True)   ########################
        self.fc1 = nn.Linear(1024, 512)     ########################
        self.bn1 = nn.BatchNorm1d(512)     ########################
        self.drop1 = nn.Dropout(0.5)     ########################
        self.fc2 = nn.Linear(512, 256)     ########################
        self.bn2 = nn.BatchNorm1d(256)     ########################
        self.drop2 = nn.Dropout(0.5)     ########################
        self.fc3 = nn.Linear(256, num_class)    ########################
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)    ########################
        ###########################################################################

        # self.fc1 = nn.Linear(1024, 512)            #########%%%%%%%%%%
        # self.bn1 = nn.BatchNorm1d(512)             #########%%%%%%%%%%
        # self.drop1 = nn.Dropout(0.4)               #########%%%%%%%%%%
        # self.fc2 = nn.Linear(512, 256)             #########%%%%%%%%%%
        # self.bn2 = nn.BatchNorm1d(256)             #########%%%%%%%%%%
        # self.drop2 = nn.Dropout(0.5)               #########%%%%%%%%%%
        # self.fc3 = nn.Linear(256, num_class)       #########%%%%%%%%%%

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        ########################################################################################################
        # branch1_xyz, final_points = self.keepHigh(xyz, norm)  ########################
        # # final_xyz, final_points = self.finalGroup(branch1_xyz, branch1_points)    ########################
        # # x = final_points.view(B, 256)                      ########################
        # x = self.drop1(self.lrelu(self.bn1(self.fc1(final_points))))   ########################
        # x = self.drop2(self.lrelu(self.bn2(self.fc2(x))))    ########################
        # x = self.fc3(x)                     ########################

        # x = self.keepHigh(xyz, norm)
        # x = F.log_softmax(x, -1)         ########################
        #############################################################################################################

        # l1_xyz, l1_points = self.sa1(xyz, norm)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # x = l3_points.view(B, 1024)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)


        #######################################################
        x = self.dgcnn(xyz)


        # return x, l3_points    ##################%%%%%%%%%%%%%%%%%
        # return x, final_xyz   ################################
        return x, xyz


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        # total_loss = F.nll_loss(pred, target)  ##################%%%%%%%%%%%%%%%%%%%%%%%%%%

        ########################################################################
        target = target.contiguous().view(-1)

        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        total_loss = -(one_hot * log_prb).sum(dim=1).mean()
        ###########################################################################


        return total_loss


