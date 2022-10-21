"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import os
import importlib
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))   #################%%%%%%%%%%%%%%%%%%%%%
# sys.path.append(os.path.join(ROOT_DIR, '/log/classification/2022-09-11_13-35_modulenet10/'))    ##########################



class PointcloudScale(object):  # input random scaling
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            scales = torch.from_numpy(xyz).float().cuda()
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], scales)
        return pc


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    # parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')   ##############%%%%%%%%%%%%%%%
    parser.add_argument('--log_dir', type=str, default='2022-09-16_05-10',  help='Experiment root')   ###########################
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting')
    return parser.parse_args()


def test(model, loader, num_class=40, vote_num=1):     ###################################
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    speed_list = []   #####################
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()



        for _ in range(vote_num):

            start_time = time.time()    ################
            pred, _ = classifier(points)
            end_time = time.time()   ################

            vote_pool += pred

        speed = end_time - start_time   ################
        speed_list.append(speed)   ################
        if j % 10 ==0:         ################
            # b = speed_list.pop(0)     ################
            a = sum(speed_list)/200   ################
            print("avg speed: ", a)   ################
            speed_list = []            ####################

        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc



# def test(model, loader, num_class=40, vote_num=10):
#     NUM_REPEAT = 200
#     best_acc = 0
#     best_ins = 0
#     pointscale = PointcloudScale(scale_low=0.95, scale_high=1.05)
#     for i in range(NUM_REPEAT):
#         mean_correct = []
#
#         class_acc = np.zeros((num_class,3))
#         for j, data in tqdm(enumerate(loader), total=len(loader)):
#             points, target = data
#             # target = target[:, 0]
#             #-----------------------change by wjw  to test the transformer------------------
#             # points = points.transpose(2, 1)
#             points, target = points.cuda(), target.cuda()
#             classifier = model.eval()
#             vote_pool = torch.zeros(target.size()[0],num_class).cuda()
#             for v in range(vote_num):
#                 if v > 0:
#                     points.data = pointscale(points.data)
#                 pred, _ = classifier(points.transpose(2,1))
#                 vote_pool += pred
#             pred = vote_pool/vote_num
#             pred_choice = pred.data.max(1)[1]
#             for cat in np.unique(target.cpu()):
#                 classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
#                 class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
#                 class_acc[cat,1]+=1
#             correct = pred_choice.eq(target.long().data).cpu().sum()
#             mean_correct.append(correct.item()/float(points.size()[0]))
#         class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
#         class_acc = np.mean(class_acc[:,2])
#         instance_acc = np.mean(mean_correct)
#         print(instance_acc,class_acc)
#         if instance_acc > best_ins:
#             best_ins = instance_acc
#             best_acc = class_acc
#
#
#     return best_ins,best_acc       #instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  ###############%%%%%%%%%%%%%%%%%
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  #######################################

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # data_path = 'data/modelnet40_normal_resampled/'   #################%%%%%%%%%%%%%%%%%%%%%
    data_path = '/Data_SSD/Chenru/pointdataset/modelnet40_normal_resampled/'    #######################################

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)


    classifier = model.get_model(num_class, normal_channel=args.use_normals)

    # #####################################################
    # if torch.cuda.device_count() > 1:
    #     print("we will use {} GPUs!".format(torch.cuda.device_count()))
    #     classifier = torch.nn.DataParallel(classifier)
    # #####################################################

    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])    #########################
    # classifier.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['model_state_dict'].items()})  #########%%%%%%%%%%%%%%%


    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
