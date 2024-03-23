import os
import sys
import numpy as np
from random import choice
from scipy.io import loadmat

import torch
from torch.utils.data.dataloader import default_collate

ten_label_indexes = [17, 9, 36, 20, 3, 16, 34, 38, 23, 15]
ten_label_names = ['airplane', 'bed', 'bookshelf', 'bottle', 'chair', 'monitor', 'sofa', 'table', 'toilet', 'vase']


class ModelNet40():
    def __init__(self, data_mat_file='../Data/modelnet10_250instances_1024.mat', attack_label='All', resample_num=-1, is_half_forward=False):
        self.data_root = data_mat_file
        self.attack_label = attack_label
        self.is_half_forward = is_half_forward

        if not os.path.isfile(self.data_root):
            assert False, 'No exists .mat file!'

        dataset = loadmat(self.data_root)
        data = torch.FloatTensor(dataset['data'])
        normal = torch.FloatTensor(dataset['normal'])
        label = dataset['label']

        if resample_num>0:
            tmp_data_set = []
            tmp_normal_set = []
            for j in range(data.size(0)):
                tmp_data, tmp_normal  = self.__farthest_points_normalized(data[j].t(), resample_num, normal[j].t())
                tmp_data_set.append(torch.from_numpy(tmp_data).t().float())
                tmp_normal_set.append(torch.from_numpy(tmp_normal).t().float())
            data = torch.stack(tmp_data_set)
            normal = torch.stack(tmp_normal_set)

        if attack_label in ten_label_names:
            for k, label_name in enumerate(ten_label_names):
                if attack_label == label_name:
                    self.start_index = k*25
                    self.data = data[k*25:(k+1)*25]
                    self.normal = normal[k*25:(k+1)*25]
                    self.label = label[k*25:(k+1)*25]
        elif attack_label == 'All':
            self.start_index = 0
            self.data = data
            self.normal = normal
            self.label = label
        elif attack_label == 'Untarget' or attack_label == 'Random':
            self.start_index = 0
            self.data = data
            self.normal = normal
            self.label = label
        else:
            assert False

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):

        if (self.attack_label in ten_label_names) or (self.attack_label == 'All'):
            label = self.label[index]

            target_labels = []
            for i in ten_label_indexes:
                if label != i:
                    target_labels.append(i)
            target_labels = torch.IntTensor(np.array(target_labels)).long()
            gt_labels = torch.IntTensor(label).long().expand_as(target_labels)
            assert target_labels.size(0)==9

            pc = self.data[index].contiguous().t()
            pcs = pc.unsqueeze(0).expand(9, -1, -1)

            normal = self.normal[index].contiguous().t()
            normals = normal.unsqueeze(0).expand(9, -1, -1)
            if self.is_half_forward:
                return [[pcs[:4,:,:], normals[:4,:,:], gt_labels[:4], target_labels[:4]],[pcs[4:,:,:], normals[4:,:,:], gt_labels[4:], target_labels[4:]]]
            else:
                return [pcs, normals, gt_labels, target_labels]

        elif (self.attack_label == 'Untarget'):
            label = self.label[index]
            gt_labels = torch.IntTensor(label).long()

            pc = self.data[index].contiguous().t()
            pcs = pc.unsqueeze(0).expand(1, -1, -1)

            normal = self.normal[index].contiguous().t()
            normals = normal.unsqueeze(0).expand(1, -1, -1)
            return [pcs, normals, gt_labels]

        elif (self.attack_label == 'Random'):
            label = self.label[index]
            gt_labels = torch.IntTensor(label).long()

            pc = self.data[index].contiguous().t()
            pcs = pc.unsqueeze(0).expand(1, -1, -1)

            normal = self.normal[index].contiguous().t()
            normals = normal.unsqueeze(0).expand(1, -1, -1)

            target_labels = torch.IntTensor([choice([i for i in range(0,40) if i not in [gt_labels.item()]])]).long()

            return [pcs, normals, gt_labels, target_labels]

    def __farthest_points_normalized(self, obj_points, num_points, normal):
        first = np.random.randint(len(obj_points))
        selected = [first]
        dists = np.full(shape = len(obj_points), fill_value = np.inf)

        for _ in range(num_points - 1):
            dists = np.minimum(dists, np.linalg.norm(obj_points - obj_points[selected[-1]][np.newaxis, :], axis = 1))
            selected.append(np.argmax(dists))
        res_points = np.array(obj_points[selected])
        res_normal = np.array(normal[selected])

        # normalize the points and faces
        avg = np.average(res_points, axis = 0)
        res_points = res_points - avg[np.newaxis, :]
        dists = np.max(np.linalg.norm(res_points, axis = 1), axis = 0)
        res_points = res_points / dists

        return res_points, res_normal


if __name__ == '__main__':
    from FGM.GeoA3_args import Average_meter, accuracy
    from model.GeoA3_PN import PointNet

    test_dataset_untarget = ModelNet40(data_mat_file='Data/modelnet10_250instances1024_PointNet.mat', attack_label='Untarget', resample_num=-1)
    test_loader_untarget = torch.utils.data.DataLoader(test_dataset_untarget, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    test_dataset_all = ModelNet40(data_mat_file='Data/modelnet10_250instances1024_PointNet.mat', attack_label='Untarget', resample_num=-1)
    test_loader_all = torch.utils.data.DataLoader(test_dataset_all, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    net = PointNet(40, npoint=1024).cuda()
    model_path = os.path.join('Pretrained/PointNet/1024/model_best.pth.tar')
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    test_acc = Average_meter()

    for i in range(10):
        assert (test_dataset_untarget[i][0] == test_dataset_all[i][0]).all()

    for i, data in enumerate(test_loader_untarget):
        pc, normal, gt_label = data[0], data[1], data[2]
        if pc.size(3) == 3:
            pc = pc.permute(0,1,3,2)
        if normal.size(3) == 3:
            normal = normal.permute(0,1,3,2)
        bs, l, _, n = pc.size()
        b = bs*l
        pc_ori = pc.view(b, 3, n).cuda()
        normal_ori = normal.view(b, 3, n).cuda()
        gt_target = gt_label.view(-1).cuda()

        with torch.no_grad():
            output = net(pc_ori)
        acc = accuracy(output.data, gt_target.data, topk=(1, ))
        test_acc.update(acc[0][0], output.size(0))
        print("PrecUntarget@1 {top1.avg:.3f}".format(top1=test_acc))


    for i, data in enumerate(test_loader_all):
        pc, normal, gt_label = data[0], data[1], data[2]
        if pc.size(3) == 3:
            pc = pc.permute(0,1,3,2)
        if normal.size(3) == 3:
            normal = normal.permute(0,1,3,2)
        bs, l, _, n = pc.size()
        b = bs*l
        pc_ori = pc.view(b, 3, n).cuda()
        normal_ori = normal.view(b, 3, n).cuda()
        gt_target = gt_label.view(-1).cuda()

        with torch.no_grad():
            output = net(pc_ori)
        acc = accuracy(output.data, gt_target.data, topk=(1, ))
        test_acc.update(acc[0][0], output.size(0))
        print("PrecAll@1 {top1.avg:.3f}".format(top1=test_acc))
