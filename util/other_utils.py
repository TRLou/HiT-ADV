import os
import logging
import torch
import math
from typing import List
from util import dist_utils
from tqdm import tqdm
import numpy as np
from datetime import datetime
import open3d as o3d
from FGM.GeoA3_args import uniform_loss
from pytorch3d.loss import chamfer_distance


def eval_ASR(model, test_loader, args, val_attack):
    """ Evaluate Attack Success Rate
    """
    # ***** SET LOGGER ***** ***** ***** ***** ***** ***** ***** *****
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    logger = create_logger('./log', formatted_time, 'info')
    logger.info(f'ker_weight: {args.ker_weight}')
    logger.info(f'hide_weight: {args.hide_weight}')
    logger.info(f'budget: {args.budget}')
    logger.info(f'max_sigm:{args.max_sigm}')
    logger.info(f'min_sigm:{args.min_sigm}')
    logger.info(f'central_num:{args.central_num}')
    logger.info(f'attack_type:{args.attack_type}')
    # ***** END SET LOGGER ***** ***** ***** ***** ***** ***** ***** *****

    # ***** INITIALIZE METRIC ***** ***** ***** ***** ***** ***** ***** *****
    model.eval()
    at_num, at_denom = 0, 0

    num, denom = 0, 0
    batch = 0
    knn_dist_metric = dist_utils.KNNDist(k=4)
    uniform_dist_metric = uniform_loss
    curv_std_metric = dist_utils.CurvStdDist(k=4)

    knn_dist = 0
    uniform_dist = 0
    curv_std_dist = 0
    # ***** END INITIALIZE METRIC ***** ***** ***** ***** ***** ***** ***** *****

    # ***** EVALUATING ***** ***** ***** ***** ***** ***** ***** *****
    pbar = tqdm(test_loader)
    for i, (ori_data, label) in enumerate(pbar):
        # for ori_data, label in test_loader:
        batch += 1
        ori_data, label = ori_data.float().cuda(), label.long().cuda()

        ori_data = ori_data.transpose(1, 2).contiguous()
        batch_size = label.size(0)
        ori_data = ori_data.transpose(1, 2).contiguous()
        adv_data, _ = val_attack.attack(ori_data, label)

        # ***** ***** ***** *****

        if isinstance(adv_data, tuple):
            adv_data = adv_data[0]
        if not torch.is_tensor(adv_data):
            adv_data = torch.Tensor(adv_data).cuda()
        adv_data = adv_data.transpose(1, 2).contiguous()
        ori_data = ori_data.transpose(1, 2).contiguous()
        print('complete')

        if ori_data.shape[1] == 6:
            ori_normal = ori_data[:, 3:, :]
            ori_data = ori_data[:, :3, :]

        with torch.no_grad():
            knn_dist += knn_dist_metric.forward(pc=adv_data, weights=None, batch_avg=True)
            uniform_dist += uniform_dist_metric(adv_pc=adv_data, k=args.k)
            curv_std_dist += curv_std_metric.forward(ori_data=ori_data, adv_data=adv_data, ori_normal=ori_normal)

            if args.model.lower() == 'pointnet':
                logits, _ = model(ori_data)
                adv_logits, _ = model(adv_data)
            else:
                logits = model(ori_data)
                adv_logits = model(adv_data)
            ori_preds = torch.argmax(logits, dim=-1)
            adv_preds = torch.argmax(adv_logits, dim=-1)
            mask_ori = (ori_preds == label)
            mask_adv = (adv_preds == label)
            at_denom += mask_ori.sum().float().item()
            at_num += mask_ori.sum().float().item() - (mask_ori * mask_adv).sum().float().item()
            denom += float(batch_size)
            num += mask_adv.sum().float()
    # ***** END EVALUATING ***** ***** ***** ***** ***** ***** ***** *****

    # ***** LOGGING ***** ***** ***** ***** ***** ***** ***** *****
    ASR = at_num / (at_denom + 1e-9)
    logger.info(f'Overall attack success rate: {ASR}')
    logger.info(f'Overall KNN dist: {knn_dist / batch}')
    logger.info(f'Overall Uniform dist: {uniform_dist / batch}')
    logger.info(f'Overall CurvStd dist: {curv_std_dist / batch}')
    # ***** END LOGGING ***** ***** ***** ***** ***** ***** ***** *****

    return ASR


def reconstruct_from_pc(npoint, output_path, output_file_name, pc, output_type='mesh', normal=None,
                        reconstruct_type='PRS', central_points=None):
    # assert pc.size() == 2
    # assert pc.size(2) == 3
    # assert normal.size() == pc.size()
    print('pc', pc.shape)
    # print('central_points', central_points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # central_pcd = o3d.geometry.PointCloud()
    # central_pcd.points = o3d.utility.Vector3dVector(central_points)

    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal)

    if reconstruct_type == 'BPA':
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist

        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(
            [radius, radius * 2]))
        output_mesh = bpa_mesh
    elif reconstruct_type == 'PRS':
        poisson_mesh = \
            o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd, depth=9, width=0, scale=1.1,
                                                                      linear_fit=True, n_threads=-1)[0]
        bbox = pcd.get_axis_aligned_bounding_box()
        output_mesh = poisson_mesh.crop(bbox)

    o3d.io.write_triangle_mesh(os.path.join(output_path, output_file_name + ".obj"), output_mesh)

    output_mesh.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.visualization.draw_geometries([output_mesh], mesh_show_wireframe=True)

    if output_type == 'mesh':
        return output_mesh
    elif output_type == 'recon_pc':
        return o3d.geometry.TriangleMesh.sample_points_uniformly(output_mesh, number_of_points=npoint)
    else:
        raise NotImplementedError


def create_logger(save_path='', file_type='', level='debug'):
    if level == 'debug':
        _level = logging.DEBUG
    else:
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger


def save_checkpoint(now_epoch, net, optimizer_adv, file_name, lr_scheduler=None):
    checkpoint = {'epoch': now_epoch,
                  'state_dict': net.state_dict(),
                  'optimizer_adv_state_dict': optimizer_adv.state_dict(),
                  'lr_scheduler_state_dict': lr_scheduler.state_dict()}
    # if not os.path.exists(file_name):
    #     os.mkdir(file_name)
    torch.save(checkpoint, file_name)
    # link_name = os.path.join(args.model_dir, 'last.checkpoint')
    # print(link_name)
    # make_symlink(source=file_name, link_name=link_name)


def load_checkpoint(file_name, net=None, optimizer_adv=None, lr_scheduler=None):
    if os.path.isfile(file_name):
        print("=> loading checkpoint '{}'".format(file_name))
        check_point = torch.load(file_name, map_location={'cuda:2': 'cuda:0'})
        if net is not None:
            print('Loading network state dict')
            net.load_state_dict(check_point['last'])
        # if optimizer_adv is not None:
        #     print('Loading optimizer_adv state dict')
        #     optimizer_adv.load_state_dict(check_point['optimizer_adv_state_dict'])
        # if lr_scheduler is not None:
        #     print('Loading lr_scheduler state dict')
        #     # lr_scheduler.load_state_dict(check_point['lr_scheduler_state_dict'])

        # return check_point['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(file_name))


# def make_symlink(source, link_name):
#     if os.path.exists(link_name):
#         # print("Link name already exist! Removing '{}' and overwriting".format(link_name))
#         os.remove(link_name)
#     if os.path.exists(source):
#         os.symlink(source, link_name)
#         return
#     else:
#         print('Source path not exists')
#     # print('SymLink Wrong!')


def torch_accuracy(output, target, topk=(1,)) -> List[torch.Tensor]:
    # assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    # assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    # print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans


def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists)  # Very Important for dist = 0.
    return torch.sqrt(dists).float()


def fps(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e5
    inds = torch.randint(0, N, size=(B,), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :]  # (B, 3)
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz))
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids


class AvgMeter(object):
    name = 'No name'
    sum = 0
    mean = 0
    num = 0
    now = 0

    def __init__(self, name='No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0

    def update(self, mean_var, count=1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num
