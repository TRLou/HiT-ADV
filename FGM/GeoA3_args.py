from __future__ import absolute_import, division, print_function
import argparse
import math
import os
import sys
import re
import shutil
import time

import numpy as np
import scipy.io as sio
from pytorch3d.ops import knn_points, knn_gather
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils

matplotlib.use('Agg')
seaborn.set()
seaborn.set(rc={'figure.figsize': (11.7000, 8.27000)})
linewidth = 4.0
fontsize = 15.0 + 11.0
markersize = 10.0
fixpoint_markersize = 15.0
bins = 20
color_list = ['r', 'b', 'g']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR + '/../'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'Model'))


def get_args():
    parser = argparse.ArgumentParser(description='Point Cloud Attacking')
    # ------------Model-----------------------
    parser.add_argument('--id', type=int, default=0, help='')
    parser.add_argument('--arch', default='PointNet', type=str, metavar='ARCH', help='')
    # ------------Dataset-----------------------
    parser.add_argument('--data_dir_file', default='./Dataset/modelnet40_388instances1024_PointNet.mat', type=str, help='')
    parser.add_argument('--dense_data_dir_file', default=None, type=str, help='')
    parser.add_argument('-c', '--classes', default=40, type=int, metavar='N', help='num of classes (default: 40)')
    parser.add_argument('-b', '--batch_size', default=2, type=int, metavar='B', help='batch_size (default: 2)')
    parser.add_argument('--npoint', default=1024, type=int, help='')
    # ------------Attack-----------------------
    parser.add_argument('--attack', default='GeoA3', type=str, help='GeoA3 | GeoA3_mesh')
    parser.add_argument('--attack_label', default='Untarget', type=str, help='[All; ...; Untarget]')
    # 10
    parser.add_argument('--binary_max_steps', type=int, default=10, help='')
    parser.add_argument('--initial_const', type=float, default=10, help='')
    # 500
    parser.add_argument('--iter_max_steps', default=500, type=int, metavar='M', help='max steps')
    parser.add_argument('--optim', default='adam', type=str, help='adam| sgd')
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--eval_num', type=int, default=1, help='')
    ## cls loss
    parser.add_argument('--cls_loss_type', default='CE', type=str, help='Margin | CE')
    parser.add_argument('--confidence', type=float, default=0, help='confidence for margin based attack method')
    ## distance loss
    parser.add_argument('--dis_loss_type', default='CD', type=str, help='CD | L2 | None')
    parser.add_argument('--dis_loss_weight', type=float, default=1.0, help='')
    parser.add_argument('--is_cd_single_side', action='store_true', default=False, help='')
    ## hausdorff loss
    parser.add_argument('--hd_loss_weight', type=float, default=0.1, help='')
    ## normal loss
    parser.add_argument('--curv_loss_weight', type=float, default=1.0, help='')
    parser.add_argument('--curv_loss_knn', type=int, default=16, help='')
    ## uniform loss
    parser.add_argument('--uniform_loss_weight', type=float, default=0.0, help='')
    ## KNN smoothing loss
    parser.add_argument('--knn_smoothing_loss_weight', type=float, default=5.0, help='')
    parser.add_argument('--knn_smoothing_k', type=int, default=5, help='')
    parser.add_argument('--knn_threshold_coef', type=float, default=1.10, help='')
    ## Laplacian loss for mesh
    parser.add_argument('--laplacian_loss_weight', type=float, default=0, help='')
    parser.add_argument('--edge_loss_weight', type=float, default=0, help='')
    ## Mesh opt
    parser.add_argument('--is_partial_var', dest='is_partial_var', action='store_true', default=False, help='')
    parser.add_argument('--knn_range', type=int, default=3, help='')
    parser.add_argument('--is_subsample_opt', dest='is_subsample_opt', action='store_true', default=False, help='')
    parser.add_argument('--is_use_lr_scheduler', dest='is_use_lr_scheduler', action='store_true', default=False,
                        help='')
    ## perturbation clip setting
    parser.add_argument('--cc_linf', type=float, default=0.0, help='Coefficient for infinity norm')
    ## Proj offset
    parser.add_argument('--is_real_offset', action='store_true', default=False, help='')
    parser.add_argument('--is_pro_grad', action='store_true', default=False, help='')
    ## Jitter
    parser.add_argument('--is_pre_jitter_input', action='store_true', default=False, help='')
    parser.add_argument('--is_previous_jitter_input', action='store_true', default=False, help='')
    parser.add_argument('--calculate_project_jitter_noise_iter', default=50, type=int, help='')
    parser.add_argument('--jitter_k', type=int, default=16, help='')
    parser.add_argument('--jitter_sigma', type=float, default=0.01, help='')
    parser.add_argument('--jitter_clip', type=float, default=0.05, help='')
    ## PGD-like attack
    parser.add_argument('--step_alpha', type=float, default=5, help='')
    # ------------Recording settings-------
    parser.add_argument('--is_record_converged_steps', action='store_true', default=False, help='')
    parser.add_argument('--is_record_loss', action='store_true', default=False, help='')
    # ------------OS-----------------------
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--is_save_normal', action='store_true', default=False, help='')
    parser.add_argument('--is_debug', action='store_true', default=False, help='')
    parser.add_argument('--is_low_memory', action='store_true', default=False, help='')

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    return args


def norm_l2_loss(adv_pc, ori_pc):
    return ((adv_pc - ori_pc) ** 2).sum(1).sum(1)


def chamfer_loss(adv_pc, ori_pc):
    # Chamfer distance (two sides)
    # intra_dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    # dis_loss = intra_dis.min(2)[0].mean(1) + intra_dis.min(1)[0].mean(1)
    adv_KNN = knn_points(adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1)  # [dists:[b,n,1], idx:[b,n,1]]
    ori_KNN = knn_points(ori_pc.permute(0, 2, 1), adv_pc.permute(0, 2, 1), K=1)  # [dists:[b,n,1], idx:[b,n,1]]
    dis_loss = adv_KNN.dists.contiguous().squeeze(-1).mean(-1) + ori_KNN.dists.contiguous().squeeze(-1).mean(-1)  # [b]
    return dis_loss


def pseudo_chamfer_loss(adv_pc, ori_pc):
    # Chamfer pseudo distance (one side)
    # intra_dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1) #b*n*n
    # dis_loss = intra_dis.min(2)[0].mean(1)
    adv_KNN = knn_points(adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1)  # [dists:[b,n,1], idx:[b,n,1]]
    dis_loss = adv_KNN.dists.contiguous().squeeze(-1).mean(-1)  # [b]
    return dis_loss


def hausdorff_loss(adv_pc, ori_pc):
    # dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    # hd_loss = torch.max(torch.min(dis, dim=2)[0], dim=1)[0]
    adv_KNN = knn_points(adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1)  # [dists:[b,n,1], idx:[b,n,1]]
    hd_loss = adv_KNN.dists.contiguous().squeeze(-1).max(-1)[0]  # [b]
    return hd_loss


# equation (5) in paper
def _get_kappa_ori(pc, normal, k=2):
    b, _, n = pc.size()
    # inter_dis = ((pc.unsqueeze(3) - pc.unsqueeze(2))**2).sum(1)
    # inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    # nn_pts = torch.gather(pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    inter_KNN = knn_points(pc.permute(0, 2, 1), pc.permute(0, 2, 1), K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]
    nn_pts = knn_gather(pc.permute(0, 2, 1), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :,
             1:].contiguous()  # [b, 3, n ,k]
    vectors = nn_pts - pc.unsqueeze(3)
    vectors = _normalize(vectors)

    return torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2)  # [b, n]


# get adv kappa through the estimation of ori pc
def _get_kappa_adv(adv_pc, ori_pc, ori_normal, k=2):
    b, _, n = adv_pc.size()
    # compute knn between advPC and oriPC to get normal n_p
    # intra_dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    # intra_idx = torch.topk(intra_dis, 1, dim=2, largest=False, sorted=True)[1]
    # normal = torch.gather(ori_normal, 2, intra_idx.view(b,1,n).expand(b,3,n))
    intra_KNN = knn_points(adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1)  # [dists:[b,n,1], idx:[b,n,1]]
    normal = knn_gather(ori_normal.permute(0, 2, 1), intra_KNN.idx).permute(0, 3, 1, 2).squeeze(
        3).contiguous()  # [b, 3, n]

    # compute knn between advPC and itself to get \|q-p\|_2
    # inter_dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2))**2).sum(1)
    # inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    # nn_pts = torch.gather(adv_pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    inter_KNN = knn_points(adv_pc.permute(0, 2, 1), adv_pc.permute(0, 2, 1),
                           K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]
    nn_pts = knn_gather(adv_pc.permute(0, 2, 1), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :,
             1:].contiguous()  # [b, 3, n ,k]
    vectors = nn_pts - adv_pc.unsqueeze(3)
    vectors = _normalize(vectors)

    return torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2), normal  # [b, n], [b, 3, n]


def curvature_loss(adv_pc, ori_pc, adv_kappa, ori_kappa, k=2):
    b, _, n = adv_pc.size()

    # intra_dis = ((input_curr_iter.unsqueeze(3) - pc_ori.unsqueeze(2))**2).sum(1)
    # intra_idx = torch.topk(intra_dis, 1, dim=2, largest=False, sorted=True)[1]
    # knn_theta_normal = torch.gather(theta_normal, 1, intra_idx.view(b,n).expand(b,n))
    # curv_loss = ((curv_loss - knn_theta_normal)**2).mean(-1)

    intra_KNN = knn_points(adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1)  # [dists:[b,n,1], idx:[b,n,1]]
    onenn_ori_kappa = torch.gather(ori_kappa, 1, intra_KNN.idx.squeeze(-1)).contiguous()  # [b, n]

    curv_loss = ((adv_kappa - onenn_ori_kappa) ** 2).mean(-1)

    return curv_loss


def displacement_loss(adv_pc, ori_pc, k=16):
    b, _, n = adv_pc.size()
    with torch.no_grad():
        inter_dis = ((ori_pc.unsqueeze(3) - ori_pc.unsqueeze(2)) ** 2).sum(1)
        inter_idx = torch.topk(inter_dis, k + 1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()

    theta_distance = ((adv_pc - ori_pc) ** 2).sum(1)
    nn_theta_distances = torch.gather(theta_distance, 1, inter_idx.view(b, n * k)).view(b, n, k)
    return ((nn_theta_distances - theta_distance.unsqueeze(2)) ** 2).mean(2)


def corresponding_normal_loss(adv_pc, normal, k=2):
    b, _, n = adv_pc.size()

    inter_dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2)) ** 2).sum(1)
    inter_idx = torch.topk(inter_dis, k + 1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    nn_pts = torch.gather(adv_pc, 2, inter_idx.view(b, 1, n * k).expand(b, 3, n * k)).view(b, 3, n, k)
    vectors = nn_pts - adv_pc.unsqueeze(3)
    vectors = _normalize(vectors)
    return torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2)


def repulsion_loss(pc, k=4, h=0.03):
    dis = ((pc.unsqueeze(3) - pc.unsqueeze(2)) ** 2).sum(1)
    dis = torch.topk(dis, k + 1, dim=2, largest=False, sorted=True)[0][:, :, 1:].contiguous()

    return -(dis * torch.exp(-(dis ** 2) / (h ** 2))).mean(2)


def distance_kmean_loss(pc, k):
    b, _, n = pc.size()
    dis = ((pc.unsqueeze(3) - pc.unsqueeze(2) + 1e-12) ** 2).sum(1).sqrt()
    dis, idx = torch.topk(dis, k + 1, dim=2, largest=False, sorted=True)
    dis_mean = dis[:, :, 1:].contiguous().mean(-1)  # b*n
    idx = idx[:, :, 1:].contiguous()
    dis_mean_k = torch.gather(dis_mean, 1, idx.view(b, n * k)).view(b, n, k)

    return torch.abs(dis_mean.unsqueeze(2) - dis_mean_k).mean(-1)


def kNN_smoothing_loss(adv_pc, k, threshold_coef=1.05):
    b, _, n = adv_pc.size()
    # dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2))**2).sum(1) #[b,n,n]
    # dis, idx = torch.topk(dis, k+1, dim=2, largest=False, sorted=True)#[b,n,k+1]
    inter_KNN = knn_points(adv_pc.permute(0, 2, 1), adv_pc.permute(0, 2, 1),
                           K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]

    knn_dis = inter_KNN.dists[:, :, 1:].contiguous().mean(-1)  # [b,n]
    knn_dis_mean = knn_dis.mean(-1)  # [b]
    knn_dis_std = knn_dis.std(-1)  # [b]
    threshold = knn_dis_mean + threshold_coef * knn_dis_std  # [b]

    condition = torch.gt(knn_dis, threshold.unsqueeze(1)).float()  # [b,n]
    dis_mean = knn_dis * condition  # [b,n]

    return dis_mean.mean(1)  # [b]


def uniform_loss(adv_pc, percentages=[0.004, 0.006, 0.008, 0.010, 0.012], radius=1.0, k=2):
    if adv_pc.size(1) == 3:
        adv_pc = adv_pc.permute(0, 2, 1).contiguous()
    b, n, _ = adv_pc.size()
    npoint = int(n * 0.05)
    for p in percentages:
        p = p * 4
        nsample = int(n * p)
        r = math.sqrt(p * radius)
        disk_area = math.pi * (radius ** 2) * p / nsample
        expect_len = torch.sqrt(torch.Tensor([disk_area])).cuda()

        adv_pc_flipped = adv_pc.transpose(1, 2).contiguous()
        new_xyz = pointnet2_utils.gather_operation(adv_pc_flipped,
                                                   pointnet2_utils.furthest_point_sample(adv_pc, npoint)).transpose(1,
                                                                                                                    2).contiguous()  # (batch_size, npoint, 3)

        idx = pointnet2_utils.ball_query(r, nsample, adv_pc, new_xyz)  # (batch_size, npoint, nsample)

        grouped_pcd = pointnet2_utils.grouping_operation(adv_pc_flipped, idx).permute(0, 2, 3,
                                                                                      1).contiguous()  # (batch_size, npoint, nsample, 3)
        grouped_pcd = torch.cat(torch.unbind(grouped_pcd, axis=1), axis=0)

        grouped_pcd = grouped_pcd.permute(0, 2, 1).contiguous()
        # dis = torch.sqrt(((grouped_pcd.unsqueeze(3) - grouped_pcd.unsqueeze(2))**2).sum(1)+1e-12) # (batch_size*npoint, nsample, nsample)
        # dists, _ = torch.topk(dis, k+1, dim=2, largest=False, sorted=True) # (batch_size*npoint, nsample, k+1)
        inter_KNN = knn_points(grouped_pcd.permute(0, 2, 1), grouped_pcd.permute(0, 2, 1),
                               K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]

        # inter_KNN : [B*npoint, nsample, k+1]
        uniform_dis = inter_KNN.dists[:, :, 1:].contiguous()
        uniform_dis = torch.sqrt(torch.abs(uniform_dis) + 1e-12)
        uniform_dis = uniform_dis.mean(axis=[-1])
        uniform_dis = (uniform_dis - expect_len) ** 2 / (expect_len + 1e-12)
        uniform_dis = torch.reshape(uniform_dis, [-1])

        mean = uniform_dis.mean()
        mean = mean * math.pow(p * 100, 2)

        # nothing 4
        try:
            loss = loss + mean
        except:
            loss = mean
    return loss / len(percentages)


def _normalize(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

def jitter_input(data, sigma=0.01, clip=0.05):
    assert data.size(1) == 3
    assert(clip > 0)
    B, _, N = data.size()
    jittered_data = torch.clamp(sigma * torch.randn(B, 3, N), -1*clip, clip).cuda()
    return jittered_data

def estimate_normal(pc, k):
    with torch.no_grad():
        # pc : [b, 3, n]
        b,_,n=pc.size()
        # get knn point set matrix
        inter_KNN = knn_points(pc.permute(0,2,1), pc.permute(0,2,1), K=k+1) #[dists:[b,n,k+1], idx:[b,n,k+1]]
        nn_pts = knn_gather(pc.permute(0,2,1), inter_KNN.idx).permute(0,3,1,2)[:,:,:,1:].contiguous() # [b, 3, n ,k]

        # get covariance matrix and smallest eig-vector of individual point
        normal_vector = []
        for i in range(b):
            if int(torch.__version__.split('.')[1])>=4:
                curr_point_set = nn_pts[i].detach().permute(1,0,2) #curr_point_set:[n, 3, k]
                curr_point_set_mean = torch.mean(curr_point_set, dim=2, keepdim=True) #curr_point_set_mean:[n, 3, 1]
                curr_point_set = curr_point_set - curr_point_set_mean #curr_point_set:[n, 3, k]
                curr_point_set_t = curr_point_set.permute(0,2,1) #curr_point_set_t:[n, k, 3]
                fact = 1.0 / (k-1)
                cov_mat = fact * torch.bmm(curr_point_set, curr_point_set_t) #curr_point_set_t:[n, 3, 3]
                eigenvalue, eigenvector=torch.symeig(cov_mat, eigenvectors=True)    # eigenvalue:[n, 3], eigenvector:[n, 3, 3]
                persample_normal_vector = torch.gather(eigenvector, 2, torch.argmin(eigenvalue, dim=1).unsqueeze(1).unsqueeze(2).expand(n, 3, 1)).squeeze() #persample_normal_vector:[n, 3]

                #recorrect the direction via neighbour direction
                nbr_sum = curr_point_set.sum(dim=2)  #curr_point_set:[n, 3]
                sign = -torch.sign(torch.bmm(persample_normal_vector.view(n, 1, 3), nbr_sum.view(n, 3, 1))).squeeze(2)
                persample_normal_vector = sign * persample_normal_vector

                normal_vector.append(persample_normal_vector.permute(1,0))

            else:
                persample_normal_vector = []
                for j in range(n):
                    curr_point_set = nn_pts[i,:,j,:].cpu()
                    curr_point_set_np = curr_point_set.detach().numpy()#curr_point_set_np:[3,k]
                    cov_mat_np = np.cov(curr_point_set_np)   #cov_mat:[3,3]
                    eigenvalue_np, eigenvector_np=np.linalg.eig(cov_mat_np)   #eigenvalue:[3], eigenvector:[3,3]; note that v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
                    curr_normal_vector_np = torch.from_numpy(eigenvector_np[:,np.argmin(eigenvalue_np)]) #curr_normal_vector:[3]
                    persample_normal_vector.append(curr_normal_vector_np)
                persample_normal_vector = torch.stack(persample_normal_vector, 1)

                #recorrect the direction via neighbour direction
                nbr_sum = curr_point_set.sum(dim=1)  #curr_point_set:[3]
                sign = -torch.sign(torch.bmm(persample_normal_vector.view(1, 3), nbr_sum.view(3, 1))).squeeze(1)
                persample_normal_vector = sign * persample_normal_vector

                normal_vector.append(persample_normal_vector.permute(1,0))

                normal_vector.append(persample_normal_vector)

        normal_vector = torch.stack(normal_vector, 0) #normal_vector:[b, 3, n]
    return normal_vector.float()

def estimate_normal_via_ori_normal(pc_adv, pc_ori, normal_ori, k):
    # pc_adv, pc_ori, normal_ori : [b,3,n]
    b,_,n=pc_adv.size()
    intra_KNN = knn_points(pc_adv.permute(0,2,1), pc_ori.permute(0,2,1), K=k) #[dists:[b,n,k], idx:[b,n,k]]
    inter_value = intra_KNN.dists[:, :, 0].contiguous()
    inter_idx = intra_KNN.idx.permute(0,2,1).contiguous()
    normal_pts = knn_gather(normal_ori.permute(0,2,1), intra_KNN.idx).permute(0,3,1,2).contiguous() # [b, 3, n ,k]

    normal_pts_avg = normal_pts.mean(dim=-1)
    normal_pts_avg = normal_pts_avg/(normal_pts_avg.norm(dim=1)+1e-12)

    # If the points are not modified (distance = 0), use the normal directly from the original
    # one. Otherwise, use the mean of the normals of the k-nearest points.
    normal_ori_select = normal_pts[:,:,:,0]
    condition = (inter_value<1e-6).unsqueeze(1).expand_as(normal_ori_select)
    normals_estimated = torch.where(condition, normal_ori_select, normal_pts_avg)

    return normals_estimated

def get_perpendicular_jitter(vector, sigma=0.01, clip=0.05):
    b,_,n=vector.size()
    aux_vector1 = sigma * torch.randn(b,3,n).cuda()
    aux_vector2 = sigma * torch.randn(b,3,n).cuda()
    return torch.clamp(torch.cross(vector, aux_vector1), -1*clip, clip) + torch.clamp(torch.cross(vector, aux_vector2), -1*clip, clip)

def estimate_perpendicular(pc, k, sigma=0.01, clip=0.05):
    with torch.no_grad():
        # pc : [b, 3, n]
        b,_,n=pc.size()
        inter_KNN = knn_points(pc.permute(0,2,1), pc.permute(0,2,1), K=k+1) #[dists:[b,n,k+1], idx:[b,n,k+1]]
        nn_pts = knn_gather(pc.permute(0,2,1), inter_KNN.idx).permute(0,3,1,2)[:,:,:,1:].contiguous() # [b, 3, n ,k]

        # get covariance matrix and smallest eig-vector of individual point
        perpendi_vector_1 = []
        perpendi_vector_2 = []
        for i in range(b):
            curr_point_set = nn_pts[i].detach().permute(1,0,2) #curr_point_set:[n, 3, k]
            curr_point_set_mean = torch.mean(curr_point_set, dim=2, keepdim=True) #curr_point_set_mean:[n, 3, 1]
            curr_point_set = curr_point_set - curr_point_set_mean #curr_point_set:[n, 3, k]
            curr_point_set_t = curr_point_set.permute(0,2,1) #curr_point_set_t:[n, k, 3]
            fact = 1.0 / (k-1)
            cov_mat = fact * torch.bmm(curr_point_set, curr_point_set_t) #curr_point_set_t:[n, 3, 3]
            eigenvalue, eigenvector=torch.symeig(cov_mat, eigenvectors=True)    # eigenvalue:[n, 3], eigenvector:[n, 3, 3]

            larger_dim_idx = torch.topk(eigenvalue, 2, dim=1, largest=True, sorted=False, out=None)[1] # eigenvalue:[n, 2]

            persample_perpendi_vector_1 = torch.gather(eigenvector, 2, larger_dim_idx[:,0].unsqueeze(1).unsqueeze(2).expand(n, 3, 1)).squeeze() #persample_perpendi_vector_1:[n, 3]
            persample_perpendi_vector_2 = torch.gather(eigenvector, 2, larger_dim_idx[:,1].unsqueeze(1).unsqueeze(2).expand(n, 3, 1)).squeeze() #persample_perpendi_vector_2:[n, 3]

            perpendi_vector_1.append(persample_perpendi_vector_1.permute(1,0))
            perpendi_vector_2.append(persample_perpendi_vector_2.permute(1,0))

        perpendi_vector_1 = torch.stack(perpendi_vector_1, 0) #perpendi_vector_1:[b, 3, n]
        perpendi_vector_2 = torch.stack(perpendi_vector_2, 0) #perpendi_vector_1:[b, 3, n]

        aux_vector1 = sigma * torch.randn(b,n).unsqueeze(1).cuda() #aux_vector1:[b, 1, n]
        aux_vector2 = sigma * torch.randn(b,n).unsqueeze(1).cuda() #aux_vector2:[b, 1, n]

    return torch.clamp(perpendi_vector_1*aux_vector1, -1*clip, clip) + torch.clamp(perpendi_vector_2*aux_vector2, -1*clip, clip)

def _compare(output, target, gt, targeted):
    if targeted:
        return output == target
    else:
        return output != gt

def farthest_points_normalized_single_numpy(obj_points, num_points):
    first = np.random.randint(len(obj_points))
    selected = [first]
    dists = np.full(shape = len(obj_points), fill_value = np.inf)

    for _ in range(num_points - 1):
        dists = np.minimum(dists, np.linalg.norm(obj_points - obj_points[selected[-1]][np.newaxis, :], axis = 1))
        selected.append(np.argmax(dists))
    res_points = np.array(obj_points[selected])

    # normalize the points
    avg = np.average(res_points, axis = 0)
    res_points = res_points - avg[np.newaxis, :]
    dists = np.max(np.linalg.norm(res_points, axis = 1), axis = 0)
    res_points = res_points / dists

    return res_points

def farthest_points_sample(obj_points, num_points):
    assert obj_points.size(1) == 3
    b,_,n = obj_points.size()

    selected = torch.randint(obj_points.size(2), [obj_points.size(0),1]).cuda()
    dists = torch.full([obj_points.size(0), obj_points.size(2)], fill_value = np.inf).cuda()

    for _ in range(num_points - 1):
        dists = torch.min(dists, torch.norm(obj_points - torch.gather(obj_points, 2, selected[:,-1].unsqueeze(1).unsqueeze(2).expand(b,3,1)), dim = 1))
        selected = torch.cat([selected, torch.argmax(dists, dim=1, keepdim=True)], dim = 1)
    res_points = torch.gather(obj_points, 2, selected.unsqueeze(1).expand(b, 3, num_points))

    return res_points

def farthest_points_normal_sample(obj_points, obj_normal, num_points):
    assert obj_points.size(1) == 3
    assert obj_points.size(2) == obj_normal.size(2)
    b,_,n = obj_points.size()

    selected = torch.randint(obj_points.size(2), [obj_points.size(0),1]).cuda()
    dists = torch.full([obj_points.size(0), obj_points.size(2)], fill_value = np.inf).cuda()

    for _ in range(num_points - 1):
        dists = torch.min(dists, torch.norm(obj_points - torch.gather(obj_points, 2, selected[:,-1].unsqueeze(1).unsqueeze(2).expand(b,3,1)), dim = 1))
        selected = torch.cat([selected, torch.argmax(dists, dim=1, keepdim=True)], dim = 1)
    res_points = torch.gather(obj_points, 2, selected.unsqueeze(1).expand(b, 3, num_points))
    res_normal = torch.gather(obj_normal, 2, selected.unsqueeze(1).expand(b, 3, num_points))

    return res_points, res_normal


def pad_larger_tensor_with_index(small_verts, small_in_larger_idx_list, larger_tensor_shape):
    full_deform_verts = torch.zeros(larger_tensor_shape,3).cuda()
    full_deform_verts[small_in_larger_idx_list] = small_verts
    return full_deform_verts

def pad_larger_tensor_with_index_batch(small_verts, small_in_larger_idx_list, larger_tensor_shape):
    b, _, n = small_verts.size()
    full_deform_verts = torch.zeros(b, 3, larger_tensor_shape).cuda()
    for i in range(b):
        full_deform_verts[i, :, small_in_larger_idx_list[i][0][1:]] = small_verts[i]
    return full_deform_verts

def read_lines_from_xyz(path, num_points):
    with open(path) as file:
        vertices = []
        if num_points == -1:
            num_points = len(open(path,'r').readlines())
        for i in range(num_points):
            line = file.readline()
            vertices.append([float(x) for x in line.split()[0:3]])

    return vertices

def write_obj(file, vertices, faces):
    """
    Writes the given vertices and faces to OBJ.

    :param vertices: vertices as tuples of (x, y, z) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        for vertex in vertices:
            assert len(vertex) == 3, 'invalid vertex with %d dimensions found (%s)' % (len(vertex), file)
            fp.write('v' + ' ' + str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')

        for face in faces:
            assert len(face) == 3, 'only triangular faces supported (%s)' % file
            fp.write('f ')

            for i in range(len(face)):
                assert face[i] >= 0 and face[i] < num_vertices, 'invalid vertex index %d (of %d vertices) (%s)' % (face[i], num_vertices, file)

                # face indices are 1-based
                fp.write(str(face[i] + 1))
                if i < len(face) - 1:
                    fp.write(' ')

            fp.write('\n')

        # add empty line to be sure
        fp.write('\n')

def read_obj(file):
    """
    Reads vertices and faces from an obj file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines if line.strip()]

        vertices = []
        faces = []
        for line in lines:
            parts = line.split(' ')
            parts = [part.strip() for part in parts if part]

            if parts[0] == 'v':
                assert len(parts) == 4 or len(parts) == 7,\
                    'vertex should be of the form v x y z nx ny nz, but found %d parts instead (%s)' % (len(parts), file)
                assert parts[1] != '', 'vertex x coordinate is empty (%s)' % file
                assert parts[2] != '', 'vertex y coordinate is empty (%s)' % file
                assert parts[3] != '', 'vertex z coordinate is empty (%s)' % file

                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                assert len(parts) == 4, \
                    'face should be of the form f v1/vt1/vn1 v2/vt2/vn2 v2/vt2/vn2, but found %d parts (%s) instead (%s)' % (len(parts), line, file)

                components = parts[1].split('/')
                assert len(components) >= 1 and len(components) <= 3, \
                   'face component should have the forms v, v/vt or v/vt/vn, but found %d components instead (%s)' % (len(components), file)
                assert components[0].strip() != '', \
                    'face component is empty (%s)' % file
                v1 = int(components[0])

                components = parts[2].split('/')
                assert len(components) >= 1 and len(components) <= 3, \
                    'face component should have the forms v, v/vt or v/vt/vn, but found %d components instead (%s)' % (len(components), file)
                assert components[0].strip() != '', \
                    'face component is empty (%s)' % file
                v2 = int(components[0])

                components = parts[3].split('/')
                assert len(components) >= 1 and len(components) <= 3, \
                    'face component should have the forms v, v/vt or v/vt/vn, but found %d components instead (%s)' % (len(components), file)
                assert components[0].strip() != '', \
                    'face component is empty (%s)' % file
                v3 = int(components[0])

                #assert v1 != v2 and v2 != v3 and v3 != v2, 'degenerate face detected: %d %d %d (%s)' % (v1, v2, v3, file)
                if v1 == v2 or v2 == v3 or v1 == v3:
                    print('[Info] skipping degenerate face in %s' % file)
                else:
                    faces.append([v1 - 1, v2 - 1, v3 - 1]) # indices are 1-based!
            else:
                #assert False, 'expected either vertex or face but got line: %s (%s)' % (line, file)
                pass

        return vertices, faces

    assert False, 'could not open %s' % file

def write_off(file, vertices, faces):
    """
    Writes the given vertices and faces to OFF.

    :param vertices: vertices as tuples of (x, y, z) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        fp.write('OFF\n')
        fp.write(str(num_vertices) + ' ' + str(num_faces) + ' 0\n')

        for vertex in vertices:
            assert len(vertex) == 3, 'invalid vertex with %d dimensions found (%s)' % (len(vertex), file)
            fp.write(str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')

        for face in faces:
            assert face[0] == 3, 'only triangular faces supported (%s)' % file
            assert len(face) == 4, 'faces need to have 3 vertices, but found %d (%s)' % (len(face), file)

            for i in range(len(face)):
                assert face[i] >= 0 and face[i] < num_vertices, 'invalid vertex index %d (of %d vertices) (%s)' % (face[i], num_vertices, file)

                fp.write(str(face[i]))
                if i < len(face) - 1:
                    fp.write(' ')

            fp.write('\n')

        # add empty line to be sure
        fp.write('\n')

def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        # Fix for ModelNet bug were 'OFF' and the number of vertices and faces are
        # all in the first line.
        if len(lines[0]) > 3 and lines[0][:4] != 'COFF':
            assert lines[0][:3] == 'OFF' or lines[0][:3] == 'off', 'invalid OFF file %s' % file

            parts = lines[0][3:].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 1
        # This is the regular case!
        else:
            assert lines[0] == 'OFF' or lines[0] == 'off' or lines[0][:4] == 'COFF', 'invalid OFF file %s' % file

            parts = lines[1].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 2

        vertices = []
        for i in range(num_vertices):
            vertex = lines[start_index + i].split(' ')
            vertex = [float(point.strip()) for point in vertex if point != '']
            if len(vertex) == 3:
                vertices.append(vertex)
            else:
                vertices.append(vertex[0:3])

        faces = []
        for i in range(num_faces):
            face = lines[start_index + num_vertices + i].split(' ')
            face = [index.strip() for index in face if index != '']

            # check to be sure
            for index in face:
                assert index != '', 'found empty vertex index: %s (%s)' % (lines[start_index + num_vertices + i], file)

            face = [int(index) for index in face]

            assert face[0] == len(face) - 1, 'face should have %d vertices but as %d (%s)' % (face[0], len(face) - 1, file)
            assert face[0] == 3, 'only triangular meshes supported (%s)' % file
            for index in face:
                assert index >= 0 and index < num_vertices, 'vertex %d (of %d vertices) does not exist (%s)' % (index, num_vertices, file)

            assert len(face) > 1

            faces.append(face)

        return vertices, faces

    assert False, 'could not open %s' % file

def pc_normalize_torch(point):
    #point:[n,3]
    assert len(point.size()) == 2
    assert point.size(1) == 3
    # normalize the point and face
    with torch.no_grad():
        avg = torch.mean(point.t(), dim = 1)
        normed_point = point - avg[np.newaxis, :]
        scale = torch.max(torch.norm(normed_point, dim = 1), dim = 0)[0]
        normed_point = normed_point / scale[np.newaxis, np.newaxis]
    return normed_point

# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 30.
TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    step_time = Average_meter()
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: {0}'.format(format_time(step_time)))
    L.append(' | Tot: {0}'.format(format_time(tot_time)))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class Average_meter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Training_aux(object):
    def __init__(self, fsave):
        self.fsave = fsave
        if not os.path.exists(self.fsave):
            os.makedirs(self.fsave)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        '''
        usage:
        Training_aux.save_checkpoint(
            state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
            }, is_best = is_best)
        '''
        filename = os.path.join(self.fsave, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, '%s/'%(self.fsave) + 'modelBest.pth.tar')
        return

    def load_checkpoint(self, model, optimizer, is_best):
        """Loads checkpoint from disk"""
        '''
        usage:
        start_epoch, best_prec1 = Training_aux.load_checkpoint(model = model, is_best = is_best)
        '''
        if is_best:
            filename = os.path.join(self.fsave, 'modelBest.pth.tar')
            print("=> loading best model '{}'".format(filename))
        else:
            filename = os.path.join(self.fsave, 'checkpoint.pth.tar')
            print("=> loading checkpoint '{}'".format(filename))

        if os.path.isfile(filename):
            checkpoint = torch.load(filename)

            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                model_dict = model.state_dict()
                pretrained_dict = checkpoint['state_dict']

                new_dict = {k: v for k, v in pretrained_dict.items()[:-2] if k in model_dict.keys()}
                model_dict.update(new_dict)
                print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
                model.load_state_dict(model_dict)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print("=> No optimizer loaded in '{}'".format(filename))
            print("==> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return start_epoch, best_prec1

    def write_err_to_file(self, info):
        """write error to txt"""
        fpath = os.path.join(self.fsave, 'state.txt')
        if os.path.isfile(fpath):
            file = open(fpath, 'a')
        else:
            file = open(fpath, "w")

        file.write(info)

        file.close()
        return

class Count_converge_iter(object):
    def __init__(self, fsave):
        self.fsave = fsave
        if not os.path.exists(self.fsave):
            os.makedirs(self.fsave)
        self.attack_step_list = []

    def record_converge_iter(self, attack_step_list):
        if -1 in attack_step_list:
            attack_step_list.remove(-1)
        self.attack_step_list += attack_step_list

    def save_converge_iter(self):
        fpath = os.path.join(self.fsave, 'converge_iter.mat')
        sio.savemat(fpath, {"attack_step_list": self.attack_step_list})

    def plot_converge_iter_hist(self):
        fpath = os.path.join(self.fsave, 'converge_iter.png')
        used_bins=np.histogram(np.hstack((self.attack_step_list)), bins=bins)[1]
        fig = plt.figure()
        ax = seaborn.distplot(self.attack_step_list, bins=used_bins)
        ax.set_xlabel('Converged iteration', fontsize=fontsize)
        ax.set_ylabel('Number of Samples', fontsize=fontsize)
        plt.savefig(fpath)


class Count_loss_iter(object):
    def __init__(self, fsave):
        self.fsave = fsave
        if not os.path.exists(self.fsave):
            os.makedirs(self.fsave)

    def record_loss_iter(self, loss_list):
        # loss_list:[steps, b]
        try:
            self.loss_numpy = np.concatenate((self.loss_numpy, np.array(loss_list)), axis=1)
        except:
            self.loss_numpy = np.array(loss_list)

    def save_loss_iter(self):
        fpath = os.path.join(self.fsave, 'loss_iter.mat')
        sio.savemat(fpath, {"loss": self.loss_numpy})

    def plot_loss_iter_hist(self):
        fpath = os.path.join(self.fsave, 'loss_iter.png')
        num_iter, num_sample = self.loss_numpy.shape

        start_iter = 1
        x = np.arange(start_iter, num_iter+start_iter)
        x.astype(int)
        loss_mean = self.loss_numpy.mean(1)
        loss_std = self.loss_numpy.std(1)

        f, ax = plt.subplots(1,1)
        ax.plot(x, loss_mean, color=color_list[0])
        r1 = list(map(lambda x: x[0]-x[1], zip(loss_mean, loss_std)))
        r2 = list(map(lambda x: x[0]+x[1], zip(loss_mean, loss_std)))
        ax.fill_between(x, r1, r2, color=color_list[0], alpha=0.2)
        ax.set_xlabel('Number of iteration', fontsize=fontsize)
        ax.set_ylabel('Magnitude of loss', fontsize=fontsize)
        plt.savefig(fpath)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
