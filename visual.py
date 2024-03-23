import os
import argparse
import torch
from model.pointnet_cls import get_model
from model.pointnet2_cls_ssg import get_model as get_model_pnp
from model.dgcnn_cls import DGCNN_cls
from model.pct_cls import Pct
from FGM import CWPert_args

from util.clip_utils import ClipPointsL2, ClipPointsLinf, ProjectInnerClipLinf
from util.adv_utils import LogitsAdvLoss, CrossEntropyAdvLoss, UntargetedLogitsAdvLoss
from util.other_utils import create_logger, reconstruct_from_pc
import numpy as np
from mayavi import mlab
from util.dist_utils import L2Dist, LaplacianDist, ChamferkNNDist

from datetime import datetime
from CW import CWAOF
from ShapeAttack.HiT_ADV import HiT_ADV


def evalit(net, val_attack=None, data=None, target=None):
    # args.step_size = args.budget / 10.0
    net.eval()

    with torch.no_grad():
        points, target = data.float().cuda(non_blocking=True), \
            target.long().cuda(non_blocking=True)
        pc = points.clone().detach()
        pc = torch.unsqueeze(pc, 0)

    adv_pc, _ = val_attack.attack(pc, target)

    if isinstance(adv_pc, tuple):
        adv_pc = adv_pc[0]
    if not torch.is_tensor(adv_pc):
        adv_pc = torch.Tensor(adv_pc).cuda()

    with torch.no_grad():
        logits = model(adv_pc.transpose(1, 2).contiguous())
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = torch.argmax(logits, dim=-1)
        print('pred is', pred)

    adv_pc = torch.squeeze(adv_pc, 0)
    pc = torch.squeeze(pc[:, :, :3], 0)
    return pc, adv_pc


def visualize_point_cloud(point_cloud, sampling_rate=1.0, point_size=0.05):
    # random sampling
    num_points = point_cloud.shape[0]
    num_sampled_points = int(num_points * sampling_rate)
    sampled_indices = np.random.choice(num_points, num_sampled_points, replace=False)
    sampled_points = point_cloud[sampled_indices, :]

    # plot point cloud
    x, y, z = sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    np.savetxt('{}_pc.asc'.format(formatted_time), sampled_points[:, :3])
    mlab.figure(bgcolor=(1, 1, 1), size=(400, 400))
    mlab.points3d(x, y, z, scale_factor=point_size * 1.3, color=(0.4549, 0.8666, 1.0))

    mlab.view(azimuth=0, elevation=0, distance=5, focalpoint='auto')
    mlab.orientation_axes()
    mlab.draw()
    mlab.show()


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--num_class', type=int, default=40, help='class numbers')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--adv_func', type=str, default='cross_entropy',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--budget', type=float, default=0.05,
                        help='FGM attack budget')
    parser.add_argument('--attack_type', type=str, default='SA_ours', metavar='N',
                        help='Attack method to use')
    parser.add_argument('--num_iter', type=int, default=100,
                        help='IFGM iterate step')
    parser.add_argument('--mu', type=float, default=1.,
                        help='momentum factor for MIFGM attack')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 16, 40], help='training on ModelNet10/40')
    parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet', 'dgcnn', 'pointnet++', 'pct'],
                        help='model for training')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--dropout', type=float, default=0.5, help='initial dropout rate')
    parser.add_argument('--k', type=int, default=1, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')

    return parser.parse_args()


def knn(x, k):
    """
    x:(B, 3, N)
    """
    with torch.no_grad():
        inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)

        vec = x.transpose(2, 1).unsqueeze(2) - x.transpose(2, 1).unsqueeze(1)
        dist = -torch.sum(vec ** 2, dim=-1)
        # print("distance check:", torch.allclose(pairwise_distance, dist))

        # print(f"dist shape:{pairwise_distance.shape}")
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_Laplace_from_pc(ori_pc):
    """
    ori_pc:(B, 3, N)
    """
    # print("shape of ori pc:",ori_pc.shape)
    pc = ori_pc.detach().clone()
    with torch.no_grad():
        # pc = pc.to('cpu').to(torch.double)
        idx = knn(pc, 30)
        pc = pc.transpose(2, 1).contiguous()  # (B, N, 3)
        point_mat = pc.unsqueeze(2) - pc.unsqueeze(1)  # (B, N, N, 3)
        A = torch.exp(-torch.sum(point_mat.square(), dim=3))  # (B, N, N)
        mask = torch.zeros_like(A)
        mask.scatter_(2, idx, 1)
        mask = mask + mask.transpose(2, 1)
        mask[mask > 1] = 1

        A = A * mask
        D = torch.diag_embed(torch.sum(A, dim=2))
        L = D - A
        e, v = torch.symeig(L, eigenvectors=True)
    return e.to(ori_pc), v.to(ori_pc)


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    state_dict = torch.load('Checkpoint/PN_NT.checkpoint')

    args.step_size = args.budget / float(args.num_iter)

    data = np.loadtxt('../../PC_Dataset/modelnet40_normal_resampled/car/car_0005.txt', delimiter=',',
                      dtype=np.float32)

    target = torch.Tensor([7])
    ori_data = data[:, :3]
    if args.model == 'pointnet':
        print('using pointnet')
        model = get_model(args.num_class, normal_channel=args.use_normals).cuda()
    elif args.model == 'dgcnn':
        print('using dgcnn')
        model = DGCNN_cls(args, output_channels=args.num_class).cuda()
    elif args.model == 'pointnet++':
        print('using pointnet++')
        model = get_model_pnp(args.num_class, normal_channel=args.use_normals).cuda()
    elif args.model == 'pct':
        print('using pct')
        model = Pct(args, output_channels=args.num_class).cuda()
    else:
        raise Exception("Not implemented")
    model.load_state_dict(state_dict['model_state_dict'])
    # model.load_state_dict(state_dict['last'])

    model.eval()
    logger = create_logger('./log', 'eval_last', 'info')

    adv_func = CrossEntropyAdvLoss()
    CWPerturb_args = CWPert_args.get_args()
    CW_adv_func = UntargetedLogitsAdvLoss(kappa=CWPerturb_args.kappa)

    HiT_attacker = HiT_ADV(model, adv_func=CW_adv_func, attack_lr=0.01,
                               init_weight=10., max_weight=80., binary_step=CWPerturb_args.binary_step,
                               num_iter=CWPerturb_args.num_iter, clip_func=None,
                               cd_weight=0.01, ker_weight=1,
                               hide_weight=1, curv_loss_knn=16, central_num=100, total_central_num=200,
                               max_sigm=1.5, min_sigm=0.2, budget=0.55)

    data = torch.Tensor(data)
    # visualize_point_cloud(data.detach().cpu().numpy(), sampling_rate=1, point_size=0.01)

    clean_data, adv_data = evalit(model, val_attack=HiT_attacker, data=data[:, :6], target=target)

    adv_data = adv_data.reshape(-1, 3).detach().cpu().numpy()
    clean_data = clean_data.reshape(-1, 3).detach().cpu().numpy()

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    visualize_point_cloud(adv_data, sampling_rate=1, point_size=0.01)
