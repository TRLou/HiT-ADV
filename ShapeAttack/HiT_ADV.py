"""Implementation of HiT-ADV.
"""

import time
import torch
import torch.optim as optim
import numpy as np
from util.dist_utils import L2Dist, LaplacianDist, ChamferDist, HausdorffDist
from pytorch3d.ops import knn_points, knn_gather
from mayavi import mlab
import torch.nn.functional as F
import open3d as o3d


class HiT_ADV:
    """Class for HiT_ADV Attack
    """
    def __init__(self, model, adv_func, attack_lr=1e-2,
                 init_weight=10., max_weight=80., binary_step=10, num_iter=500, clip_func=None,
                 cd_weight=0, curv_weight=0, ker_weight=0, hide_weight=0, curv_loss_knn=32, central_num=32,
                 total_central_num=128, max_sigm=0.7,
                 min_sigm=0.1, budget=0.1, alpha=1):
        self.model = model.cuda()
        self.model.eval()
        self.adv_func = adv_func
        self.attack_lr = attack_lr
        self.init_weight = init_weight
        self.max_weight = max_weight
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.clip_func = clip_func
        self.cd_weight = cd_weight
        self.curv_weight = curv_weight
        self.hide_weight = hide_weight
        self.ker_weight = ker_weight
        self.curv_loss_knn = curv_loss_knn
        self.central_num = central_num
        self.max_sigm = max_sigm
        self.min_sigm = min_sigm
        self.budget = budget
        self.alpha = alpha
        self.total_central_num = total_central_num

    def attack(self, data, target):
        """Attack on given data to target.

        Args:
            data (torch.FloatTensor): victim data, [B, num_points, 3]
            target (torch.LongTensor): target output, [B]
        """
        B, K = data.shape[:2]  # [B, N, 6]
        ori_data = data[:, :, :3].float().cuda().clone().detach()
        ori_normal = data[:, :, 3:].float().cuda().clone().detach()  # [B, N, 3]
        ori_data = ori_data.transpose(1, 2).contiguous()  # [B, 3, N]
        ori_normal = ori_normal.transpose(1, 2).contiguous()

        ori_data.requires_grad = False
        target = target.long().cuda().detach()
        label_val = target.detach().cpu().numpy()  # [B]

        ori_kappa_std = self._get_kappa_std_ori(ori_data, ori_normal, k=self.curv_loss_knn)  # [b, n]
        grad, success_num = self.get_gradient(ori_data, target)
        with (torch.no_grad()):
            center = torch.median(ori_data, dim=-1)[0].clone().detach()  # [B, 3]
            r = torch.sum((ori_data - center[:, :, None]) ** 2, dim=1) ** 0.5  # [B, K]
            saliency = -1. * (r ** (self.alpha)) * torch.sum((ori_data - center[:, :, None]) * grad, dim=1)  # [B, K]
            normalized_saliency = (saliency - torch.min(saliency)) / (torch.max(saliency)
                                                                      - torch.min(saliency) + 1e-7)  # [B, K]
            normalized_std = (ori_kappa_std - torch.min(ori_kappa_std)) / (torch.max(ori_kappa_std)
                                                                           - torch.min(ori_kappa_std) + 1e-7)  # [B, K]
            score = 0.001 * normalized_saliency + normalized_std  # [B, K]
            # score = normalized_std  # [B, K]

            far_points_idx = self.farthest_point_sample(ori_data.transpose(1, 2).contiguous(),
                                                        self.total_central_num)  # [B, central_points]
            far_points = self.index_points(ori_data.transpose(1, 2).contiguous(),
                                           far_points_idx)  # [B, central_points, 3]
            far_knn = knn_points(far_points, ori_data.permute(0, 2, 1),
                                 K=self.curv_loss_knn + 1)  # [dists:[b,central_points,k+1], idx:[b,central_points,k+1]]
            far_knn_points = knn_gather(ori_data.permute(0, 2, 1), far_knn.idx).permute(0, 3, 1, 2).contiguous()
            far_knn_score = self.index_points(score.unsqueeze(2), far_knn.idx)
            total_central_points_idx = far_knn_score.topk(k=1, dim=2)[1].squeeze(dim=-1)

            total_central_points = self.index_points(
                far_knn_points.permute(0, 2, 3, 1).contiguous().view(-1, self.curv_loss_knn + 1, 3),
                total_central_points_idx.view(-1, 1))
            total_central_points = total_central_points.view(B, -1, 3).transpose(1, 2).contiguous()
            total_central_points_score = self.index_points(far_knn_score.view(-1, self.curv_loss_knn + 1, 1),
                                                          total_central_points_idx.view(-1, 1)).view(B, -1)

            central_points_score, tmp_idx = torch.topk(total_central_points_score, k=self.central_num)
            central_points = self.index_points(total_central_points.transpose(1, 2).contiguous(),
                                              tmp_idx).transpose(1, 2).contiguous()
        # weight factor for budget regularization
        # lower_bound = np.zeros((B,))
        # upper_bound = np.ones((B,)) * self.max_weight
        # scale_const = np.ones((B,)) * self.init_weight

        lower_bound = torch.ones(B) * 0
        scale_const = torch.ones(B) * self.init_weight
        upper_bound = torch.ones(B) * self.max_weight

        # init dist func
        chamfer_dist = ChamferDist()

        # weight factors for different distance loss
        cd_weight = np.ones((B,)) * self.cd_weight  # cd
        ker_weight = np.ones((B,)) * self.ker_weight  # def
        hide_weight = np.ones((B,)) * self.hide_weight

        # record best results in binary search
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, 3, K))
        o_best_pertmat = np.zeros((B, self.central_num, 3))
        o_best_gaussdelta = np.zeros((B, self.central_num))

        ori_kappa = self._get_kappa_ori(ori_data, ori_normal, k=self.curv_loss_knn)  # [b, n]
        # central_kappa_std = self.index_points(ori_kappa_std.unsqueeze(2), central_points_idx).squeeze(2)    # [b, central]
        far_kappa_std = self.index_points(ori_kappa.unsqueeze(2), far_knn.idx)
        total_central_kappa_std = self.index_points(far_kappa_std.view(-1, self.curv_loss_knn + 1, 1),
                                                   total_central_points_idx.view(-1, 1)).view(B, -1, 1)
        central_kappa_std = self.index_points(total_central_kappa_std, tmp_idx)
        # perform binary search
        for binary_step in range(self.binary_step):
            adv_data = ori_data.clone().detach()
            adv_data_score = score.clone().detach()
            init_trans_scale = torch.tensor(self.budget)

            perturb_mat = (torch.rand(B, self.central_num, 3) * init_trans_scale).cuda()

            # init delta for Gauss Kernel  [B, central_num]
            gauss_delta = (torch.ones((B, self.central_num)).cuda() * self.min_sigm + torch.rand((B, self.central_num))
                           .cuda() * (self.max_sigm - self.min_sigm))

            perturb_mat.requires_grad_()
            gauss_delta.requires_grad_()

            bestdist = np.array([1e10] * B)
            bestscore = np.array([-1] * B)
            # 5 3
            opt = optim.Adam([
                {'params': perturb_mat, 'lr': self.attack_lr * 5},
                {'params': gauss_delta, 'lr': self.attack_lr * 3},
            ], weight_decay=0.)

            adv_loss = torch.tensor(0.).cuda()
            dist_loss = torch.tensor(0.).cuda()

            total_time = 0.
            forward_time = 0.
            backward_time = 0.
            update_time = 0.

            # one step in binary search
            for iteration in range(self.num_iter):

                t1 = time.time()

                tmp_adv_data = torch.zeros_like(adv_data).cuda()
                tmp_deno = torch.zeros(B, 1, K).cuda()

                # clip transformation matrix
                with torch.no_grad():
                    perturb_mat.data = torch.clamp(perturb_mat.data, min=-self.budget, max=self.budget).detach()
                    gauss_delta.data = torch.clamp(gauss_delta.data, min=self.min_sigm, max=self.max_sigm).detach()

                ker_density_mat = self.kernel_density(central_points, ori_data, gauss_delta)  # [B, central_num, K]

                for j in range(0, self.central_num):
                    tmp_adv_data += (((adv_data + perturb_mat[:, j, :].unsqueeze(dim=2))
                                      * ker_density_mat[:, j, :].unsqueeze(dim=1)))
                    tmp_deno += ker_density_mat[:, j, :].unsqueeze(1)

                tmp_adv_data = tmp_adv_data / tmp_deno

                # forward passing
                logits = self.model(tmp_adv_data)  # [B, num_classes]
                if isinstance(logits, tuple):  # PointNet
                    logits = logits[0]

                t2 = time.time()
                forward_time += t2 - t1

                # print
                pred = torch.argmax(logits, dim=1)  # [B]
                # tar*
                success_num = (pred != target).sum().item()
                if iteration % (self.num_iter // 5) == 0:
                    print('Step {}, iteration {}, success {}/{}\n'
                          'adv_loss: {:.4f}, dist_loss: {:.4f}'.
                          format(binary_step, iteration, success_num, B,
                                 adv_loss.item(), dist_loss.item()))
                dist_loss = torch.tensor(0.).cuda()
                dist_val = self.transformation_loss(tmp_adv_data, perturb_mat, gauss_delta, batch_avg=False)

                pred_val = pred.detach().cpu().numpy()  # [B]
                input_val = tmp_adv_data.detach().cpu().numpy()  # [B, 3, K]
                perturb_mat_val = perturb_mat.view(self.central_num, B, 3).transpose(0,
                                                                                    1).clone().detach().cpu().numpy()  # [B, central_num, 3]
                gauss_delta_val = gauss_delta.clone().detach().cpu().numpy()

                # update
                for e, (dist, pred, label, ii, pert, gd) in \
                        enumerate(zip(dist_val, pred_val, label_val, input_val, perturb_mat_val, gauss_delta_val)):
                    # tar*
                    if dist < bestdist[e] and pred != label:
                        bestdist[e] = dist
                        bestscore[e] = pred
                    # tar*
                    if dist < o_bestdist[e] and pred != label:
                        # print('pert', pert, 'gauss_delta', gd)
                        o_bestdist[e] = dist
                        o_bestscore[e] = pred
                        o_bestattack[e] = ii
                        o_best_pertmat[e] = pert
                        o_best_gaussdelta[e] = gd

                t3 = time.time()
                update_time += t3 - t2

                # compute loss and backward
                adv_loss = self.adv_func(logits, target)

                # l2 dist or chamfer dist
                # l2_loss = l2_dist(tmp_adv_data, ori_data, torch.from_numpy(l2_weight))
                # dist_loss += l2_loss

                if cd_weight[0] != 0:
                    chamfer_loss = chamfer_dist(tmp_adv_data, ori_data, torch.from_numpy(cd_weight))
                    dist_loss += chamfer_loss

                if ker_weight[0] != 0:
                    transformation_loss = self.transformation_loss(tmp_adv_data, perturb_mat, gauss_delta)
                    dist_loss += (transformation_loss * ker_weight[0])

                if hide_weight[0] != 0:
                    hide_loss = self.curv_std_loss(gauss_delta, central_kappa_std, self.max_sigm,
                                                   self.min_sigm) * self.hide_weight
                    dist_loss += hide_loss.mean()

                scale_const = scale_const.float().cuda()
                loss = adv_loss + scale_const * dist_loss
                opt.zero_grad()
                loss.mean().backward()
                opt.step()

                t4 = time.time()
                backward_time += t4 - t3
                total_time += t4 - t1

                if iteration % 100 == 0:
                    print('total time: {:.2f}, for: {:.2f}, '
                          'back: {:.2f}, update: {:.2f}'.
                          format(total_time, forward_time,
                                 backward_time, update_time))
                    total_time = 0.
                    forward_time = 0.
                    backward_time = 0.
                    update_time = 0.
                    torch.cuda.empty_cache()

            # adjust weight factor
            for e, label in enumerate(label_val):
                # tar*
                if bestscore[e] != label and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:
                    # success
                    lower_bound[e] = max(lower_bound[e], scale_const[e])
                    scale_const[e] = (lower_bound[e] + upper_bound[e]) / 2.
                else:
                    # failure
                    upper_bound[e] = min(upper_bound[e], scale_const[e])
                    scale_const[e] = (lower_bound[e] + upper_bound[e]) / 2.

            torch.cuda.empty_cache()

        print('lower_bound is', lower_bound)
        fail_idx = (lower_bound == 0.)
        for fi in enumerate(fail_idx):
            if fi[1]:
                o_bestattack[fi[0]] = input_val[fi[0]]
                o_bestdist[fi[0]] = dist_val[fi[0]]
        # return final results
        success_num = (lower_bound > 0.).sum()
        print('Successfully attack {}/{}'.format(success_num, B))

        return o_bestattack.transpose((0, 2, 1)), success_num

    def chamfer_distance(self, points1, points2):
        dist1 = torch.cdist(points1, points2, p=2)  # [npoint, nsample]
        dist2 = torch.cdist(points2, points1, p=2)  # [npoint, nsample]
        min_dist1, _ = torch.min(dist1, dim=1)  # [npoint]
        min_dist2, _ = torch.min(dist2, dim=1)  # [npoint]
        chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)

        return chamfer_dist

    def kernel_density(self, central_points, pc, delta):
        B, _, N = pc.shape
        pc = pc.unsqueeze(3).repeat(1, 1, 1, self.central_num)
        central_points = central_points.unsqueeze(2).repeat(1, 1, N, 1)
        norm = torch.norm(pc - central_points, dim=1)  # [B, K, central_num]
        density = torch.exp(-norm / (2 * delta * delta).unsqueeze(1))  # [B, K, central_num]
        return density.transpose(1, 2).contiguous()

    def transformation_loss(self, adv_data, perturb_mat, gauss_delta, batch_avg=True):
        B, _, N = adv_data.shape
        if batch_avg:
            transformation_loss = torch.tensor(0.0).cuda()
            transformation_loss += (torch.norm(perturb_mat))
            transformation_loss += 1 * (torch.norm(1 - gauss_delta))
        else:
            transformation_loss = torch.zeros((B)).cuda()
            transformation_loss += (torch.norm(perturb_mat, dim=(1, 2)))
            transformation_loss += 1 * (torch.norm(1 - gauss_delta, dim=1))
        return transformation_loss / self.central_num

    def _get_kappa_ori(self, pc, normal, k=2):
        b, _, n = pc.size()
        inter_KNN = knn_points(pc.permute(0, 2, 1), pc.permute(0, 2, 1), K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]
        nn_pts = knn_gather(pc.permute(0, 2, 1), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :,
                 1:].contiguous()  # [b, 3, n ,k]
        vectors = nn_pts - pc.unsqueeze(3)
        vectors = self._normalize(vectors)
        return torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2)  # [b, n]

    def _get_kappa_std_ori(self, pc, normal, k=10):
        b, _, n = pc.size()
        inter_KNN = knn_points(pc.permute(0, 2, 1), pc.permute(0, 2, 1), K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]
        nn_pts = knn_gather(pc.permute(0, 2, 1), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :,
                 1:].contiguous()  # [b, 3, n ,k]
        vectors = nn_pts - pc.unsqueeze(3)
        vectors = self._normalize(vectors)
        kappa_ori = torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2)  # [b, n]

        nn_kappa = knn_gather(kappa_ori.unsqueeze(2), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :,
                   1:].contiguous()  # [b, 1, n ,k]
        std_kappa = torch.std(nn_kappa.squeeze(1), dim=2)
        return std_kappa

    def curv_std_loss(self, gauss_delta, central_kappa_std, max_delta, min_delta):
        norm_std = ((central_kappa_std - torch.min(central_kappa_std))
                    / (torch.max(central_kappa_std) - torch.min(central_kappa_std) + 1e-7))
        norm_gauss_delta = ((gauss_delta - min_delta) / (max_delta - min_delta + 1e-7))
        cos_sim = F.cosine_similarity(norm_std.squeeze(-1), norm_gauss_delta)
        return cos_sim

    def curvature_loss(self, adv_pc, ori_pc, adv_kappa, ori_kappa, k=2):
        b, _, n = adv_pc.size()
        intra_KNN = knn_points(adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1)  # [dists:[b,n,1], idx:[b,n,1]]
        onenn_ori_kappa = torch.gather(ori_kappa, 1, intra_KNN.idx.squeeze(-1)).contiguous()  # [b, n]

        curv_loss = ((adv_kappa - onenn_ori_kappa) ** 2).mean(-1)

        return curv_loss

    # get adv kappa through the estimation of ori pc
    def _get_kappa_adv(self, adv_pc, ori_pc, ori_normal, k=2):
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
        vectors = self._normalize(vectors)

        return torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2), normal  # [b, n], [b, 3, n]

    #################
    # UTILS FUNCTION
    #################
    def sample_and_group(self, npoint, radius, nsample, xyz, points, returnfps=False):
        """
        Input:
            npoint: number of samples
            radius: local region radius
            nsample: max sample number in local region
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, npoint, nsample, 3]    [B, npoint, C]??
            new_points: sampled points data, [B, npoint, nsample, 3+D]
        """
        B, N, C = xyz.shape
        S = npoint
        fps_idx = self.farthest_point_sample(xyz, npoint)
        new_xyz = self.index_points(xyz, fps_idx)
        idx = self.query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = self.index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

        if points is not None:
            grouped_points = self.index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
        else:
            new_points = grouped_xyz_norm
        if returnfps:
            return new_xyz, new_points, grouped_xyz, fps_idx
        else:
            return new_xyz, new_points

    def sample_and_group_all(self, xyz, points):
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

    def KNN_indices(self, x):
        pc = x.clone().detach().double()  # [B, 3, K]
        B, _, K = pc.shape
        # pc = pc.transpose(2, 1)
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K]
        assert dist.min().item() >= -1e-6
        # the min is self so we take top (k + 1)
        neg_value, nearest_indices = (-dist).topk(k=self.k + 1, dim=-1)  # [B, K, k + 1]
        value = -(neg_value[..., 1:])
        nearest_indices = nearest_indices[..., 1:]  # [B, K, k]
        return value, nearest_indices

    def square_distance(self, src, dst):
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

    def index_points(self, points, idx):
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

    def farthest_point_sample(self, xyz, npoint):
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
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    def query_ball_point(self, radius, nsample, xyz, new_xyz):
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
        sqrdists = self.square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        return group_idx

    def _normalize(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

    def get_gradient(self, data, target):
        """Calculate gradient on data.

        Args:
            data (torch.FloatTensor): victim data, [B, 3, K]
            target (torch.LongTensor): target output, [B]
        """
        input_data = data.clone().detach().float().cuda()
        input_data.requires_grad_()
        target = target.long().cuda()

        # forward pass
        logits = self.model(input_data)
        if isinstance(logits, tuple):  # PoitnNet
            logits = logits[0]
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        with torch.no_grad():
            grad = input_data.grad.detach()  # [B, 3, K]
            # success num
            pred = torch.argmax(logits, dim=-1)  # [B]
            num = (pred != target).sum().detach().cpu().item()
        return grad, num


def visualize_point_cloud(point_cloud, sampling_rate=1.0, point_size=0.05):
    # random sampling
    num_points = point_cloud.shape[0]
    num_sampled_points = int(num_points * sampling_rate)
    sampled_indices = np.random.choice(num_points, num_sampled_points, replace=False)
    sampled_points = point_cloud[sampled_indices, :]

    # plot point cloud
    x, y, z = sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
    mlab.figure(bgcolor=(1, 1, 1), size=(400, 400))
    mlab.points3d(x, y, z, scale_factor=point_size, color=(0.2, 0.5, 1.0))
    mlab.view(distance=5)

    # current_time2 = datetime.now()
    # formatted_time2 = current_time2.strftime("%Y%m%d%H%M%S")

    # mlab.savefig('./{}.png'.format(formatted_time2), size=(500, 500), figure=None, magnification='auto')
    mlab.show()

