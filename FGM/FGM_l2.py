import torch
import numpy as np
import random
import time
from config import args


class FGM_l2:
    """Class for FGM_l2 attack.
    """

    def __init__(self, model, adv_func, budget, pre_head,
                 dist_metric='l2'):
        """FGM_l2 attack.
        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            budget (float): \epsilon ball for FGM_l2 attack
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        self.model = model.cuda()
        self.model.eval()
        self.adv_func = adv_func
        self.budget = budget
        self.dist_metric = dist_metric.lower()
        self.pre_head = pre_head

    def get_norm(self, x):
        """Calculate the norm of a given data x.

        Args:
            x (torch.FloatTensor): [B, 3, K]
        """
        # use global l2 norm here!
        norm = torch.sum(x ** 2, dim=[1, 2]) ** 0.5
        return norm

    def get_gradient(self, data, target, normalize=True):
        """Generate one step gradient.

        Args:
            data (torch.FloatTensor): batch pc, [B, 3, K]
            target (torch.LongTensor): target label, [B]
            normalize (bool, optional): whether l2 normalize grad. Defaults to True.
        """
        data = data.float().cuda()
        data.requires_grad_()
        target = target.long().cuda()

        # forward pass
        if self.pre_head != None:
            logits = self.model(self.pre_head(data))
        else:
            logits = self.model(data)
        # print('calculate logits')
        if isinstance(logits, tuple):
            logits = logits[0]  # [B, class]
        pred = torch.argmax(logits, dim=-1)  # [B]

        # backward pass
        loss = self.adv_func(logits, target)
        loss.backward()
        with torch.no_grad():
            grad = data.grad.detach()  # [B, 3, K]
            if normalize:
                # print('normalize')
                norm = self.get_norm(grad)
                grad = grad / (norm[:, None, None] + 1e-9)
        return grad, pred

    def attack(self, data, target):
        """One step FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3] [24, 1024, 3]
            target (torch.LongTensor): target label, [B] [24]
        """
        data = data.float().cuda().detach()
        if data.shape[1] == 1024:
            data = data.transpose(1, 2).contiguous()
        if data.shape[1] == 6:
            data = data[:, :3, :]
        pc = data.clone().detach()
        target = target.long().cuda()

        # gradient
        normalized_grad, _ = self.get_gradient(pc, target, True)  # [B, 3, K]
        # num = int(torch.nonzero(normalized_grad == 0, as_tuple=False).shape[0] / 24)
        perturbation = normalized_grad * self.budget
        with torch.no_grad():
            # perturbation = perturbation.transpose(1, 2).contiguous()
            # target版本'-'，让目标类别logit上升；untar版本'+'，让目标类别logit下降。
            data = data + perturbation  # no need to clip
            data = torch.clamp(data, min=-1, max=1).detach()
            # test attack performance
            if self.pre_head != None:
                logits = self.model(self.pre_head(data))
            else:
                logits = self.model(data)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred != target).sum().item()

        print('Successfully attack {}/{}'.format(success_num, data.shape[0]))
        torch.cuda.empty_cache()
        return data.transpose(1, 2).contiguous().detach(), success_num


class IFGM_l2(FGM_l2):
    """Class for IFGM_l2 attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter, pre_head,
                 dist_metric='l2'):
        """Iterative FGM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(IFGM_l2, self).__init__(model, adv_func, budget, dist_metric)
        self.clip_func = clip_func
        self.step_size = step_size
        self.num_iter = num_iter
        self.pre_head = pre_head

    def attack(self, data, target):
        """Iterative FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """


        if data.shape[1] > 6:
            data = data.transpose(1, 2).contiguous()
        if data.shape[1] == 6:
            data = data[:, :3, :]
        pc = data.clone().detach()
        B, _, K = data.shape
        # data = data.float().cuda().detach()
        # pc = data.clone().detach().transpose(1, 2).contiguous()

        pc = pc + torch.randn((B, 3, K)).cuda() * 1e-7
        ori_pc = pc.clone().detach()
        target = target.long().cuda()

        # start iteration
        for iteration in range(self.num_iter):
            # gradient
            normalized_grad, pred = self.get_gradient(pc, target, True)
            success_num = (pred != target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                print('iter {}/{}, success: {}/{}'.
                      format(iteration, self.num_iter,
                             success_num, B))
                torch.cuda.empty_cache()
            perturbation = self.step_size * normalized_grad

            # add perturbation and clip
            with torch.no_grad():
                # target版本'-'，让目标类别logit上升；untar版本'+'，让目标类别logit下降。
                pc = pc + perturbation
                pc = self.clip_func(pc, ori_pc)
                pc = torch.clamp(pc, min=-1., max=1.).detach()

        # end of iteration
        with torch.no_grad():
            if self.pre_head != None:
                logits = self.model(self.pre_head(pc))
            else:
                logits = self.model(pc)
            # logits = self.model(pc)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred != target).sum().item()
        print('Final success: {}/{}'.format(success_num, B))
        return pc.transpose(1, 2).contiguous().detach(), \
               success_num

