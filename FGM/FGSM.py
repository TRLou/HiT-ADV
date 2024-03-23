import torch
import numpy as np
import random
import time
from config import args


class FGSM:
    """Class for FGSM attack.
    """
    def __init__(self, model, adv_func, budget, pre_head,
                 dist_metric='linf'):
        """FGSM attack.
        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            budget (float): \epsilon ball for FGSM attack
            dist_metric (str, optional): type of constraint. Defaults to 'linf'.
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

    def get_gradient(self, data, target, normalize=False):
        """Generate one step gradient.
        Args:
            data (torch.FloatTensor): batch pc, [B, 3, K]
            target (torch.LongTensor): target label, [B]
            normalize (bool, optional): whether l2 normalize grad. Defaults to False.
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
                print('normalize')
                norm = self.get_norm(grad)
                grad = grad / (norm[:, None, None] + 1e-9)
        return grad, pred

    def attack(self, data, target, pre_head):
        """One step FGSM attack.
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
        normalized_grad, _ = self.get_gradient(pc, target, False)  # [B, 3, K]
        # num = int(torch.nonzero(normalized_grad == 0, as_tuple=False).shape[0] / 24)
        perturbation = normalized_grad.sign() * self.budget
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


class IFGSM(FGSM):
    """Class for I-FGSM attack.
    """
    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter, pre_head,
                 dist_metric='linf'):
        """Iterative FGSM attack.
        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'linf'.
        """
        super(IFGSM, self).__init__(model, adv_func, budget, dist_metric)
        self.clip_func = clip_func
        self.step_size = step_size
        self.num_iter = num_iter
        self.pre_head = pre_head

    def attack(self, data, target):
        """Iterative FGSM attack.
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

        pc = pc + torch.randn((B, 3, K)).cuda() * 1e-7
        ori_pc = pc.clone().detach()
        target = target.long().cuda()

        # start iteration
        for iteration in range(self.num_iter):
            # gradient
            normalized_grad, pred = self.get_gradient(pc, target, False)
            success_num = (pred != target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                print('iter {}/{}, success: {}/{}'.
                      format(iteration, self.num_iter,
                             success_num, B))
                torch.cuda.empty_cache()
            perturbation = self.step_size * normalized_grad.sign()

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
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred != target).sum().item()
        print('Final success: {}/{}'.format(success_num, B))
        return pc.transpose(1, 2).contiguous().detach(), \
               success_num


class MIFGSM(FGSM):
    """Class for MI-FGSM attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter, mu=1.,
                 dist_metric='linf'):
        """Momentum enhanced iterative FGSM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGSM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            mu (float): momentum factor
            dist_metric (str, optional): type of constraint. Defaults to 'linf'.
        """
        super(MIFGSM, self).__init__(model, adv_func,
                                    budget, dist_metric)
        self.clip_func = clip_func
        self.step_size = step_size
        self.num_iter = num_iter
        self.mu = mu

    def attack(self, data, target):
        """Momentum enhanced iterative FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        pc = data.clone().detach().transpose(1, 2).contiguous()
        pc = pc + torch.randn((B, 3, K)).cuda() * 1e-7
        ori_pc = pc.clone().detach()
        target = target.long().cuda()
        momentum_g = torch.tensor(0.).cuda()

        # start iteration
        for iteration in range(self.num_iter):
            # gradient
            grad, pred = self.get_gradient(pc, target, normalize=False)
            success_num = (pred != target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                print('iter {}/{}, success: {}/{}'.
                      format(iteration, self.num_iter,
                             success_num, B))
                torch.cuda.empty_cache()

            # grad is [B, 3, K]
            # normalized by l1 norm
            grad_l1_norm = torch.sum(torch.abs(grad), dim=[1, 2])  # [B]
            normalized_grad = grad / (grad_l1_norm[:, None, None] + 1e-9)
            momentum_g = self.mu * momentum_g + normalized_grad
            g_norm = self.get_norm(momentum_g)
            normalized_g = momentum_g / (g_norm[:, None, None] + 1e-9)
            perturbation = self.step_size * normalized_g.sign()

            # add perturbation and clip
            with torch.no_grad():
                # target版本'-'，让目标类别logit上升；untar版本'+'，让目标类别logit下降。
                pc = pc + perturbation
                pc = self.clip_func(pc, ori_pc)
                pc = torch.clamp(pc, min=-1., max=1.).detach()

        # end of iteration
        with torch.no_grad():
            logits = self.model(pc)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred != target).sum().item()
        print('Final success: {}/{}'.format(success_num, B))
        return pc.transpose(1, 2).contiguous().detach(), \
               success_num


class PGD(IFGSM):
    """Class for PGD attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter,
                 dist_metric='linf'):
        """PGD attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGSM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'linf'.
        """
        super(PGD, self).__init__(model, adv_func, clip_func,
                                  budget, step_size, num_iter,
                                  dist_metric)

    def attack(self, data, target):
        """PGD attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        # the only difference between IFGM and PGD is
        # the initialization of noise
        #        epsilon = self.budget / \
        #                  ((data.shape[1] * data.shape[2]) ** 0.5)

        epsilon = self.budget
        print('eps is', epsilon)
        init_perturbation = \
            torch.empty_like(data).uniform_(-epsilon, epsilon)
        with torch.no_grad():
            init_data = data + init_perturbation
        return super(PGD, self).attack(init_data, target)


class FGSM_RS(FGSM):
    def __init__(self, model, adv_func, budget, clip_func,
                 dist_metric='linf'):
        super(FGSM_RS, self).__init__(model, adv_func, budget,
                                  dist_metric)
        self.clip_func = clip_func

    def attack(self, data, target):
        ori_pc = data.clone().detach().cuda()
        epsilon = self.budget
        init_perturbation = \
            torch.empty_like(data).uniform_(-epsilon, epsilon)
        with torch.no_grad():
            init_data = data + init_perturbation  # [B, K, 3]

        init_data = init_data.float().cuda().detach()
        pc = init_data.clone().detach().transpose(1, 2).contiguous()
        target = target.long().cuda()

        # gradient
        normalized_grad, _ = super(FGSM_RS, self).get_gradient(pc, target, False)  # [B, 3, K]
        perturbation = normalized_grad.sign() * self.budget
        with torch.no_grad():
            perturbation = perturbation.transpose(1, 2).contiguous()  # [B, K, 3]
            # target版本'-'，让目标类别logit上升；untar版本'+'，让目标类别logit下降。
            adv_pc = init_data + perturbation  # no need to clip
            adv_pc = self.clip_func(adv_pc, ori_pc)
            adv_pc = torch.clamp(adv_pc, min=-1., max=1.).detach()

            # test attack performance
            logits = self.model(adv_pc.transpose(1, 2).contiguous())
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred != target).sum().item()
        print('Successfully attack {}/{}'.format(success_num, adv_pc.shape[0]))
        torch.cuda.empty_cache()
        return adv_pc.detach(), success_num

