import FGM.CWPert_args
from Dataset.ModelNet import ModelNetDataLoader
from Dataset.ShapeNetDataLoader import PartNormalDataset

import argparse
import torch

from model.pointnet_cls import get_model
from model.pointnet2_cls_ssg import get_model as get_model_pnp
from model.dgcnn_cls import DGCNN_cls
from model.pct_cls import Pct
from model import feature_models
from model.pointconv import PointConvDensityClsSsg as PointConvClsSsg

from util.adv_utils import LogitsAdvLoss, CrossEntropyAdvLoss, UntargetedLogitsAdvLoss
from util.other_utils import create_logger, eval_ASR

from ShapeAttack.HiT_ADV import HiT_ADV


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size in training')
    parser.add_argument('--num_class', type=int, default=40, help='class numbers')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
    parser.add_argument('--adv_func', type=str, default='cross_entropy',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--budget', type=float, default=0.55,
                        help='1 for l2 attack, 0.05 for linf attack')
    parser.add_argument('--attack_type', type=str, default='HiT-ADV', metavar='N',
                        help='Attack method to use')
    parser.add_argument('--num_iter', type=int, default=100,
                        help='iteration steps')
    parser.add_argument('--mu', type=float, default=1.,
                        help='momentum factor for MIFGM attack')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 16, 40], help='training on ModelNet10/40')
    parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet', 'dgcnn', 'pointnet++', 'pct'],
                        help='model for training')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--dropout', type=float, default=0.2, help='initial dropout rate')
    parser.add_argument('--k', type=int, default=5, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--curv_loss_knn', type=int, default=16, metavar='N', help='Num of nearest neighbors to use')

    parser.add_argument('--cd_weight', type=float, default=0.0001, help='cd_weight')
    parser.add_argument('--ker_weight', type=float, default=1., help='ker_weight')
    parser.add_argument('--hide_weight', type=float, default=1., help='hide_weight')

    parser.add_argument('--max_sigm', type=float, default=1.2, help='max_sigm')
    parser.add_argument('--min_sigm', type=float, default=0.1, help='min_sigm')

    parser.add_argument('--central_num', type=int, default=192)
    parser.add_argument('--total_central_num', type=int, default=256)

    parser.add_argument('--dataset', type=str, default='ModelNet', choices=['ModelNet', 'ShapeNetPart'],
                        help='model for training')
    parser.add_argument('--defense_method', type=str, default=None,
                        help='model for training')
    parser.add_argument('--eval_defense_method', type=str, default=None,
                        help='model for training')
    parser.add_argument('--kappa', type=float, default=30.,
                        help='min margin in logits adv loss')



    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    CWPerturb_args = FGM.CWPert_args.get_args()

    state_dict = torch.load('Checkpoint/PN_NT.checkpoint')
    args.step_size = args.budget * 2 / args.num_iter
    logger = create_logger('./log', 'eval_last', 'info')

    adv_func = CrossEntropyAdvLoss()
    CW_adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)

    if args.dataset == 'ModelNet':
        data_path = '../../PC_Dataset/modelnet40_normal_resampled'
        test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=10)
    elif args.dataset == 'ShapeNetPart':
        data_path = '../../PC_Dataset/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
        TEST_DATASET = PartNormalDataset(
            root=data_path,
            npoints=args.num_point,
            split='test',
            normal_channel=True
        )
        testDataLoader = torch.utils.data.DataLoader(
            TEST_DATASET,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=10
        )

    if args.model == 'pointnet':
        print('using pointnet')
        # model = get_model(args.num_class, normal_channel=args.use_normals).cuda()
        model = feature_models.PointNetFeatureModel(args.num_class, normal_channel=False).cuda()
    elif args.model == 'dgcnn':
        print('using dgcnn')
        model = DGCNN_cls(args, output_channels=args.num_class).cuda()
    elif args.model == 'pointnet++':
        print('using pointnet++')
        model = get_model_pnp(args.num_class, normal_channel=False).cuda()
    elif args.model == 'pct':
        print('using pct')
        model = Pct(args, output_channels=args.num_class).cuda()
    else:
        raise Exception("Not implemented")
    # model = torch.nn.DataParallel(model)

    model.load_state_dict(state_dict['model_state_dict'])
    # model.load_state_dict(state_dict['last'])

    HiT_attacker = HiT_ADV(model, adv_func=CW_adv_func, attack_lr=CWPerturb_args.attack_lr,
                               central_num=args.central_num, total_central_num=args.total_central_num,
                               init_weight=10., max_weight=80., binary_step=CWPerturb_args.binary_step,
                               num_iter=CWPerturb_args.num_iter, clip_func=None,
                               cd_weight=args.cd_weight, ker_weight=args.ker_weight,
                               hide_weight=args.hide_weight, curv_loss_knn=args.curv_loss_knn,
                               max_sigm=args.max_sigm, min_sigm=args.min_sigm,
                               budget=args.budget)

    eval_ASR(model, testDataLoader, args, HiT_attacker)

