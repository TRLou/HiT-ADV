import argparse

'''PARAMETERS'''
parser = argparse.ArgumentParser('training')
parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
parser.add_argument('--adv_func', type=str, default='cross_entropy',
                    choices=['logits', 'cross_entropy'],
                    help='Adversarial loss function to use')
parser.add_argument('--budget', type=float, default=1,
                    help='FGM attack budget')
parser.add_argument('--num_iter', type=int, default=10,
                    help='IFGM iterate step')
parser.add_argument('--mu', type=float, default=1.,
                    help='momentum factor for MIFGM attack')
parser.add_argument('--gpu', type=str, default='7', help='specify gpu device')
parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='Path to latest checkpoint (default: none)')
parser.add_argument('--val_interval', default=1, type=int, help='Number of epochs between evaluations')
parser.add_argument('--model-dir', default='./log/models', help='Saving path of models')
parser.add_argument('--mix_loss', action='store_true', default=False, help='mix loss')
parser.add_argument('--num_drop', type=int, default=0, help='Point drop')
parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
parser.add_argument('--dropout', type=float, default=0.5, help='initial dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
parser.add_argument('--train_attack_type', type=str, default='none', metavar='N',
                    help='Attack method to use')
parser.add_argument('--test_attack_type', type=str, default='PGD', metavar='N',
                    help='Attack method to use')
parser.add_argument('--epochs', default=200, type=int, help='number of epoch in training')

parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')

parser.add_argument('--model', type=str, default='dgcnn', choices=['pointnet', 'dgcnn', 'pointnet++', 'pct', 'pointconv']
                    , help='model for training')
parser.add_argument('--dataset', type=str, default='ShapeNetPart', choices=['ModelNet', 'ShapeNetPart'],
                    help='model for training')

parser.add_argument('--num_class', type=int, default=40, help='class numbers')
parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')

parser.add_argument('--scheduler', type=str, default='step', metavar='N',
                    choices=['cos', 'step'],
                    help='Scheduler to use, [cos, step]')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()