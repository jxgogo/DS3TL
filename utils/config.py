import argparse
import torch


def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--repeat', type=int, default=5)
    # model
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--feature_dim', type=int, default=50)
    parser.add_argument('--sp_epochs', type=int, default=5, choices=[5, 100])
    parser.add_argument('--epochs', type=int, default=100)
    # batch size
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ulb_batch_size', type=int, default=128)
    # ************************************ semi-supervised related******************************************************
    parser.add_argument('--alg', type=str, default='simmatch', choices=['fullysupervised', 'pseudolabel', 'pimodel', 'temporal', 'meanteacher', 'vat', 'ict', 'mixmatch', 'fixmatch', 'simmatch', 'ds3tl'])
    parser.add_argument('--log', type=str, default='')
    parser.add_argument('--ratio_label', type=float, default=0.1)
    parser.add_argument('--label_exclude', type=str2bool, default=False)
    parser.add_argument('--transform', default='noise')
    parser.add_argument('--weak', type=float, default=0.5)
    parser.add_argument('--strong', type=float, default=1.0)
    parser.add_argument('--rampuplength', type=int, default=0)
    parser.add_argument('--T', default=0.5, type=float, help='temperature for sharpening')
    parser.add_argument('--threshold', default=0.9, type=float, help='confidence mask threshold')
    parser.add_argument('--ema_decay', default=0.99, type=float, help='ema weight decay')
    parser.add_argument('--eps', default=0.1, type=float, help='adv amplitude epsilon in VAT')
    parser.add_argument('--lamda_in', default=1, type=float, help='simmatch loss_in weight')
    parser.add_argument('--unlab_loss', type=str2bool, default=True)
    parser.add_argument('--ulb_weight', type=float, default=1.0)
    parser.add_argument('--ctr_loss', type=str2bool, default=True)
    parser.add_argument('--ctr_weight', type=float, default=5.0)
    parser.add_argument('--entmin_loss', type=str2bool, default=True)
    parser.add_argument('--entmin_weight', type=float, default=0.5)
    parser.add_argument('--topm', type=int, default=16)
    parser.add_argument('--threshold_push', type=float, default=0.9)
    # dataset
    parser.add_argument('--dataset', type=str, default='MI4C')
    args = parser.parse_args()
    classes = {'MI4C': 4}
    subjects = {'MI4C': 9}
    args.classes, args.subjects = classes[args.dataset], subjects[args.dataset]

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
