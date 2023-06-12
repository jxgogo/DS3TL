import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.config import create_parser
from utils.data_loader import Load_ssl_cross
from model import MLPFeature, Classifier, Head
from utils.pytorch_utils import init_weights, print_args, seed, weight_for_balanced_classes

from trainer import *
build_model = {
    'pseudolabel': pseudolabel_trainer.Trainer,
    'pimodel':     pimodel_trainer.Trainer,
    'temporal':    temporal_trainer.Trainer,
    'meanteacher': meanteacher_trainer.Trainer,
    'vat':         vat_trainer.Trainer,
    'ict':         ict_trainer.Trainer,
    'mixmatch':    mixmatch_trainer.Trainer,
    'fixmatch':    fixmatch_trainer.Trainer,
    'simmatch':    simmatch_trainer.Trainer,
    'ds3tl':       ds3tl_trainer.Trainer,
}


def run(args):
    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # print logging
    print_log = logging.StreamHandler()
    logger.addHandler(print_log)
    # save path
    log_name = args.alg if not len(args.log) else args.alg + f'_{args.log}'
    results_path = os.path.join('results', args.dataset, str(args.ratio_label), log_name)
    if not os.path.exists(results_path):
        os.system('mkdir -p ' + results_path)
    save_log = logging.FileHandler(os.path.join(results_path, log_name+'.log'), mode='w', encoding='utf8')
    logger.addHandler(save_log)
    logging.info(print_args(args) + '\n')
    # data path
    path = os.path.join('dataset', args.dataset)
    # model train
    acc_list, bca_list = [], []
    dfs = pd.DataFrame()
    for t in range(args.repeat):
        seed(t)
        acc, bca = [], []
        for s in range(args.subjects):
            logging.info('repeat id: {} test id: {}'.format(t, s))
            # load dataset
            labeled_dataset, unlabeled_dataset, test_dataset = \
                Load_ssl_cross(data_path=path,
                               test_id=s,
                               num_sub=args.subjects,
                               num_class=args.classes,
                               alg=args.alg,
                               lb_ratio=args.ratio_label,
                               label_exclude=args.label_exclude,
                               augment=args.transform,
                               weak=args.weak,
                               strong=args.strong)

            # data loader
            sample_weights = weight_for_balanced_classes(labeled_dataset.targets)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(labeled_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=0,
                                      sampler=sampler,
                                      drop_last=True)
            unlabeled_loader = DataLoader(unlabeled_dataset,
                                          batch_size=args.ulb_batch_size,
                                          shuffle=True,
                                          num_workers=0,
                                          drop_last=True)
            test_loader = DataLoader(test_dataset,
                                     batch_size=args.ulb_batch_size,
                                     shuffle=True,
                                     num_workers=0,
                                     drop_last=False)
            # initialize the model
            modelF = MLPFeature(input_dim=labeled_dataset.data.shape[1],
                                feature_dim=args.feature_dim,
                                t=0.1).to(args.device)
            modelC = Classifier(input_dim=args.feature_dim,
                                n_classes=len(torch.unique(labeled_dataset.targets))).to(args.device)
            modelF.apply(init_weights)
            modelC.apply(init_weights)
            # trainable parameters
            params = []
            for _, v in modelF.named_parameters():
                params += [{'params': v, 'lr': args.lr}]
            for _, v in modelC.named_parameters():
                params += [{'params': v, 'lr': args.lr}]
            if args.alg == 'simmatch' or args.alg == 'ds3tl':
                modelH = Head(input_dim=args.feature_dim,
                              dim=args.feature_dim).to(args.device)
                modelH.apply(init_weights)
                for _, v in modelH.named_parameters():
                    params += [{'params': v, 'lr': args.lr}]
            optimizer = optim.SGD(params,
                                  momentum=0.9,
                                  weight_decay=5e-4,
                                  nesterov=True)

            # **** PHASE 1: SUPERVISED TRAINING ON LABELED DATA ****
            _, modelF, modelC = supervised_trainer.train(modelF, modelC, train_loader, test_loader, optimizer, args.sp_epochs)
            # ********* PHASE 2: SEMI-SUPERVISED TRAINING  **********
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * args.lr_decay
            if args.alg == 'fullysupervised':
                hists, _, _ = supervised_trainer.train(modelF, modelC, train_loader, test_loader, optimizer, args.epochs)
            else:
                if args.alg == 'simmatch' or args.alg == 'ds3tl':
                    trainer = build_model[args.alg](modelF, modelC, modelH, optimizer, args)
                else:
                    trainer = build_model[args.alg](modelF, modelC, optimizer, args)
                hists = trainer.loop(args.epochs, train_loader, unlabeled_loader, test_loader)
            acc.append(hists[-1]['acc'])
            bca.append(hists[-1]['bca'])

            df = pd.DataFrame(hists)
            df["method"] = [log_name] * len(hists)
            df["rep"] = [t] * len(hists)
            df["s"] = [s] * len(hists)
            dfs = pd.concat([dfs, df], axis=0)
        print('*'*200)
        print(bca, '\n', 'Mean bca:', np.mean(bca))
        np.savez(os.path.join(results_path, f'r{str(t)}_results.npz'), acc=np.array(acc), bca=np.array(bca))
        acc_list.append(acc)
        bca_list.append(bca)
    # csv
    dfs.to_csv(os.path.join(results_path, log_name+'_raw.csv'), index=False)
    dfs_temp = dfs.groupby(by=["method", "rep", "epoch"]).mean().reset_index()
    avg_dfs = dfs_temp.groupby(by=["method", "epoch"]).mean().reset_index()
    avg_dfs = avg_dfs.sort_values(by=["method", "epoch"])
    avg_dfs["s"] = ["avg"] * len(avg_dfs)
    std_dfs = dfs_temp.groupby(by=["method", "epoch"]).std().reset_index()
    std_dfs = std_dfs.sort_values(by=["method", "epoch"])
    std_dfs["s"] = ["std"] * len(std_dfs)
    dfs = pd.concat([avg_dfs, std_dfs], axis=0)
    dfs = dfs.drop("rep", axis=1)
    dfs.to_csv(os.path.join(results_path, log_name+'_avg.csv'), index=False)
    # log
    logging.info(f'acc: {acc_list}')
    logging.info(f'bca: {bca_list}')
    logging.info(f'Mean -- acc on subjects: {np.mean(acc_list, axis=0)}, bca: {np.mean(bca_list, axis=0)}')
    logging.info(f'Std -- acc on subjects: {np.std(acc_list, axis=0)}, bca: {np.std(bca_list, axis=0)}')
    logging.info(f'Mean acc: {np.mean(acc_list)} Mean bca: {np.mean(bca_list)}')
    logging.info(f'Std acc: {np.std(np.mean(acc_list, axis=1))} Std bca: {np.std(np.mean(bca_list, axis=1))}')
    # csv
    pd_acc = pd.DataFrame(np.mean(acc_list, axis=0).reshape(1, -1), index=[log_name + '_acc'])
    pd_acc.insert(pd_acc.shape[1], 'mean', value=np.mean(acc_list))
    pd_acc.insert(pd_acc.shape[1], 'std', value=np.std(np.mean(acc_list, axis=1)))
    pd_acc.to_csv(results_path.replace(log_name, '') + 'results' + str(args.epochs) + '.csv', index=True, header=False, mode='a')
    pd_bca = pd.DataFrame(np.mean(bca_list, axis=0).reshape(1, -1), index=[log_name + '_bca'])
    pd_bca.insert(pd_bca.shape[1], 'mean', value=np.mean(bca_list))
    pd_bca.insert(pd_bca.shape[1], 'std', value=np.std(np.mean(bca_list, axis=1)))
    pd_bca.to_csv(results_path.replace(log_name, '') + 'results' + str(args.epochs) + '.csv', index=True, header=False, mode='a')


if __name__ == '__main__':
    args = create_parser()
    run(args)
