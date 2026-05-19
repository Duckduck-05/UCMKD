import os
import sys
import torch
import argparse
import numpy as np
import random
import wandb
from datetime import datetime

from utils.helper import (
    gen_data, gen_data_challenging, pre_train,
    train_network_distill_unpair_sumall,
    train_network_distill_unpair_ce,
    train_network_distill_unpair_bilevel,
    train_network_distill_unpair_vanillaKD,
    train_network_distill_unpair_fea,
    train_network_distill_unpair_reviewkd,
    train_network_distill_unpair_norm,
    train_network_distill_unpair_bilevel_with_different_metric,
)
from utils.model_res import ImageNet, AudioNet
from utils.module import Tea, Stu, TeaViT, StuViT


class _Tee:
    """Write to both terminal and a log file simultaneously."""
    def __init__(self, log_path):
        self._terminal = sys.stdout
        self._file = open(log_path, 'w', buffering=1)

    def write(self, msg):
        self._terminal.write(msg)
        self._file.write(msg)

    def flush(self):
        self._terminal.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_overlap_tag(loader, device, args):
    stu_type = args.stu_type
    tea_type = 1 - stu_type

    tea_model = ImageNet(args).to(device) if tea_type == 0 else AudioNet(args).to(device)
    arch = args.image_arch if stu_type == 1 else args.audio_arch
    if stu_type == 1:
        print(f'teacher:image ({args.image_arch}); student:audio ({args.audio_arch})')
    else:
        print(f'teacher:audio ({args.audio_arch}); student:image ({args.image_arch})')

    ckpt_path = os.path.join(args.ckpt_dir, f'teacher_mod_{tea_type}_{arch}_{args.num_frame}_overlap.pkl')
    tea_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    print('Finish loading teacher model')

    net = ImageNet(args).to(device) if stu_type == 0 else AudioNet(args).to(device)

    _image_is_vit = args.image_arch in ('vit_b_16', 'vit_l_16')
    _audio_is_vit = args.audio_arch in ('vit_s_16', 'vit_l_16')
    _tea_is_vit = _image_is_vit if tea_type == 0 else _audio_is_vit
    _stu_is_vit = _image_is_vit if stu_type == 0 else _audio_is_vit

    tea = TeaViT(feat_dim=tea_model.feature_dim, num_classes=8).to(device) if _tea_is_vit else Tea().to(device)
    stu = StuViT(feat_dim=net.feature_dim, num_classes=8).to(device) if _stu_is_vit else Stu().to(device)

    all_params = (list(net.parameters()) + list(tea_model.parameters())
                  + list(tea.parameters()) + list(stu.parameters()))
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(all_params, lr=args.lr,
                                      betas=(args.beta1, args.beta2),
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(all_params, lr=args.lr, momentum=0.9)

    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.min_lr / args.lr, end_factor=1.0, total_iters=args.warmup_epoch)
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs - args.warmup_epoch, eta_min=args.min_lr)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.warmup_epoch])

    method_map = {
        'cost_metric': train_network_distill_unpair_bilevel_with_different_metric,
        'bilevel':     train_network_distill_unpair_bilevel,
        'ce':          train_network_distill_unpair_ce,
        'sumall':      train_network_distill_unpair_sumall,
        'vanillaKD':   train_network_distill_unpair_vanillaKD,
        'feadistill':  train_network_distill_unpair_fea,
        'reviewkd':    train_network_distill_unpair_reviewkd,
        'norm':        train_network_distill_unpair_norm,
    }
    train_fn = method_map[args.method_type]
    return train_fn(stu_type, tea_model, args.num_epochs, loader, net, device,
                    optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--stu-type', type=int, default=0, help='0=image student, 1=audio student')
    parser.add_argument('--num-runs', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--batch-size2', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_frame', type=int, default=1)
    parser.add_argument('--weight', type=float, default=1)
    parser.add_argument('--audio_arch', type=str, default='resnet18')
    parser.add_argument('--image_arch', type=str, default='resnet18')
    parser.add_argument('--krc', type=float, default=0.0)
    parser.add_argument('--pre_train', type=int, default=0)
    parser.add_argument('--cmkd', type=int, default=1)
    parser.add_argument('--group', type=str, default='c2kd')
    parser.add_argument('--weight-decay', default=1e-5, type=float)
    parser.add_argument('--min-lr', default=1e-5, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--warmup-epoch', default=5, type=int)
    parser.add_argument('--method_type', type=str, default='ce')
    parser.add_argument('--metric', type=str, default='cosine', choices=['l1', 'l2', 'cosine', 'chordal'])
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'])
    parser.add_argument('--pre-train-epochs', default=10, type=int)
    parser.add_argument('--la-weight', type=float, default=1.0)
    parser.add_argument('--fa-weight', type=float, default=1.0)
    parser.add_argument('--ot-reg', type=float, default=0.1)
    parser.add_argument('--ot-iter', type=int, default=100)
    parser.add_argument('--kd-temp', type=float, default=4.0)
    parser.add_argument('--kd-alpha', type=float, default=0.9)
    parser.add_argument('--max-norm', type=float, default=5.0)
    parser.add_argument('--n1-steps', type=int, default=1)
    parser.add_argument('--n2-steps', type=int, default=1)
    parser.add_argument('--inner-wd', type=float, default=0.05)
    parser.add_argument('--inner-lr', type=float, default=1e-5)
    parser.add_argument('--review-weight', type=float, default=0.0, dest='review_weight')
    parser.add_argument('--norm-weight', type=float, default=0.0, dest='norm_weight')
    parser.add_argument('--norm-n', type=int, default=4, dest='norm_n')
    parser.add_argument('--data-root', type=str, default='')
    parser.add_argument('--ckpt-dir', type=str, default='./ckpts')
    parser.add_argument('--challenge', action='store_true', default=False)
    parser.add_argument('--challenge_preset', type=str, default='moderate',
                        choices=['clean', 'mild', 'moderate', 'hard',
                                 'marginal_only', 'domain_only', 'imbalance_only'])
    parser.add_argument('--marginal_mismatch', action='store_true', default=False)
    parser.add_argument('--marginal_ratio', type=float, default=0.5)
    parser.add_argument('--domain_shift', action='store_true', default=False)
    parser.add_argument('--domain_shift_level', type=float, default=0.5)
    parser.add_argument('--label_imbalance', action='store_true', default=False)
    parser.add_argument('--imbalance_modality', type=str, default='audio', choices=['audio', 'image'])
    parser.add_argument('--imbalance_factor', type=float, default=10.0)

    args = parser.parse_args()

    # Setup log file
    _log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(_log_dir, exist_ok=True)
    _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    _lr_tag = f"lr{args.lr:.0e}".replace('-0', '-').replace('+0', '')
    _mode = 'pretrain' if args.pre_train else args.method_type
    _log_name = f"stu{args.stu_type}_{args.image_arch}_{args.audio_arch}_{_mode}_{_lr_tag}_ep{args.num_epochs}_{_ts}.log"
    sys.stdout = _Tee(os.path.join(_log_dir, _log_name))
    print(f"Log: {os.path.join(_log_dir, _log_name)}\n")
    print(args)

    device = torch.device('cpu') if args.gpu < 0 else torch.device(f'cuda:{args.gpu}')
    data_dir = args.data_root

    if args.challenge:
        loader = gen_data_challenging(data_dir, args.batch_size, args.num_workers, args)
    else:
        loader = gen_data(data_dir, args.batch_size, args.num_workers, args)

    if args.pre_train:
        loader_fb = gen_data(data_dir, args.batch_size2, args.num_workers, args)
        pre_train(args.stu_type, loader, args.pre_train_epochs, args.lr, device, args)

    if args.cmkd:
        log_np = np.zeros((args.num_runs, 4))
        for run in range(args.num_runs):
            set_random_seed(run)
            print(f'Seed {run}')
            run_name = (f"data:ravdess, {args.method_type}, seed:{run}, "
                        f"lr{args.lr}, bs{args.batch_size}, ep{args.num_epochs}, stu{args.stu_type}")
            wandb.init(
                entity='cmkd', project='experiments', name=run_name,
                config=vars(args), group=args.group, reinit='finish_previous',
                mode=os.environ.get('WANDB_MODE', 'online'),
                settings=wandb.Settings(init_timeout=30),
            )
            log_np[run, :] = eval_overlap_tag(loader, device, args)
            wandb.finish()

        log_mean = np.mean(log_np, axis=0)
        log_std = np.std(log_np, axis=0)
        print(f'Finish {args.num_runs} runs')
        print(f'Student Val Acc {log_mean[0]:.3f} ± {log_std[0]:.3f} | Test Acc {log_mean[1]:.3f} ± {log_std[1]:.3f}')
        print(f'Teacher Val Acc {log_mean[2]:.3f} ± {log_std[2]:.3f} | Test Acc {log_mean[3]:.3f} ± {log_std[3]:.3f}')
        if args.stu_type == 1:
            print(f'teacher:image ({args.image_arch}); student:audio ({args.audio_arch})')
        else:
            print(f'teacher:audio ({args.audio_arch}); student:image ({args.image_arch})')
