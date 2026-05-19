import os
import sys
import torch
import argparse
import numpy as np
import random
import wandb
from datetime import datetime

from utils.helper import gen_data, pre_train, train_network_distill_c2kd
from utils.model_res import ImageNet, AudioNet
from utils.module import Tea, Stu


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

    tea = Tea(tea_type).to(device)
    stu = Stu(tea_type).to(device)

    optimizer = torch.optim.SGD([
        {'params': net.parameters()},
        {'params': tea_model.parameters()},
        {'params': tea.parameters()},
        {'params': stu.parameters()},
    ], lr=args.lr, momentum=0.9)

    return train_network_distill_c2kd(stu_type, tea_model, args.num_epochs, loader,
                                      net, device, optimizer, args, stu, tea)


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
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_frame', type=int, default=3)
    parser.add_argument('--weight', type=float, default=1)
    parser.add_argument('--audio_arch', type=str, default='resnet18')
    parser.add_argument('--image_arch', type=str, default='resnet18')
    parser.add_argument('--krc', type=float, default=0.0)
    parser.add_argument('--pre_train', type=int, default=0)
    parser.add_argument('--cmkd', type=int, default=1)
    parser.add_argument('--group', type=str, default='c2kd')
    parser.add_argument('--data-root', type=str, default='',
                        help='root directory of the VGGSound dataset')
    parser.add_argument('--ckpt-dir', type=str, default='./ckpts',
                        help='directory to save/load checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='directory for log files')

    args = parser.parse_args()

    # Setup log file
    os.makedirs(args.log_dir, exist_ok=True)
    _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    _lr_tag = f"lr{args.lr:.0e}".replace('-0', '-').replace('+0', '')
    _mode = 'pretrain' if args.pre_train else 'c2kd'
    _log_name = f"stu{args.stu_type}_{args.image_arch}_{args.audio_arch}_{_mode}_{_lr_tag}_ep{args.num_epochs}_{_ts}.log"
    sys.stdout = _Tee(os.path.join(args.log_dir, _log_name))
    print(f"Log: {os.path.join(args.log_dir, _log_name)}\n")
    print(args)

    device = torch.device('cpu') if args.gpu < 0 else torch.device(f'cuda:{args.gpu}')

    loader = gen_data(args.data_root, args.batch_size, args.num_workers, args)
    if args.pre_train:
        pre_train(args.stu_type, loader, args.num_epochs, args.lr, device, args)

    if args.cmkd:
        log_np = np.zeros((args.num_runs, 4))
        for run in range(args.num_runs):
            set_random_seed(run)
            print(f'Seed {run}')
            run_name = (f"data:vggsound, c2kd, seed:{run}, "
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
