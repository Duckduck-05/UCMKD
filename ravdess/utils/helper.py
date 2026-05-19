import random
import numpy as np
import torch
import os
import time
import wandb
import torch.nn.functional as F
import ot
import sys as _sys, os as _os

from copy import deepcopy
from torch.utils.data import DataLoader
from scipy.stats.stats import kendalltau

from utils.model_res import ImageNet, AudioNet
from utils.RavvdessDataset import RavvdessDataset, CHALLENGE_PRESETS

_ucmkd_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ucmkd_root not in _sys.path:
    _sys.path.insert(0, _ucmkd_root)
from kd_losses import ReviewKDLoss, NORMLoss


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()


def adjust_lr(lr=1e-2, iter=None, max_iter=100, power=0.9, optimizer=None):
    cur_lr = 1e-4 + (lr - 1e-4) * ((1 - float(iter) / max_iter) ** power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(loader, device, net, in_type):
    correct, v_loss, total = 0, 0, 0
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in loader:
            img_inputs, aud_inputs, labels = (
                data['image'].to(device), data['audio'].to(device), data['label'].to(device))
            total += labels.size(0)
            if in_type == 0:
                outputs, _, _ = net(img_inputs)
            elif in_type == 1:
                outputs, _, _ = net(aud_inputs)
            elif in_type == 2:
                outputs, _, _ = net(img_inputs, aud_inputs)
            else:
                raise ValueError('in_type must be 0, 1, or 2')
            _, predicted = torch.max(outputs.detach(), 1)
            correct += (predicted == labels).sum().item()
            v_loss += criterion(outputs, labels).item()
    return v_loss / len(loader), 100 * correct / total


def evaluate_allacc(loader, device, net, in_type):
    _, train_acc = evaluate(loader['train'], device, net, in_type)
    _, val_acc   = evaluate(loader['val'],   device, net, in_type)
    _, test_acc  = evaluate(loader['test'],  device, net, in_type)
    return train_acc, val_acc, test_acc


def pairwise_ot_cost(stu_f, tea_f, metric='chordal'):
    if metric == 'l2':
        return torch.cdist(stu_f, tea_f, p=2)
    if metric == 'l1':
        return torch.cdist(stu_f, tea_f, p=1)
    stu = F.normalize(stu_f, p=2, dim=1)
    tea = F.normalize(tea_f, p=2, dim=1)
    if metric == 'chordal':
        return torch.cdist(stu, tea, p=2)
    if metric == 'cosine':
        return 1.0 - stu @ tea.T
    raise ValueError(f'Unknown metric: {metric}')


def ntkl(logits_student, logits_teacher, target, mask, criterion4, temperature=1):
    gt_mask = _get_gt_mask(logits_student, target)
    logits_teacher = logits_teacher * (~gt_mask)
    logits_student = logits_student * (~gt_mask)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    if mask.sum() == 0:
        return torch.tensor(0)
    return (mask * criterion4(log_pred_student, pred_teacher.detach()).sum(1)).mean()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def gen_data(data_dir, batch_size, num_workers, args):
    num_workers = min(num_workers, 8)
    audio_dir  = os.path.join(data_dir, 'aud_features')
    image_dir  = os.path.join(data_dir, 'vid_features')
    data_file  = os.path.join(data_dir, 'data_file')

    train_dataset = RavvdessDataset(os.path.join(data_file, 'spa_dl.csv'),  audio_dir, image_dir, mode='train')
    val_dataset   = RavvdessDataset(os.path.join(data_file, 'spa_val.csv'), audio_dir, image_dir, mode='val')
    test_dataset  = RavvdessDataset(os.path.join(data_file, 'spa_test.csv'),audio_dir, image_dir, mode='test')

    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        'val':   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        'test':  DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }


def gen_data_challenging(data_dir, batch_size, num_workers, args):
    num_workers = min(num_workers, 8)
    audio_dir = os.path.join(data_dir, 'aud_features')
    image_dir = os.path.join(data_dir, 'vid_features')
    data_file = os.path.join(data_dir, 'data_file')

    preset = getattr(args, 'challenge_preset', 'clean')
    if preset in CHALLENGE_PRESETS:
        challenge_kw = CHALLENGE_PRESETS[preset].copy()
        print(f'[DataLoader] Using challenge preset: {preset}')
    else:
        challenge_kw = {}
        print(f'[DataLoader] Unknown preset "{preset}", using individual flags')

    if getattr(args, 'marginal_mismatch', False):
        challenge_kw['marginal_mismatch'] = True
        challenge_kw['marginal_ratio'] = getattr(args, 'marginal_ratio', 0.5)
    if getattr(args, 'domain_shift', False):
        challenge_kw['domain_shift'] = True
        challenge_kw['domain_shift_level'] = getattr(args, 'domain_shift_level', 0.5)
    if getattr(args, 'label_imbalance', False):
        challenge_kw['label_imbalance'] = True
        challenge_kw['imbalance_modality'] = getattr(args, 'imbalance_modality', 'audio')
        challenge_kw['imbalance_factor'] = getattr(args, 'imbalance_factor', 10.0)

    train_dataset = RavvdessDataset(
        os.path.join(data_file, 'spa_dl.csv'), audio_dir, image_dir, mode='train', **challenge_kw)
    val_dataset   = RavvdessDataset(
        os.path.join(data_file, 'spa_val.csv'), audio_dir, image_dir, mode='val')
    test_dataset  = RavvdessDataset(
        os.path.join(data_file, 'spa_test.csv'), audio_dir, image_dir, mode='test')

    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        'val':   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        'test':  DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }


# ---------------------------------------------------------------------------
# Shared epoch logging
# ---------------------------------------------------------------------------

def _log_epoch(epoch, epochs, loader, net, tea_model, stu_type,
               device, train_loss, val_best_acc, test_best_acc,
               val_best_acc_t, test_best_acc_t, extra_metrics=None):
    _, train_acc   = evaluate(loader['train'], device, net,       stu_type)
    val_loss, val_acc   = evaluate(loader['val'],   device, net,       stu_type)
    test_loss, test_acc = evaluate(loader['test'],  device, net,       stu_type)
    _, train_acc_t      = evaluate(loader['train'], device, tea_model, 1 - stu_type)
    val_loss_t, val_acc_t   = evaluate(loader['val'],   device, tea_model, 1 - stu_type)
    test_loss_t, test_acc_t = evaluate(loader['test'],  device, tea_model, 1 - stu_type)

    log = {
        'train/loss': train_loss,
        'student/train_acc': train_acc, 'student/val_acc': val_acc, 'student/test_acc': test_acc,
        'teacher/train_acc': train_acc_t, 'teacher/val_acc': val_acc_t, 'teacher/test_acc': test_acc_t,
        'student/best_val_acc': val_best_acc, 'student/best_test_acc': test_best_acc,
        'teacher/best_val_acc': val_best_acc_t, 'teacher/best_test_acc': test_best_acc_t,
    }
    if extra_metrics:
        log.update(extra_metrics)
    wandb.log(log, step=epoch)

    print(f"Epoch {epoch}/{epochs} | Loss {train_loss:.3f}")
    print(f"  Student  Train|Val|Test: {train_acc:.3f}|{val_acc:.3f}|{test_acc:.3f}")
    print(f"  Teacher  Train|Val|Test: {train_acc_t:.3f}|{val_acc_t:.3f}|{test_acc_t:.3f}")
    print(f"  Best Stu Val|Test: {val_best_acc:.3f}|{test_best_acc:.3f}")
    print(f"  Best Tea Val|Test: {val_best_acc_t:.3f}|{test_best_acc_t:.3f}")
    print('-' * 70)

    return val_acc, test_acc, val_acc_t, test_acc_t


def _save_model(model, args, stu_type, test_best_acc, test_best_acc_t, subdir='results/our'):
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir,
        f'distillednet_mod_{stu_type}_{args.num_frame}_kdweight{args.weight}'
        f'_stu_acc_{round(test_best_acc, 2)}_tea_acc_{round(test_best_acc_t, 2)}.pkl')
    torch.save(model.state_dict(), path)
    print(f'Saving best model to {path}')


def _freeze_teacher(tea_model):
    tea_model.eval()
    for param in tea_model.parameters():
        param.requires_grad = False


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_network_distill_c2kd(stu_type, tea_model, epochs, loader, net, device, optimizer, args, tea, stu):
    val_best_acc = test_best_acc = val_best_acc_t = test_best_acc_t = 0
    model_best = net
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    net.train(); tea.train(); stu.train(); tea_model.train()

    for epoch in range(epochs):
        train_loss = corr_mean = 0.0
        for data in loader['train']:
            img_inputs, aud_inputs, labels = (
                data['image'].to(device), data['audio'].to(device), data['label'].to(device))

            if stu_type == 0:
                outputs, _, stu_fit = net(img_inputs)
                pseu_label, _, tea_fit = tea_model(aud_inputs)
            else:
                outputs, _, stu_fit = net(aud_inputs)
                pseu_label, _, tea_fit = tea_model(img_inputs)

            tea_logits = tea(tea_fit)
            stu_logits = stu(stu_fit)

            optimizer.zero_grad()
            criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
            XE_s = F.cross_entropy(outputs, labels, reduction='none')
            XE_t = F.cross_entropy(pseu_label, labels, reduction='none')
            kl_t  = criterion3(F.log_softmax(tea_logits, -1), F.softmax(pseu_label.detach(), -1))
            kl_t2 = criterion3(F.log_softmax(pseu_label, -1), F.softmax(tea_logits.detach(), -1))
            kl_s  = criterion3(F.log_softmax(stu_logits, -1), F.softmax(outputs.detach(), -1))
            kl_s2 = criterion3(F.log_softmax(outputs, -1), F.softmax(stu_logits.detach(), -1))

            kendall = np.array([
                kendalltau(tea_logits[i].cpu().detach().numpy(),
                           stu_logits[i].cpu().detach().numpy())[0]
                for i in range(tea_logits.size(0))])
            kendall_t = torch.from_numpy(kendall).cuda()
            mask = (kendall_t > args.krc).int()

            kl_st = ntkl(stu_logits, tea_logits.detach(), labels, mask, criterion4)
            kl_ts = ntkl(tea_logits, stu_logits.detach(), labels, mask, criterion4)

            loss = (XE_t.mean() + kl_t + kl_ts + kl_t2) + (XE_s.mean() + kl_s + kl_st + kl_s2)
            loss.backward()
            optimizer.step()
            adjust_lr(iter=epoch, optimizer=optimizer)
            train_loss += loss.item()
            corr_mean += kendall_t.mean().item()

        val_acc, test_acc, val_acc_t, test_acc_t = _log_epoch(
            epoch, epochs, loader, net, tea_model, stu_type, device,
            train_loss / len(loader['train']), val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t)

        if val_acc >= val_best_acc:
            val_best_acc, test_best_acc, model_best = val_acc, test_acc, deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t, test_best_acc_t = val_acc_t, test_acc_t

    print(f'Training finish! Best Val|Test: {val_best_acc:.3f}|{test_best_acc:.3f}')
    _save_model(model_best, args, stu_type, test_best_acc, test_best_acc_t, subdir='results')
    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


def train_network_distill_unpair_sumall(stu_type, tea_model, epochs, loader, net, device,
                                        optimizer, warmup_lr_scheduler, main_lr_scheduler,
                                        lr_scheduler, args, tea, stu):
    val_best_acc = test_best_acc = val_best_acc_t = test_best_acc_t = 0
    model_best = net
    net.train()
    _freeze_teacher(tea_model)

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = CE_total = FA_total = LA_total = 0.0

        for data in loader['train']:
            img_inputs, aud_inputs, labels = (
                data['image'].to(device), data['audio'].to(device), data['label'].to(device))

            if stu_type == 0:
                outputs, stu_f, _ = net(img_inputs)
                pseu_label, tea_f, _ = tea_model(aud_inputs)
            else:
                outputs, stu_f, _ = net(aud_inputs)
                pseu_label, tea_f, _ = tea_model(img_inputs)

            optimizer.zero_grad()
            CE_loss = F.cross_entropy(outputs, labels, reduction='none')

            bs = stu_f.size(0)
            a = torch.ones(bs, device=stu_f.device) / bs
            b = torch.ones(bs, device=tea_f.device) / bs
            M = torch.clamp(pairwise_ot_cost(stu_f, tea_f, metric=args.metric), min=0.0)
            FA_loss = torch.mean(ot.sinkhorn2(a, b, M, reg=args.ot_reg, numItermax=args.ot_iter, method='sinkhorn'))

            stu_latent_2_tea = tea_model.fc(stu_f)
            log_probs_s = F.log_softmax(outputs, dim=-1)
            log_probs_t = F.log_softmax(stu_latent_2_tea, dim=-1)
            kappa = F.softmax(stu_latent_2_tea, dim=-1).detach()
            LA_loss = F.nll_loss(log_probs_s - kappa * log_probs_t, labels)

            loss = CE_loss.mean() + args.fa_weight * FA_loss + args.la_weight * LA_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            optimizer.step()
            train_loss += loss.item(); CE_total += CE_loss.mean().item()
            FA_total += FA_loss.item(); LA_total += LA_loss.item()

        lr_scheduler.step()
        n = len(loader['train'])
        val_acc, test_acc, val_acc_t, test_acc_t = _log_epoch(
            epoch, epochs, loader, net, tea_model, stu_type, device, train_loss / n,
            val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t,
            extra_metrics={'train/CE_loss': CE_total/n, 'train/FA_loss': FA_total/n, 'train/LA_loss': LA_total/n})
        print(f"  epoch_duration: {time.time()-t0:.1f}s")

        if val_acc >= val_best_acc:
            val_best_acc, test_best_acc, model_best = val_acc, test_acc, deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t, test_best_acc_t = val_acc_t, test_acc_t

    print(f'Training finish! Best Val|Test: {val_best_acc:.3f}|{test_best_acc:.3f}')
    _save_model(model_best, args, stu_type, test_best_acc, test_best_acc_t)
    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


def _bilevel_step(stu_type, net, tea_model, img_inputs, aud_inputs, labels, args, n1_steps, n2_steps):
    """Run bilevel inner loop on a temporary copy of net; return the copy."""
    net_tmp = deepcopy(net)
    net_tmp.train()
    opt_tmp = torch.optim.AdamW(net_tmp.parameters(), lr=args.inner_lr, weight_decay=args.inner_wd)

    for _ in range(n1_steps):
        opt_tmp.zero_grad()
        if stu_type == 0:
            _, stu_f, _ = net_tmp(img_inputs)
            _, tea_f, _ = tea_model(aud_inputs)
        else:
            _, stu_f, _ = net_tmp(aud_inputs)
            _, tea_f, _ = tea_model(img_inputs)
        bs = stu_f.size(0)
        a = torch.ones(bs, device=stu_f.device) / bs
        b = torch.ones(bs, device=tea_f.device) / bs
        M = pairwise_ot_cost(stu_f, tea_f, metric=args.metric)
        FA_loss = torch.mean(ot.sinkhorn2(a, b, M, reg=args.ot_reg, numItermax=args.ot_iter))
        FA_loss.backward()
        opt_tmp.step()

    Cross_Entropy = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
    for _ in range(n2_steps):
        opt_tmp.zero_grad()
        if stu_type == 0:
            stu_f, _ = net_tmp.forward_encoder(img_inputs)
        else:
            stu_f, _ = net_tmp.forward_encoder(aud_inputs)
        outputs_tmp = net_tmp.forward_head(stu_f.detach())
        if stu_f.shape[-1] == tea_model.feature_dim:
            with torch.no_grad():
                stu_latent_2_tea = tea_model.fc(stu_f)
            kappa = F.softmax(stu_latent_2_tea, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            LA_loss = (Cross_Entropy(outputs_tmp, labels) - kappa * Cross_Entropy(stu_latent_2_tea, labels)).mean()
        else:
            LA_loss = Cross_Entropy(outputs_tmp, labels).mean()
        LA_loss.backward()
        opt_tmp.step()

    return net_tmp, FA_loss, LA_loss


def _bilevel_outer_update(net, net_tmp, optimizer, args, img_inputs, aud_inputs, labels, stu_type):
    if stu_type == 0:
        outputs_final, _, _ = net_tmp(img_inputs)
    else:
        outputs_final, _, _ = net_tmp(aud_inputs)
    CE_loss = F.cross_entropy(outputs_final, labels, label_smoothing=0.1)
    net_tmp.zero_grad()
    CE_loss.backward()
    optimizer.zero_grad()
    for real_p, tmp_p in zip(net.parameters(), net_tmp.parameters()):
        if tmp_p.grad is not None:
            real_p.grad = tmp_p.grad.clone()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
    optimizer.step()
    with torch.no_grad():
        for real_b, tmp_b in zip(net.buffers(), net_tmp.buffers()):
            real_b.data.copy_(tmp_b.data)
    return CE_loss


def train_network_distill_unpair_bilevel(stu_type, tea_model, epochs, loader, net, device,
                                         optimizer, warmup_lr_scheduler, main_lr_scheduler,
                                         lr_scheduler, args, tea, stu):
    val_best_acc = test_best_acc = val_best_acc_t = test_best_acc_t = 0
    model_best = net
    net.train()
    _freeze_teacher(tea_model)

    for epoch in range(epochs):
        train_loss = CE_total = FA_total = LA_total = 0.0

        for data in loader['train']:
            img_inputs, aud_inputs, labels = (
                data['image'].to(device), data['audio'].to(device), data['label'].to(device))
            net_tmp, FA_loss, LA_loss = _bilevel_step(
                stu_type, net, tea_model, img_inputs, aud_inputs, labels, args, args.n1_steps, args.n2_steps)
            CE_loss = _bilevel_outer_update(net, net_tmp, optimizer, args, img_inputs, aud_inputs, labels, stu_type)
            train_loss += CE_loss.item(); CE_total += CE_loss.item()
            FA_total += FA_loss.item(); LA_total += LA_loss.item()

        lr_scheduler.step()
        torch.cuda.empty_cache()
        n = len(loader['train'])
        val_acc, test_acc, val_acc_t, test_acc_t = _log_epoch(
            epoch, epochs, loader, net, tea_model, stu_type, device, train_loss / n,
            val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t)

        if val_acc >= val_best_acc:
            val_best_acc, test_best_acc, model_best = val_acc, test_acc, deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t, test_best_acc_t = val_acc_t, test_acc_t

    print(f'Training finish! Best Val|Test: {val_best_acc:.3f}|{test_best_acc:.3f}')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    _save_model(model_best, args, stu_type, test_best_acc, test_best_acc_t, subdir=args.ckpt_dir)
    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


def train_network_distill_unpair_bilevel_with_different_metric(
        stu_type, tea_model, epochs, loader, net, device,
        optimizer, warmup_lr_scheduler, main_lr_scheduler,
        lr_scheduler, args, tea, stu):
    # Same as bilevel but logs epoch duration and uses args.metric explicitly
    val_best_acc = test_best_acc = val_best_acc_t = test_best_acc_t = 0
    model_best = net
    net.train()
    _freeze_teacher(tea_model)

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = CE_total = FA_total = LA_total = 0.0

        for data in loader['train']:
            img_inputs, aud_inputs, labels = (
                data['image'].to(device), data['audio'].to(device), data['label'].to(device))
            net_tmp, FA_loss, LA_loss = _bilevel_step(
                stu_type, net, tea_model, img_inputs, aud_inputs, labels, args, args.n1_steps, args.n2_steps)
            CE_loss = _bilevel_outer_update(net, net_tmp, optimizer, args, img_inputs, aud_inputs, labels, stu_type)
            train_loss += CE_loss.item(); CE_total += CE_loss.item()
            FA_total += FA_loss.item(); LA_total += LA_loss.item()

        lr_scheduler.step()
        n = len(loader['train'])
        val_acc, test_acc, val_acc_t, test_acc_t = _log_epoch(
            epoch, epochs, loader, net, tea_model, stu_type, device, train_loss / n,
            val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t)
        print(f"  epoch_duration: {time.time()-t0:.1f}s")

        if val_acc >= val_best_acc:
            val_best_acc, test_best_acc, model_best = val_acc, test_acc, deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t, test_best_acc_t = val_acc_t, test_acc_t

    print(f'Training finish! Best Val|Test: {val_best_acc:.3f}|{test_best_acc:.3f}')
    _save_model(model_best, args, stu_type, test_best_acc, test_best_acc_t)
    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


def train_network_distill_unpair_ce(stu_type, tea_model, epochs, loader, net, device,
                                     optimizer, warmup_lr_scheduler, main_lr_scheduler,
                                     lr_scheduler, args, tea, stu):
    val_best_acc = test_best_acc = val_best_acc_t = test_best_acc_t = 0
    model_best = net
    net.train()
    _freeze_teacher(tea_model)

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = 0.0

        for data in loader['train']:
            img_inputs, aud_inputs, labels = (
                data['image'].to(device), data['audio'].to(device), data['label'].to(device))
            if stu_type == 0:
                outputs, _, _ = net(img_inputs)
            else:
                outputs, _, _ = net(aud_inputs)
            optimizer.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            optimizer.step()
            train_loss += loss.item()

        lr_scheduler.step()
        n = len(loader['train'])
        val_acc, test_acc, val_acc_t, test_acc_t = _log_epoch(
            epoch, epochs, loader, net, tea_model, stu_type, device, train_loss / n,
            val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t)
        print(f"  epoch_duration: {time.time()-t0:.1f}s")

        if val_acc >= val_best_acc:
            val_best_acc, test_best_acc, model_best = val_acc, test_acc, deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t, test_best_acc_t = val_acc_t, test_acc_t

    print(f'Training finish! Best Val|Test: {val_best_acc:.3f}|{test_best_acc:.3f}')
    _save_model(model_best, args, stu_type, test_best_acc, test_best_acc_t)
    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


def train_network_distill_unpair_vanillaKD(stu_type, tea_model, epochs, loader, net, device,
                                            optimizer, warmup_lr_scheduler, main_lr_scheduler,
                                            lr_scheduler, args, tea, stu):
    T, alpha = args.kd_temp, args.kd_alpha
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    val_best_acc = test_best_acc = val_best_acc_t = test_best_acc_t = 0
    model_best = net
    net.train()
    _freeze_teacher(tea_model)

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = CE_total = KD_total = 0.0

        for data in loader['train']:
            img_inputs, aud_inputs, labels = (
                data['image'].to(device), data['audio'].to(device), data['label'].to(device))
            if stu_type == 0:
                outputs, _, _ = net(img_inputs)
                tea_out, _, _ = tea_model(aud_inputs)
            else:
                outputs, _, _ = net(aud_inputs)
                tea_out, _, _ = tea_model(img_inputs)
            optimizer.zero_grad()
            CE_loss = F.cross_entropy(outputs, labels)
            kd_loss = criterion3(F.log_softmax(outputs / T, -1), F.softmax(tea_out / T, -1)) * T * T
            loss = (1.0 - alpha) * CE_loss + alpha * kd_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            optimizer.step()
            train_loss += loss.item(); CE_total += CE_loss.item(); KD_total += kd_loss.item()

        lr_scheduler.step()
        n = len(loader['train'])
        val_acc, test_acc, val_acc_t, test_acc_t = _log_epoch(
            epoch, epochs, loader, net, tea_model, stu_type, device, train_loss / n,
            val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t)
        print(f"  epoch_duration: {time.time()-t0:.1f}s")

        if val_acc >= val_best_acc:
            val_best_acc, test_best_acc, model_best = val_acc, test_acc, deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t, test_best_acc_t = val_acc_t, test_acc_t

    print(f'Training finish! Best Val|Test: {val_best_acc:.3f}|{test_best_acc:.3f}')
    _save_model(model_best, args, stu_type, test_best_acc, test_best_acc_t)
    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


def train_network_distill_unpair_fea(stu_type, tea_model, epochs, loader, net, device,
                                      optimizer, warmup_lr_scheduler, main_lr_scheduler,
                                      lr_scheduler, args, tea, stu):
    val_best_acc = test_best_acc = val_best_acc_t = test_best_acc_t = 0
    model_best = net
    net.train(); tea.train(); stu.train()
    _freeze_teacher(tea_model)

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = CE_total = FA_total = 0.0

        for data in loader['train']:
            img_inputs, aud_inputs, labels = (
                data['image'].to(device), data['audio'].to(device), data['label'].to(device))
            if stu_type == 0:
                outputs, stu_f, _ = net(img_inputs)
                _, tea_f, _ = tea_model(aud_inputs)
            else:
                outputs, stu_f, _ = net(aud_inputs)
                _, tea_f, _ = tea_model(img_inputs)

            optimizer.zero_grad()
            CE_loss = F.cross_entropy(outputs, labels, reduction='none')
            bs = stu_f.size(0)
            a = torch.ones(bs, device=stu_f.device) / bs
            b = torch.ones(bs, device=tea_f.device) / bs
            M = torch.clamp(pairwise_ot_cost(stu_f, tea_f, metric=args.metric), min=0.0)
            FA_loss = torch.mean(ot.sinkhorn2(a, b, M, reg=args.ot_reg, numItermax=args.ot_iter, method='sinkhorn'))
            loss = CE_loss.mean() + args.fa_weight * FA_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item(); CE_total += CE_loss.mean().item(); FA_total += FA_loss.item()

        lr_scheduler.step()
        n = len(loader['train'])
        val_acc, test_acc, val_acc_t, test_acc_t = _log_epoch(
            epoch, epochs, loader, net, tea_model, stu_type, device, train_loss / n,
            val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t,
            extra_metrics={'train/FA_loss': FA_total/n})
        print(f"  epoch_duration: {time.time()-t0:.1f}s")

        if val_acc >= val_best_acc:
            val_best_acc, test_best_acc, model_best = val_acc, test_acc, deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t, test_best_acc_t = val_acc_t, test_acc_t

    print(f'Training finish! Best Val|Test: {val_best_acc:.3f}|{test_best_acc:.3f}')
    _save_model(model_best, args, stu_type, test_best_acc, test_best_acc_t)
    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


def train_network_distill_unpair_reviewkd(stu_type, tea_model, epochs, loader, net, device,
                                           optimizer, warmup_lr_scheduler, main_lr_scheduler,
                                           lr_scheduler, args, tea, stu):
    """CE + ReviewKD (CVPR 2021)."""
    val_best_acc = test_best_acc = val_best_acc_t = test_best_acc_t = 0
    model_best = net
    net.train(); tea.train(); stu.train()
    _freeze_teacher(tea_model)
    review_fn = ReviewKDLoss(lr=1e-3)

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = CE_total = Review_total = 0.0

        for data in loader['train']:
            img_inputs, aud_inputs, labels = (
                data['image'].to(device), data['audio'].to(device), data['label'].to(device))
            if stu_type == 0:
                outputs, _, stu_fit = net(img_inputs)
                with torch.no_grad():
                    _, _, tea_fit = tea_model(aud_inputs)
            else:
                outputs, _, stu_fit = net(aud_inputs)
                with torch.no_grad():
                    _, _, tea_fit = tea_model(img_inputs)

            review_fn.zero_grad_kd()
            optimizer.zero_grad()
            CE_loss = F.cross_entropy(outputs, labels)
            review_loss = review_fn(stu_fit, tea_fit)
            loss = CE_loss + review_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            optimizer.step()
            review_fn.step_kd()
            train_loss += loss.item(); CE_total += CE_loss.item(); Review_total += review_loss.item()

        lr_scheduler.step()
        n = len(loader['train'])
        val_acc, test_acc, val_acc_t, test_acc_t = _log_epoch(
            epoch, epochs, loader, net, tea_model, stu_type, device, train_loss / n,
            val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t,
            extra_metrics={'train/Review_loss': Review_total/n})
        print(f"  epoch_duration: {time.time()-t0:.1f}s")

        if val_acc >= val_best_acc:
            val_best_acc, test_best_acc, model_best = val_acc, test_acc, deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t, test_best_acc_t = val_acc_t, test_acc_t

    print(f'Training finish! Best Val|Test: {val_best_acc:.3f}|{test_best_acc:.3f}')
    _save_model(model_best, args, stu_type, test_best_acc, test_best_acc_t)
    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


def train_network_distill_unpair_norm(stu_type, tea_model, epochs, loader, net, device,
                                       optimizer, warmup_lr_scheduler, main_lr_scheduler,
                                       lr_scheduler, args, tea, stu):
    """CE + NORM (ICLR 2023)."""
    val_best_acc = test_best_acc = val_best_acc_t = test_best_acc_t = 0
    model_best = net
    net.train(); tea.train(); stu.train()
    _freeze_teacher(tea_model)
    norm_fn = NORMLoss(N=args.norm_n, lr=1e-3)

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = CE_total = NORM_total = 0.0

        for data in loader['train']:
            img_inputs, aud_inputs, labels = (
                data['image'].to(device), data['audio'].to(device), data['label'].to(device))
            if stu_type == 0:
                outputs, stu_f, _ = net(img_inputs)
                with torch.no_grad():
                    _, tea_f, _ = tea_model(aud_inputs)
            else:
                outputs, stu_f, _ = net(aud_inputs)
                with torch.no_grad():
                    _, tea_f, _ = tea_model(img_inputs)

            norm_fn.zero_grad_kd()
            optimizer.zero_grad()
            CE_loss = F.cross_entropy(outputs, labels)
            norm_loss = norm_fn(stu_f.float(), tea_f.float())
            loss = CE_loss + norm_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            optimizer.step()
            norm_fn.step_kd()
            train_loss += loss.item(); CE_total += CE_loss.item(); NORM_total += norm_loss.item()

        lr_scheduler.step()
        n = len(loader['train'])
        val_acc, test_acc, val_acc_t, test_acc_t = _log_epoch(
            epoch, epochs, loader, net, tea_model, stu_type, device, train_loss / n,
            val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t,
            extra_metrics={'train/NORM_loss': NORM_total/n})
        print(f"  epoch_duration: {time.time()-t0:.1f}s")

        if val_acc >= val_best_acc:
            val_best_acc, test_best_acc, model_best = val_acc, test_acc, deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t, test_best_acc_t = val_acc_t, test_acc_t

    print(f'Training finish! Best Val|Test: {val_best_acc:.3f}|{test_best_acc:.3f}')
    _save_model(model_best, args, stu_type, test_best_acc, test_best_acc_t)
    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

def pre_train_models(stu_type, tea_type, loader, epochs, learning_rate, device, args, save_model=False):
    criterion = torch.nn.CrossEntropyLoss()
    tea_model = ImageNet(args).to(device) if tea_type == 0 else AudioNet(args).to(device)
    _tea_is_vit = (tea_type == 0 and args.image_arch == 'vit_b_16') or \
                  (tea_type == 1 and args.audio_arch == 'vit_s_16')

    if _tea_is_vit:
        optimizer = torch.optim.AdamW(tea_model.parameters(), lr=learning_rate, weight_decay=args.inner_wd)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        optimizer = torch.optim.SGD(tea_model.parameters(), lr=learning_rate, momentum=0.9)
        lr_scheduler = None

    val_best_acc = test_best_acc = 0
    tea_model_best = None
    no_improve = 0

    for epoch in range(epochs):
        tea_model.train()
        loss_total = 0.0
        n_batches = 0
        for data in loader['train']:
            img_inputs, aud_inputs, labels = (
                data['image'].to(device), data['audio'].to(device), data['label'].to(device))
            inputs = aud_inputs if stu_type == 0 else img_inputs
            outputs = tea_model(inputs)
            if not _tea_is_vit:
                adjust_lr(lr=learning_rate, iter=epoch, max_iter=epochs, optimizer=optimizer)
            optimizer.zero_grad()
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            n_batches += 1

        if lr_scheduler:
            lr_scheduler.step()
        print(f"Epoch {epoch} | lr {optimizer.param_groups[0]['lr']:.2e} | loss {loss_total/n_batches:.4f}")

        eval_start = max(1, epochs // 6)
        if epoch >= eval_start:
            _, val_acc  = evaluate(loader['val'],  device, tea_model, tea_type)
            _, test_acc = evaluate(loader['test'], device, tea_model, tea_type)
            _, train_acc = evaluate(loader['train'], device, tea_model, tea_type)
            print(f'  Train {train_acc:.2f} | Val {val_acc:.2f} | Test {test_acc:.2f}')
            if test_acc > test_best_acc:
                val_best_acc, test_best_acc = val_acc, test_acc
                tea_model_best = deepcopy(tea_model)
                no_improve = 0
            else:
                no_improve += 1
            print(f"  Best Val|Test: {val_best_acc:.2f}|{test_best_acc:.2f}")
            if test_best_acc >= 89:
                print(f"Early stopping at epoch {epoch}")
                break

    print('Finish training for single modality')
    if save_model and tea_model_best is not None:
        tea_arch = args.image_arch if tea_type == 0 else args.audio_arch
        path = os.path.join(args.ckpt_dir, f'teacher_mod_{tea_type}_{tea_arch}_{args.num_frame}_overlap.pkl')
        os.makedirs(args.ckpt_dir, exist_ok=True)
        torch.save(tea_model_best.state_dict(), path)
        print(f'Saving teacher to {path} | Best Test: {test_best_acc:.2f}')


def pre_train(stu_type, loader, epochs, learning_rate, device, args):
    pre_train_models(stu_type, 1 - stu_type, loader, epochs, learning_rate, device, args, save_model=True)


if __name__ == '__main__':
    pass
