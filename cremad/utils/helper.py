import random
import numpy as np
import torch
import os
from copy import deepcopy
from utils.model_res import ImageNet, AudioNet
import time
import torch.nn.functional as F
from utils.CremadDataset import CremadDataset
from torch.utils.data import DataLoader
from scipy.stats.stats import kendalltau
import torch.nn as nn
import wandb
import ot
import sys as _sys, os as _os
_ucmkd_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ucmkd_root not in _sys.path:
    _sys.path.insert(0, _ucmkd_root)
from kd_losses import ReviewKDLoss, NORMLoss



def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def adjust_lr(lr=1e-2, iter=None, max_iter=100, power=0.9, optimizer=None):
    cur_lr = 1e-4 + (lr - 1e-4)*((1-float(iter)/max_iter)**(power))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def evaluate(loader, device, net, in_type):
    # type == 0: image input; type == 1: audio input; type == 2: both input
    correct, v_loss, total = 0, 0, 0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            img_inputs, aud_inputs, labels = data
            img_inputs, aud_inputs, labels = img_inputs.to(device), aud_inputs.to(device), labels.to(device)
            total += labels.size(0)
            if in_type == 0:
                outputs = net(img_inputs)
            elif in_type == 1:
                outputs = net(aud_inputs)
            elif in_type == 2:
                outputs = net(img_inputs, aud_inputs)
            else:
                raise ValueError('the value of element in in_type_list should be 0,1,2\n')

            _, predicted = torch.max(outputs[0].detach(), 1)
            correct += (predicted == labels).sum().item()
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs[0], labels)  # fix to CE loss
            v_loss += loss.item()
    v_loss = v_loss / len(loader)
    val_acc = 100 * correct / total
    return v_loss, val_acc


def evaluate_allacc(loader, device, net, in_type):
    _, train_acc = evaluate(loader['train'], device, net, in_type)
    _, val_acc = evaluate(loader['val'], device, net, in_type)
    _, test_acc = evaluate(loader['test'], device, net, in_type)
    return train_acc, val_acc, test_acc


def gen_data(data_dir, batch_size, num_workers, args):
    train_dataset = CremadDataset(mode='train', data_root=args.data_root)
    test_dataset  = CremadDataset(mode='test',  data_root=args.data_root)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return {'train': train_dataloader, 'val': test_dataloader, 'test': test_dataloader}

def ntkl(logits_student, logits_teacher, target, mask=None, criterion4=None, temperature=1):

    gt_mask = _get_gt_mask(logits_student, target)
    logits_teacher = logits_teacher * (~gt_mask)
    pred_teacher_part2 = F.softmax(logits_teacher / temperature, dim=1)
    logits_student = logits_student * (~gt_mask)
    log_pred_student_part2 = F.log_softmax(logits_student / temperature, dim=1)
    if mask.sum() == 0:
        temp = torch.tensor(0)
    else:
        temp = ((mask * (criterion4(log_pred_student_part2, pred_teacher_part2.detach()).sum(1)))).mean()
    return temp


def pairwise_ot_cost(stu_f, tea_f, metric="cosine", L=None, eps=1e-6):
    if metric == "l2":
        M = torch.cdist(stu_f, tea_f, p=2)
    elif metric == "l1":
        M = torch.cdist(stu_f, tea_f, p=1)
    elif metric == "chordal":
        stu = F.normalize(stu_f, p=2, dim=1)
        tea = F.normalize(tea_f, p=2, dim=1)
        M = torch.cdist(stu, tea, p=2)
    elif metric == "cosine":
        stu = F.normalize(stu_f, p=2, dim=1)
        tea = F.normalize(tea_f, p=2, dim=1)
        M = 1.0 - stu @ tea.T
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return M


def train_network_distill_c2kd(stu_type, tea_model, epochs, loader, net, device, optimizer, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion = torch.nn.CrossEntropyLoss()
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0

    for epoch in range(epochs):
        net.train()
        tea.train()
        stu.train()
        tea_model.train()
        # for name, param in tea_model.named_parameters():
        #     if 'layer4' not in name and 'fc' not in name:
        #     # if 'fc' not in name:
        #         param.requires_grad = False
        train_loss = 0.0
        corr_mean = 0.0
        corr_num = 0.0

        loss1, loss2, loss_s2t, loss_t2s, loss_ang = 0, 0, 0, 0, 0
        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            else:
                raise ValueError("Undefined training type in distilled training")


            optimizer.zero_grad()
            XE_s = F.cross_entropy(outputs, labels, reduction='none')
            XE_t = F.cross_entropy(pseu_label, labels, reduction='none')

            kl_loss_t_t_im = criterion3(F.log_softmax(tea_logits, -1), F.softmax(pseu_label.detach(), dim=-1))
            kl_loss_t_im_t = criterion3(F.log_softmax(pseu_label, -1), F.softmax(tea_logits.detach(), dim=-1))


            kl_loss_s_s_im = criterion3(F.log_softmax(stu_logits, -1), F.softmax(outputs.detach(), dim=-1))
            kl_loss_s_im_s = criterion3(F.log_softmax(outputs, -1), F.softmax(stu_logits.detach(), dim=-1))

            kendall = np.empty(tea_logits.size(0))
            for i in range(tea_logits.size(0)):
                temp = \
                kendalltau(tea_logits[i, :].cpu().detach().numpy(), stu_logits[i, :].cpu().detach().numpy())[0]
                kendall[i] = temp
            kendall = torch.from_numpy(kendall).cuda()
            kendall_mean = kendall.mean()
            bar = sorted(kendall)[int(kendall.size(0) * 0.5) - 1]
            mask = torch.where(kendall > args.krc, 1, 0)
            # mask2 = torch.where(kendall > 0.2, 1, 0)
            mask_num = mask.sum()
            del kendall


            if mask.sum()==0:
                kl_loss_st_s_t = 0
                kl_loss_st_t_s = 0
            else:
                kl_loss_st_s_t = ntkl(stu_logits, tea_logits.detach(), labels, mask, criterion4)
                kl_loss_st_t_s = ntkl(tea_logits, stu_logits.detach(), labels, mask, criterion4)

            tmp1 = XE_t.mean() + kl_loss_t_t_im + kl_loss_st_t_s + kl_loss_t_im_t
            tmp2 = XE_s.mean() + kl_loss_s_s_im + kl_loss_st_s_t + kl_loss_s_im_s
            loss = tmp1 + tmp2
            loss.backward()
            optimizer.step()
            lr = adjust_lr(iter=epoch, optimizer=optimizer)
            train_loss += loss.item()
            loss1 += tmp1.item()
            loss2 += tmp2.item()
            corr_mean += kendall_mean.item()
            corr_num += mask_num.item()

        if epoch >= 80:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({
            'train/loss': train_loss / len(loader['train']),
            'train/loss1': loss1 / len(loader['train']),
            'train/loss2': loss2 / len(loader['train']),
            'train/corr_mean': corr_mean / len(loader['train']),
            'student/train_acc': train_acc,
            'student/val_acc': val_acc,
            'student/test_acc': test_acc,
            'teacher/train_acc': train_acc_t,
            'teacher/val_acc': val_acc_t,
            'teacher/test_acc': test_acc_t,
            'student/best_val_acc': val_best_acc,
            'student/best_test_acc': test_best_acc,
            'teacher/best_val_acc': val_best_acc_t,
            'teacher/best_test_acc': test_best_acc_t,
        }, step=epoch)
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(f"corr {corr_mean / len(loader['train']):.3f}| num: {corr_num / len(loader['train']):.3f}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f} | temp1 Loss {loss1 / len(loader['train']):.3f} | temp2 Loss {loss2 / len(loader['train']):.3f} | PL_pseu_label Loss {loss_s2t / len(loader['train']):.3f} | PL_output Loss {loss_t2s / len(loader['train']):.3f}| cosine Loss {loss_ang / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        model_path = os.path.join(args.ckpt_dir, 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t

#real unpair
def train_network_distill_unpair_sumall(stu_type, tea_model, epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    tea.train()
    stu.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False


    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            # output and net == student
            if stu_type == 0:
                # outputs = self.fc(outputs_128)
                # outputs: [b, num_class], [b, 128]
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)
                labels = labels
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                labels = labels
            else:
                raise ValueError("Undefined training type in distilled training")

            optimizer.zero_grad()

            stu_f = outputs_128
            tea_f = pseu_label_128

            # print(stu_f.shape)
            CE_loss = F.cross_entropy(outputs, labels, reduction='none')

            # Compute FA loss
            
            batch_size = stu_f.size(0)
            # normalize by 1/N
            a = torch.ones(batch_size, device = stu_f.device) / batch_size
            b = torch.ones(batch_size, device = tea_f.device) / batch_size

            M = pairwise_ot_cost(stu_f, tea_f, metric=args.metric)
            M = torch.clamp(M, min=0.0)
            # print("Requires Grad:", stu_f_tmp.requires_grad)
            # print("Grad Fn:", stu_f_tmp.grad_fn)
            FA_loss = ot.sinkhorn2(a, b, M, reg=args.ot_reg, numItermax=args.ot_iter, method='sinkhorn')
            FA_loss = torch.mean(FA_loss)
            

            # Tinh LA?
            stu_latent_2_tea = tea_model.fc(stu_f)
            log_probs_s = F.log_softmax(outputs, dim=-1) #log(p_S)
            log_probs_t = F.log_softmax(stu_latent_2_tea, dim=-1) #log(p_T)

            probs_t = F.softmax(stu_latent_2_tea, dim=-1)          # p_T (used as kappa)
            kappa = probs_t.detach() # boi vi khi ma nhan khong dung thi nhan vao expect la 0 roi nen co the coi la D^T(y|z) cho de tinh
            log_ratio_matrix = log_probs_s - kappa * log_probs_t  # log( p_S / p_T^kappa ) = log(p_S) - kappa * log(p_T)
            LA_loss = F.nll_loss(log_ratio_matrix, labels) # label = 0 if wrong class, 1 if correct
    


            loss = CE_loss.mean() + args.fa_weight * FA_loss + args.la_weight * LA_loss
            # print(loss.item(), CE_loss.mean(), FA_loss, LA_loss)
            # loss = CE_loss.mean() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            optimizer.step()
            train_loss += loss.item()
            CE_loss_total += CE_loss.mean().item()
            FA_loss_total += FA_loss.item()
            LA_loss_total += LA_loss.item()

        lr_scheduler.step()

        if epoch >= 0:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({
            'train/loss': train_loss / len(loader['train']),
            'train/CE_loss': CE_loss_total / len(loader['train']),
            'train/FA_loss': FA_loss_total / len(loader['train']),
            'train/LA_loss': LA_loss_total / len(loader['train']),
            'student/train_acc': train_acc,
            'student/val_acc': val_acc,
            'student/test_acc': test_acc,
            'teacher/train_acc': train_acc_t,
            'teacher/val_acc': val_acc_t,
            'teacher/test_acc': test_acc_t,
            'student/best_val_acc': val_best_acc,
            'student/best_test_acc': test_best_acc,
            'teacher/best_val_acc': val_best_acc_t,
            'teacher/best_test_acc': test_best_acc_t,
        }, step=epoch)
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t

def train_network_distill_unpair_bilevel(stu_type, tea_model, epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False
    
    # hyperparameter
    n1_steps = args.n1_steps
    n2_steps = args.n2_steps

    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            # copy student network
            net_tmp = deepcopy(net)
            net_tmp.train()

            optimizer_tmp = torch.optim.AdamW(list(net_tmp.parameters()), lr=args.inner_lr, weight_decay=args.inner_wd)

            for _ in range(n1_steps):
                optimizer_tmp.zero_grad()
                if stu_type == 0:
                    # outputs = self.fc(outputs_128)
                    # outputs: [b, num_class], [b, 128]
                    outputs, stu_f_tmp, stu_fit = net_tmp(img_inputs_cln)
                    pseu_label, tea_f, tea_fit = tea_model(aud_inputs_cln)
                    
                elif stu_type == 1:
                    outputs, stu_f_tmp, stu_fit = net_tmp(aud_inputs_cln)
                    pseu_label, tea_f, tea_fit = tea_model(img_inputs_cln)
                
                # Compute FA loss
                # Loi khong giam duoc FA do gap qua lon, cai FA_loss luon la 220-250 -> normalize
                # Vi wasserstein dung de sap xep
                
                batch_size = stu_f_tmp.size(0)
                a = torch.ones(batch_size, device=stu_f_tmp.device) / batch_size
                b = torch.ones(batch_size, device=tea_f.device) / batch_size
                
                

                M = pairwise_ot_cost(stu_f_tmp, tea_f, metric=args.metric)
                M = torch.clamp(M, min=0.0)

                FA_loss = ot.sinkhorn2(a, b, M, reg=args.ot_reg, numItermax=args.ot_iter)
                FA_loss = torch.mean(FA_loss)
                FA_loss.backward()

                optimizer_tmp.step()

            for _ in range(n2_steps):
                optimizer_tmp.zero_grad()

                # forward lai net_tmp sau khi cap nhat
                if stu_type == 0:
                    stu_f_tmp, _ = net_tmp.forward_encoder(img_inputs_cln)
                    stu_f_fixed = stu_f_tmp.detach() # detach gradient from tmp
                    outputs_tmp = net_tmp.forward_head(stu_f_fixed)
                elif stu_type == 1:
                    stu_f_tmp, _ = net_tmp.forward_encoder(aud_inputs_cln)
                    stu_f_fixed = stu_f_tmp.detach() # detach gradient from tmp
                    outputs_tmp = net_tmp.forward_head(stu_f_fixed)
                
                # Compute LA loss
                with torch.no_grad():
                    stu_latent_2_tea = tea_model.fc(stu_f_tmp) # Teacher view on Student z
                
                Cross_Entropy = torch.nn.CrossEntropyLoss(reduction='none')
                probs_s = F.softmax(outputs_tmp, dim=-1)
                probs_t = F.softmax(stu_latent_2_tea, dim=-1)
            
                #select target class prob as kappa 
                kappa = probs_t.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
                
                LA_loss = (Cross_Entropy(outputs_tmp,labels) - kappa*Cross_Entropy(stu_latent_2_tea,labels)).mean()

                # Second inner update for net_tmp
                LA_loss.backward()
                optimizer_tmp.step()
                    
            # update net va lay cai output cuoi cua temp
            if stu_type == 0:   
                # outputs = self.fc(outputs_128)
                # outputs: [b, num_class], [b, 128]
                outputs_final_tmp, outputs_128, stu_fit = net_tmp(img_inputs_cln)
                # outputs_final_tmp, outputs_128, stu_fit = net(img_inputs_cln)
            elif stu_type == 1:
                outputs_final_tmp, outputs_128, stu_fit = net_tmp(aud_inputs_cln)
                # outputs_final_tmp, outputs_128, stu_fit = net(aud_inputs_cln)
            else:
                raise ValueError("Undefined training type in distilled training")
            
            # optimizer.zero_grad()
            CE_loss = F.cross_entropy(outputs_final_tmp, labels)
            
            # optimizer_tmp.zero_grad()
            net_tmp.zero_grad()
            CE_loss.backward() # tinh gradient cong vao net_tmp.parameters().grad

            optimizer.zero_grad() # xoa gradient cu
            # Chep gradient tu mang tmp vao real, chi co moi celoss thoi
            for real_param, tmp_param in zip(net.parameters(), net_tmp.parameters()):
                if tmp_param.grad is not None:
                    real_param.grad = tmp_param.grad.clone()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            optimizer.step()

            # sao chep cac tham so thong ke, running mean/ running var
            with torch.no_grad():
                for real_buffer, tmp_buffer in zip(net.buffers(), net_tmp.buffers()):
                    real_buffer.data.copy_(tmp_buffer.data)

            # loss = CE_loss.mean().item() + FA_loss.item() + LA_loss.item()
            # loss = CE_loss.mean().item()
            # print(loss.item(), CE_loss.mean(), FA_loss, LA_loss)
            # loss.backward()
            # optimizer.step()
            train_loss += CE_loss.item()
            CE_loss_total += CE_loss.item()
            FA_loss_total += FA_loss.item()
            LA_loss_total += LA_loss.item()

        lr_scheduler.step()

        if epoch >= 0:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({
            'train/loss': train_loss / len(loader['train']),
            'train/CE_loss': CE_loss_total / len(loader['train']),
            'train/FA_loss': FA_loss_total / len(loader['train']),
            'train/LA_loss': LA_loss_total / len(loader['train']),
            'student/train_acc': train_acc,
            'student/val_acc': val_acc,
            'student/test_acc': test_acc,
            'teacher/train_acc': train_acc_t,
            'teacher/val_acc': val_acc_t,
            'teacher/test_acc': test_acc_t,
            'student/best_val_acc': val_best_acc,
            'student/best_test_acc': test_best_acc,
            'teacher/best_val_acc': val_best_acc_t,
            'teacher/best_test_acc': test_best_acc_t,
        }, step=epoch)
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


def train_network_distill_unpair_ce(stu_type, tea_model, epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False
    
    # hyperparameter
    # n1_steps = 1
    # n2_steps = 1

    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)         
                    
            if stu_type == 0:   
                outputs_final, outputs_128, stu_fit = net(img_inputs_cln)
            elif stu_type == 1:
                outputs_final, outputs_128, stu_fit = net(aud_inputs_cln)
            else:
                raise ValueError("Undefined training type in distilled training")
            
            optimizer.zero_grad()
            CE_loss = F.cross_entropy(outputs_final, labels)
            

            loss = CE_loss
            # print(loss.item(), CE_loss.mean(), FA_loss, LA_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            optimizer.step()
            train_loss += loss.item()
            CE_loss_total += CE_loss.item()
            # FA_loss_total += FA_loss.item()
            # LA_loss_total += LA_loss.item()

        lr_scheduler.step()

        if epoch >= 0:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({
            'train/loss': train_loss / len(loader['train']),
            'train/CE_loss': CE_loss_total / len(loader['train']),
            'train/FA_loss': FA_loss_total / len(loader['train']),
            'train/LA_loss': LA_loss_total / len(loader['train']),
            'student/train_acc': train_acc,
            'student/val_acc': val_acc,
            'student/test_acc': test_acc,
            'teacher/train_acc': train_acc_t,
            'teacher/val_acc': val_acc_t,
            'teacher/test_acc': test_acc_t,
            'student/best_val_acc': val_best_acc,
            'student/best_test_acc': test_best_acc,
            'teacher/best_val_acc': val_best_acc_t,
            'teacher/best_test_acc': test_best_acc_t,
        }, step=epoch)
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t

def train_network_distill_unpair_vanillaKD(stu_type, tea_model, epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu):
    # Config KD
    T = args.kd_temp
    alpha = args.kd_alpha

    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False
    

    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0
        KD_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)         
                    
            if stu_type == 0:   
                outputs_final, outputs_128, stu_fit = net(img_inputs_cln)
                tea_final, tea_128, tea_fit = tea_model(aud_inputs_cln)
            elif stu_type == 1:
                outputs_final, outputs_128, stu_fit = net(aud_inputs_cln)
                tea_final, tea_128, tea_fit = tea_model(img_inputs_cln)
            else:
                raise ValueError("Undefined training type in distilled training")
            
            optimizer.zero_grad()
            CE_loss = F.cross_entropy(outputs_final, labels)
            
            soft_student = F.log_softmax(outputs_final / T, dim=-1)
            soft_teacher = F.softmax(tea_final / T, dim=-1)
            kd_loss = criterion3(soft_student, soft_teacher) * (T * T)
            

            loss = (1.0 - alpha) * CE_loss + alpha * kd_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            optimizer.step()
            train_loss += loss.item()
            CE_loss_total += CE_loss.item()
            KD_loss_total += kd_loss.item()

        lr_scheduler.step()

        if epoch >= 0:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({
            'train/loss': train_loss / len(loader['train']),
            'train/CE_loss': CE_loss_total / len(loader['train']),
            'train/FA_loss': FA_loss_total / len(loader['train']),
            'train/LA_loss': LA_loss_total / len(loader['train']),
            'student/train_acc': train_acc,
            'student/val_acc': val_acc,
            'student/test_acc': test_acc,
            'teacher/train_acc': train_acc_t,
            'teacher/val_acc': val_acc_t,
            'teacher/test_acc': test_acc_t,
            'student/best_val_acc': val_best_acc,
            'student/best_test_acc': test_best_acc,
            'teacher/best_val_acc': val_best_acc_t,
            'teacher/best_test_acc': test_best_acc_t,
        }, step=epoch)
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t

def train_network_distill_unpair_fea(stu_type, tea_model, epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    tea.train()
    stu.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False


    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            # output and net == student
            if stu_type == 0:
                # outputs = self.fc(outputs_128)
                # outputs: [b, num_class], [b, 128]
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)
                labels = labels
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                labels = labels
            else:
                raise ValueError("Undefined training type in distilled training")

            optimizer.zero_grad()

            stu_f = outputs_128
            tea_f = pseu_label_128

            # print(stu_f.shape)
            CE_loss = F.cross_entropy(outputs, labels, reduction='none')

            # Compute FA loss
            
            batch_size = stu_f.size(0)
            # normalize by 1/N
            a = torch.ones(batch_size, device = stu_f.device) / batch_size
            b = torch.ones(batch_size, device = tea_f.device) / batch_size

            M = pairwise_ot_cost(stu_f, tea_f, metric=args.metric)
            M = torch.clamp(M, min=0.0)
            # print("Requires Grad:", stu_f_tmp.requires_grad)
            # print("Grad Fn:", stu_f_tmp.grad_fn)
            FA_loss = ot.sinkhorn2(a, b, M, reg=args.ot_reg, numItermax=args.ot_iter, method='sinkhorn')
            FA_loss = torch.mean(FA_loss)
            

            # Tinh LA?
            # stu_latent_2_tea = tea_model.fc(stu_f)
            # log_probs_s = F.log_softmax(outputs, dim=-1) #log(p_S)
            # log_probs_t = F.log_softmax(stu_latent_2_tea, dim=-1) #log(p_T)

            # probs_t = F.softmax(stu_latent_2_tea, dim=-1)          # p_T (used as kappa)
            # kappa = probs_t.detach() # boi vi khi ma nhan khong dung thi nhan vao expect la 0 roi nen co the coi la D^T(y|z) cho de tinh
            # log_ratio_matrix = log_probs_s - kappa * log_probs_t  # log( p_S / p_T^kappa ) = log(p_S) - kappa * log(p_T)
            # LA_loss = F.nll_loss(log_ratio_matrix, labels) # label = 0 if wrong class, 1 if correct
    


            # loss = CE_loss.mean() + args.fa_weight * FA_loss + args.la_weight * LA_loss
            loss = CE_loss.mean() + args.fa_weight * FA_loss
            # print(loss.item(), CE_loss.mean(), FA_loss, LA_loss)
            # loss = CE_loss.mean() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            optimizer.step()
            train_loss += loss.item()
            CE_loss_total += CE_loss.mean().item()
            FA_loss_total += FA_loss.item()
            # LA_loss_total += LA_loss.item()

        lr_scheduler.step()

        if epoch >= 0:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({
            'train/loss': train_loss / len(loader['train']),
            'train/CE_loss': CE_loss_total / len(loader['train']),
            'train/FA_loss': FA_loss_total / len(loader['train']),
            'train/LA_loss': LA_loss_total / len(loader['train']),
            'student/train_acc': train_acc,
            'student/val_acc': val_acc,
            'student/test_acc': test_acc,
            'teacher/train_acc': train_acc_t,
            'teacher/val_acc': val_acc_t,
            'teacher/test_acc': test_acc_t,
            'student/best_val_acc': val_best_acc,
            'student/best_test_acc': test_best_acc,
            'teacher/best_val_acc': val_best_acc_t,
            'teacher/best_test_acc': test_best_acc_t,
        }, step=epoch)
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t

def load_visual_pretrain(model, path=None):
    if not os.path.exists(path):
        print(f"Warning: checkpoint not found at {path}, skipping pretrain load.")
        return
    
    print(f"Loading ImageNet pretrain for {model.__class__.__name__}...")
    # Load checkpoint
    state_dict = torch.load(path)
    model_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in model_dict:
            # Case 1: matching shape (most layers)
            if v.shape == model_dict[k].shape:
                new_state_dict[k] = v
            # Case 2: adapt conv1 from 3-channel ImageNet weights to N-channel input
            elif k == 'conv1.weight':
                num_frames = model_dict[k].shape[1] // 3
                # Tile weights across frames and average to avoid magnitude explosion
                new_state_dict[k] = v.repeat(1, num_frames, 1, 1) / num_frames
    
    model.load_state_dict(new_state_dict, strict=False)
    print("Done loading pretrain.")

def pre_train_models(stu_type, tea_type, loader, epochs, learning_rate, device, args, save_model=False):
    criterion = torch.nn.CrossEntropyLoss()
    tea_model = ImageNet(args).to(device) if tea_type == 0 else AudioNet(args).to(device)
    optimizer = torch.optim.SGD(list(tea_model.parameters()), lr=learning_rate, momentum=0.9)


    if stu_type == 0:
        stu_arch = args.image_arch
        tea_arch = args.audio_arch
    else:
        stu_arch = args.audio_arch
        tea_arch = args.image_arch


    _tea_is_audio_vit = (tea_type == 1 and args.audio_arch == 'vit_s_16')

    for param in tea_model.parameters():
        param.requires_grad = True

    trainable = [n for n, p in tea_model.named_parameters() if p.requires_grad]
    print(f"[tea_model] Trainable params count: {len(trainable)}")

    _use_adamw = _tea_is_audio_vit or (tea_type == 0 and args.image_arch == 'vit_b_16')
    if _use_adamw:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, tea_model.parameters()),
            lr=learning_rate, weight_decay=args.inner_wd
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, tea_model.parameters()),
            lr=learning_rate, momentum=0.9
        )
        lr_scheduler = None

    val_best_acc, test_best_acc, val_best_acc_s, test_best_acc_s, tea_model_best, stu_model_best = 0, 0, 0, 0, 0, 0
    for epoch in range(epochs):
        tea_model.train()
        loss1, loss2, loss3 = 0, 0, 0
        for i, data in enumerate(loader['train']):
            img_inputs_cln, aud_inputs_cln, labels = data
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(device)
            if stu_type == 0:
                outputs2, _, _ = tea_model(aud_inputs_cln)
            elif stu_type == 1:
                outputs2, _, _ = tea_model(img_inputs_cln)
            else:
                raise ValueError("Undefined training type")

            if not _use_adamw:
                adjust_lr(lr=learning_rate, iter=epoch, optimizer=optimizer)
            optimizer.zero_grad()
            tmp2 = criterion(outputs2, labels)
            loss = tmp2
            loss.backward()
            optimizer.step()

            loss2 += tmp2.item()

        if lr_scheduler is not None:
            lr_scheduler.step()

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} | lr {cur_lr:.2e} | loss2 {loss2 / len(loader['train']):.4f}")

        if epoch >= 0:
            _, train_acc = evaluate(loader['train'], device, tea_model, tea_type)
            # _, train_acc_s = evaluate(loader['train'], device, stu_model, stu_type)
            _, val_acc = evaluate(loader['val'], device, tea_model, tea_type)
            # _, val_acc_s = evaluate(loader['val'], device, stu_model, stu_type)
            _, test_acc = evaluate(loader['test'], device, tea_model, tea_type)
            # _, test_acc_s = evaluate(loader['test'], device, stu_model, stu_type)
            print(f'Teacher train|val|test acc {train_acc:.2f}|{val_acc:.2f}|{test_acc:.2f}')
            # print(f'Student train|val|test acc {train_acc_s:.2f}|{val_acc_s:.2f}|{test_acc_s:.2f}')
            if val_acc > val_best_acc:
                val_best_acc = val_acc
                test_best_acc = test_acc
                tea_model_best = deepcopy(tea_model)
            if test_best_acc >= 70:
                break
            # if val_acc_s > val_best_acc_s:
            #     val_best_acc_s = val_acc_s
            #     test_best_acc_s = test_acc_s
            #     stu_model_best = deepcopy(stu_model)
            print(f"Best Teacher Val|Test acc: {val_best_acc:.2f}|{test_best_acc:.2f}")
            print(f"Best Student Val|Test acc: {val_best_acc_s:.2f}|{test_best_acc_s:.2f}")
            print('-' * 60)

    print('Finish training for single modality')

    if save_model:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        model_path_t = os.path.join(args.ckpt_dir, 'teacher_mod_' + str(tea_type) + '_' + tea_arch + '_' + str(args.num_frame) + '_overlap.pkl')
        torch.save(tea_model_best.state_dict(), model_path_t)
        # model_path_s = os.path.join('./results', 'student_mod_' + str(stu_type) + '_' + stu_arch + '_'+ str(args.num_frame) + '_overlap.pkl')
        # torch.save(stu_model_best.state_dict(), model_path_s)
        print(f"Saving teacher and student model to {model_path_t}")
        print(f"Best Test acc: teacher: {test_best_acc:.2f} student: {test_best_acc_s:.2f}")


def pre_train(stu_type, loader, epochs, learning_rate, device, args):
    tea_type = 1 - stu_type
    pre_train_models(stu_type, tea_type, loader, epochs, learning_rate, device, args, save_model=True)


def train_network_distill_unpair_reviewkd(stu_type, tea_model, epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu):
    """CE + ReviewKD (CVPR 2021) — cross-level feature-map distillation."""
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    net.train()
    tea.train()
    stu.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        param.requires_grad = False

    review_fn = ReviewKDLoss(lr=1e-3)

    for epoch in range(epochs):
        train_loss = 0.0
        CE_loss_total = 0.0
        Review_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            img_inputs_cln, aud_inputs_cln, labels = data
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(device)

            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                with torch.no_grad():
                    pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                with torch.no_grad():
                    pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
            else:
                raise ValueError("Undefined training type in distilled training")

            review_fn.zero_grad_kd()
            optimizer.zero_grad()

            CE_loss = F.cross_entropy(outputs, labels)
            review_loss = review_fn(stu_fit, tea_fit)
            loss = CE_loss + review_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            optimizer.step()
            review_fn.step_kd()
            train_loss        += loss.item()
            CE_loss_total     += CE_loss.item()
            Review_loss_total += review_loss.item()

        lr_scheduler.step()

        if epoch >= 0:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({
            'train/loss': train_loss / len(loader['train']),
            'train/CE_loss': CE_loss_total / len(loader['train']),
            'train/Review_loss': Review_loss_total / len(loader['train']),
            'student/train_acc': train_acc,
            'student/val_acc': val_acc,
            'student/test_acc': test_acc,
            'teacher/train_acc': train_acc_t,
            'teacher/val_acc': val_acc_t,
            'teacher/test_acc': test_acc_t,
            'student/best_val_acc': val_best_acc,
            'student/best_test_acc': test_best_acc,
            'teacher/best_val_acc': val_best_acc_t,
            'teacher/best_test_acc': test_best_acc_t,
        }, step=epoch)
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(f"Train Loss: {train_loss / len(loader['train']):.3f} | Review {Review_loss_total / len(loader['train']):.4f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        model_path = os.path.join(args.ckpt_dir, 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


def train_network_distill_unpair_norm(stu_type, tea_model, epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu):
    """CE + NORM (ICLR 2023) — N-to-one representation matching."""
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    net.train()
    tea.train()
    stu.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        param.requires_grad = False

    norm_fn = NORMLoss(N=args.norm_n, lr=1e-3)

    for epoch in range(epochs):
        train_loss = 0.0
        CE_loss_total = 0.0
        NORM_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            img_inputs_cln, aud_inputs_cln, labels = data
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(device)

            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                with torch.no_grad():
                    pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                with torch.no_grad():
                    pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
            else:
                raise ValueError("Undefined training type in distilled training")

            norm_fn.zero_grad_kd()
            optimizer.zero_grad()

            stu_f = outputs_128.float()
            tea_f = pseu_label_128.float()

            CE_loss = F.cross_entropy(outputs, labels)
            norm_loss = norm_fn(stu_f, tea_f)
            loss = CE_loss + norm_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            optimizer.step()
            norm_fn.step_kd()
            train_loss      += loss.item()
            CE_loss_total   += CE_loss.item()
            NORM_loss_total += norm_loss.item()

        lr_scheduler.step()

        if epoch >= 0:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({
            'train/loss': train_loss / len(loader['train']),
            'train/CE_loss': CE_loss_total / len(loader['train']),
            'train/NORM_loss': NORM_loss_total / len(loader['train']),
            'student/train_acc': train_acc,
            'student/val_acc': val_acc,
            'student/test_acc': test_acc,
            'teacher/train_acc': train_acc_t,
            'teacher/val_acc': val_acc_t,
            'teacher/test_acc': test_acc_t,
            'student/best_val_acc': val_best_acc,
            'student/best_test_acc': test_best_acc,
            'teacher/best_val_acc': val_best_acc_t,
            'teacher/best_test_acc': test_best_acc_t,
        }, step=epoch)
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(f"Train Loss: {train_loss / len(loader['train']):.3f} | NORM {NORM_loss_total / len(loader['train']):.4f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        model_path = os.path.join(args.ckpt_dir, 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


if __name__ == '__main__':
    pass