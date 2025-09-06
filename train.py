import argparse
import time
import gc
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
from net import GNNStack
from utils import AverageMeter, accuracy, log_msg, get_default_train_val_test_loader
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch UEA Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='D_public')####train
#parser.add_argument('-d', '--datasets', nargs='+', default=['D_public'], help='List of datasets/domains')##test
parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')
parser.add_argument('--groups', type=int, default=4, help='the number of time series groups (num_graphs)')
parser.add_argument('--pool_ratio', type=float, default=0.2, help='the ratio of pooling for nodes')
parser.add_argument('--kern_size', type=str, default="9,5,3", help='list of time conv kernel size for each layer')
parser.add_argument('--in_dim', type=int, default=64, help='input dimensions of GNN stacks')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimensions of GNN stacks')
parser.add_argument('--out_dim', type=int, default=256, help='output dimensions of GNN stacks')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--val-batch-size', default=16, type=int, metavar='V',
                    help='validation batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--use_benchmark', dest='use_benchmark', action='store_true',
                    default=True, help='use benchmark')
parser.add_argument('--tag', default='date', type=str,
                    help='the tag for identifying the log and model files. Just a string.')
parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test')
parser.add_argument('--model-path', type=str, default='E:/domain generalization/publicD_0.001_0.001.pth', help='')
parser.add_argument('--attention_dim', type=int, default=64, help='attention_dim')
parser.add_argument('--rho', type=int, default=0.3, help='rho')



def main():
    args = parser.parse_args()

    args.kern_size = [ int(l) for l in args.kern_size.split(",") ]

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.mode == 'train':
        best_acc, best_model_weights = main_work(args)
        torch.save(best_model_weights, args.model_path)
        print("Best accuracy:", best_acc)
        print("Model weights type:", type(best_model_weights))
        print(f"Best model weights saved to {args.model_path}")

    elif args.mode == 'test':
        results = []
        energy_results = {}

        for dsid in args.datasets:
            print(f"\n>>> Evaluating on domain: {dsid}")

            args.dataset = dsid

            train_loader, val_loader, test_loader, test_ood_loader, num_nodes, seq_length, num_classes = get_default_train_val_test_loader(
                args)

            args.num_classes = num_classes

            model = GNNStack(gnn_model_type=args.arch, num_layers=args.num_layers,
                             groups=args.groups, pool_ratio=args.pool_ratio, kern_size=args.kern_size,
                             in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                             attention_dim=args.attention_dim, rho=args.rho,
                             seq_len=seq_length, num_nodes=num_nodes, num_classes=num_classes)
            model.load_state_dict(torch.load(args.model_path))
            model.cuda(args.gpu)

            ce_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
            gce_criterion = GeneralizedCrossEntropyLoss(q=0.7).cuda(args.gpu)

            class_centers = compute_class_centers(train_loader, model, args)

            acc_id, auroc_id, auroc_ood, aupr_ood, fpr95_ood, neg_energy_id, neg_energy_ood, UK,H_score,threshold_95 = evaluate_ood(test_loader, test_ood_loader, model, ce_criterion, gce_criterion, args, return_energy=True, class_centers=class_centers)
            results.append((dsid, acc_id, auroc_id, auroc_ood, aupr_ood, fpr95_ood, UK,H_score))
            energy_results[dsid] = {'id': neg_energy_id, 'ood': neg_energy_ood}

        for r in results:
            print(f"[{r[0]}] ID Acc: {r[1]}, AUROC (ID): {r[2]}, AUROC (OOD): {r[3]}, AUPR: {r[4]}, FPR95: {r[5]}, UK: {r[6]}, H_score: {r[7]}")



def calculate_fpr95(y_true, y_scores, recall_level=0.95):
    """
    Calculate the False Positive Rate at a given recall level (default 95% recall).

    Parameters:
    - y_true: Ground truth labels, 1 for in-distribution (ID) and 0 for out-of-distribution (OOD).
    - y_scores: Predicted scores (probabilities or logits).
    - recall_level: Desired recall level (default is 0.95).

    Returns:
    - fpr: The false positive rate at the given recall level.
    """
    # Sort the scores and corresponding labels in descending order of score
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]
    sorted_scores = y_scores[sorted_indices]

    # Calculate cumulative true positives and false positives
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)

    # Calculate recall (TPR)
    recall = cumulative_tp / cumulative_tp[-1]

    # Find the index where recall first exceeds the recall_level
    threshold_idx = np.argmax(recall >= recall_level)

    # Calculate FPR at the corresponding threshold
    fpr = cumulative_fp[threshold_idx] / (len(y_true) - np.sum(y_true))  # FP / (FP + TN)

    return fpr


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))  # In-distribution scores
    neg = np.array(_neg[:]).reshape((-1, 1))  # OOD scores

    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[len(pos):] = 1

    # Calculate AUROC and AUPR
    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)

    # Calculate FPR@95
    fpr95 = calculate_fpr95(labels, examples, recall_level)
    return auroc, aupr, fpr95


def compute_min_center_distance(z_causal_emb, class_centers):
    distances = []
    for i in range(z_causal_emb.size(0)):
        feat = z_causal_emb[i].cpu()
        dists = [F.pairwise_distance(feat.unsqueeze(0), center.unsqueeze(0), p=2)
                 for center in class_centers.values()]
        min_dist = torch.stack(dists).min().item()
        distances.append(min_dist)
    return distances


def compute_kl_consistency_score(output, cf_out_list):
    out_soft = torch.softmax(output, dim=1)
    cf_soft_list = [torch.softmax(cf, dim=1) for cf in cf_out_list]

    scores = []
    for i in range(out_soft.size(0)):
        p = out_soft[i]
        kl_sum = 0.0
        for q in cf_soft_list:
            q_i = q[i]
            #kl = F.kl_div(q_i.log(), p, reduction='sum')
            eps = 1e-8
            p = torch.clamp(p, min=eps, max=1.0)
            q = torch.clamp(q, min=eps, max=1.0)
            kl = F.kl_div(q.log(), p, reduction='batchmean')
            kl_sum += kl
        scores.append(kl_sum.item())
    return scores  # s_cf




def evaluate_ood(test_loader, test_ood_loader, model, args, class_centers=None):
    model.eval()

    y_true = []
    y_pred = []
    y_true_ood = []
    total_loss_id = []  # Store total loss for ID data
    total_loss_ood = []  # Store total loss for OOD data
    with torch.no_grad():
        # Evaluate in-distribution data
        for data, label in test_loader:
            data = data.cuda(args.gpu).type(torch.float)
            label = label.cuda(args.gpu).type(torch.long)
            output, z_noncausal_perm_out1, z_noncausal_swap_out1,z_noncausal_perm_out2, z_noncausal_swap_out2, z_causal_emb,z_noncausal_emb, cf_emb_out, z_noncausal_perm1, z_noncausal_swap1,z_noncausal_perm2, z_noncausal_swap2= model(data,label)
            output_prob = torch.softmax(output, dim=1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(output_prob.cpu().numpy())

            cf_out_list = [z_noncausal_perm_out1, z_noncausal_swap_out1, z_noncausal_perm_out2, z_noncausal_swap_out2]
            lambda_ = 0.6
            s_center = compute_min_center_distance(z_causal_emb, class_centers)
            s_cf = compute_kl_consistency_score(output, cf_out_list)
            ood_scores = [lambda_ * sc + (1 - lambda_) * scf for sc, scf in zip(s_center, s_cf)]
            total_loss_id.extend(ood_scores)


        # Evaluate out-of-distribution data
        for data_ood, label_ood in test_ood_loader:
            data_ood = data_ood.cuda(args.gpu).type(torch.float)
            label_ood = label_ood.cuda(args.gpu).type(torch.long)
            output, z_noncausal_perm_out1, z_noncausal_swap_out1,z_noncausal_perm_out2, z_noncausal_swap_out2, z_causal_emb,z_noncausal_emb, cf_emb_out, z_noncausal_perm1, z_noncausal_swap1,z_noncausal_perm2, z_noncausal_swap2= model(data_ood,label)

            cf_out_list = [z_noncausal_perm_out1, z_noncausal_swap_out1, z_noncausal_perm_out2, z_noncausal_swap_out2]
            lambda_ = 0.6
            s_center = compute_min_center_distance(z_causal_emb, class_centers)
            s_cf = compute_kl_consistency_score(output, cf_out_list)
            ood_scores = [lambda_ * sc + (1 - lambda_) * scf for sc, scf in zip(s_center, s_cf)]
            total_loss_ood.extend(ood_scores)

            y_true_ood.extend(label_ood.cpu().numpy())


    y_pred_classes = [np.argmax(p) for p in y_pred]
    acc = sum(np.array(y_pred_classes) == np.array(y_true)) / len(y_true)


    auroc_id = roc_auc_score(y_true, y_pred, multi_class='ovr')

    if len(total_loss_id) > 0 and len(total_loss_ood) > 0:
        auroc_ood, aupr_ood, fpr95_ood = get_measures(total_loss_id, total_loss_ood, recall_level=0.95)
        labels = np.array([0] * len(total_loss_id) + [1] * len(total_loss_ood))
        scores = np.array(total_loss_id + total_loss_ood)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        valid = np.where(tpr >= 0.95)[0]
        best_idx = valid[np.argmin(fpr[valid])]
        threshold_95 = thresholds[best_idx]
        UK = np.mean(np.array(total_loss_ood) > threshold_95)
        H_score = 2*acc*UK/(acc+UK)
    else:
        auroc_ood, aupr_ood, fpr95_ood = 0.0, 0.0, 0.0
        labels = np.array([0] * len(total_loss_id) + [1] * len(total_loss_ood))
        scores = np.array(total_loss_id + total_loss_ood)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        valid = np.where(tpr >= 0.95)[0]
        best_idx = valid[np.argmin(fpr[valid])]
        threshold_95 = thresholds[best_idx]
        UK = np.mean(np.array(total_loss_ood) > threshold_95)
        H_score = 2 * acc * UK / (acc + UK)
    print("Min/Max of ID:", min(total_loss_id), max(total_loss_id))
    print("Min/Max of OOD:", min(total_loss_ood), max(total_loss_ood))
    print("Threshold_95:", threshold_95)

    return acc, auroc_id, auroc_ood, aupr_ood, fpr95_ood,total_loss_id,total_loss_ood, UK,H_score,threshold_95



class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCrossEntropyLoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        prob = torch.softmax(logits, dim=1)
        y_prob = prob.gather(1, targets.view(-1, 1)).squeeze()
        loss = (1 - y_prob.pow(self.q)) / self.q
        return loss.mean()

def compute_class_centers(train_loader, model, args):
    from collections import defaultdict
    model.eval()
    feature_dict = defaultdict(list)

    with torch.no_grad():
        for data, label in train_loader:
            data = data.cuda(args.gpu).float()
            label = label.cuda(args.gpu).long()
            # Forward pass
            out, z_noncausal_perm_out1, z_noncausal_swap_out1,z_noncausal_perm_out2, z_noncausal_swap_out2, z_causal_emb,z_noncausal_emb, cf_emb_out, z_noncausal_perm1, z_noncausal_swap1,z_noncausal_perm2, z_noncausal_swap2= model(data, label)
            for i in range(label.size(0)):
                feature_dict[label[i].item()].append(z_causal_emb[i].cpu())

    class_centers = {}
    for cls, feats in feature_dict.items():
        feats = [f.view(-1) for f in feats]
        stacked = torch.stack(feats)  # [N_cls, D]
        class_centers[cls] = stacked.mean(dim=0)  # [D]

    return class_centers

def main_work(args):
    best_acc1 = 0
    best_model_weights = None
    if args.tag == 'date':
        local_date = time.strftime('%m.%d', time.localtime(time.time()))
        args.tag = local_date
    log_file = '../log/{}_gpu{}_{}_{}_exp.txt'.format(args.tag, args.gpu, args.arch, args.dataset)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    train_loader, val_loader, test_loader, test_ood_loader, num_nodes, seq_length, num_classes = get_default_train_val_test_loader(args)
    # training model from net.py
    model = GNNStack(gnn_model_type=args.arch, num_layers=args.num_layers,
                     groups=args.groups, pool_ratio=args.pool_ratio, kern_size=args.kern_size,
                     in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                     attention_dim=args.attention_dim,rho=args.rho,
                     seq_len=seq_length, num_nodes=num_nodes, num_classes=num_classes,
                     regularizations=["feature_smoothing", "degree", "sparse"])

    # print & log
    log_msg('epochs {}, lr {}, weight_decay {}'.format(args.epochs, args.lr, args.weight_decay), log_file)
    # determine whether GPU or not
    if not torch.cuda.is_available():
        print("Warning! Using CPU!!!")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        # collect cache
        gc.collect()
        torch.cuda.empty_cache()
        model = model.cuda(args.gpu)
        if args.use_benchmark:
            cudnn.benchmark = True
        print('Using cudnn.benchmark.')
    else:
        print("Error! We only have one gpu!!!")
    # define loss function(criterion) and optimizer
    ce_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    gce_criterion = GeneralizedCrossEntropyLoss(q=0.7).cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                              patience=50, verbose=True)
    # validation
    if args.evaluate:
        validate(val_loader, model, ce_criterion, gce_criterion,args)
        return
    # train & valid
    print('****************************************************')
    print(args.dataset)
    dataset_time = AverageMeter('Time', ':6.3f')
    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    epoches = []
    end = time.time()

    for epoch in range(args.epochs):
        epoches += [epoch]
        # train for one epoch
        acc_train_per, loss_train_per = train(train_loader, model, ce_criterion, gce_criterion, optimizer, lr_scheduler, args,num_classes)
        acc_train += [acc_train_per]
        loss_train += [loss_train_per]
        msg = f'TRAIN, epoch {epoch}, loss {loss_train_per}, acc {acc_train_per}'
        log_msg(msg, log_file)
        # evaluate on validation set
        acc_val_per, loss_val_per = validate(val_loader, model, ce_criterion, gce_criterion,args, num_classes)
        acc_val += [acc_val_per]
        loss_val += [loss_val_per]

        log_msg(msg, log_file)
        if acc_val_per > best_acc1:
            best_acc1 = acc_val_per
            best_model_weights = model.state_dict()
    # measure elapsed time
    dataset_time.update(time.time() - end)
    # log & print the best_acc
    msg = f'\n\n * BEST_ACC: {best_acc1}\n * TIME: {dataset_time}\n'
    log_msg(msg, log_file)
    print(f' * best_acc1: {best_acc1}')
    print(f' * time: {dataset_time}')
    print('****************************************************')

    model.load_state_dict(best_model_weights)

    # collect cache
    gc.collect()
    torch.cuda.empty_cache()
    return best_acc1, best_model_weights




def train(train_loader, model, ce_criterion,  optimizer, lr_scheduler, args, beta=0.001,lambda_do=0.001):

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')
    model.train()

    for count, (data, label) in enumerate(train_loader):
        data = data.cuda(args.gpu).float()
        label = label.cuda(args.gpu).long()

        out, z_noncausal_perm_out1, z_noncausal_swap_out1, z_noncausal_perm_out2, z_noncausal_swap_out2,z_causal_emb,z_noncausal_emb, cf_emb_out,z_noncausal_perm1,z_noncausal_swap1,z_noncausal_perm2,z_noncausal_swap2= model(data,label)

        ce_loss_causal = ce_criterion(out, label)

        logits_list = [out, z_noncausal_perm_out1, z_noncausal_swap_out1, z_noncausal_perm_out2,
                       z_noncausal_swap_out2]
        invariance_loss = energy_based_diversity_regularization(logits_list)

        gce_loss_noncausal = ce_criterion(cf_emb_out, label)


        z_noncausal_list = [z_noncausal_emb,z_noncausal_perm1, z_noncausal_swap1, z_noncausal_perm2, z_noncausal_swap2]
        contrastive_loss = beta * contrastive_loss_supervised(z_causal_emb, z_noncausal_list, label)

        total_loss = ce_loss_causal +gce_loss_noncausal + contrastive_loss +lambda_do * invariance_loss
        acc1 = accuracy(out, label, topk=(1,))
        losses.update(total_loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    lr_scheduler.step(top1.avg)
    return top1.avg, losses.avg

def validate(val_loader, model, ce_criterion, args, beta=0.001,lambda_do=0.001):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    #lambda_smooth = 1
    model.eval()
    with torch.no_grad():
        for count, (data, label) in enumerate(val_loader):
            data = data.cuda(args.gpu, non_blocking=True).float()
            label = label.cuda(args.gpu, non_blocking=True).long()

            out, z_noncausal_perm_out1, z_noncausal_swap_out1, z_noncausal_perm_out2, z_noncausal_swap_out2, z_causal_emb,z_noncausal_emb, cf_emb_out, z_noncausal_perm1, z_noncausal_swap1, z_noncausal_perm2, z_noncausal_swap2= model(
                data,label)

            ce_loss_causal = ce_criterion(out, label)

            logits_list = [out, z_noncausal_perm_out1, z_noncausal_swap_out1, z_noncausal_perm_out2,
                           z_noncausal_swap_out2]
            invariance_loss= energy_based_diversity_regularization(logits_list)

            gce_loss_noncausal = ce_criterion(cf_emb_out, label)

            z_noncausal_list=[z_noncausal_emb,z_noncausal_perm1, z_noncausal_swap1,z_noncausal_perm2, z_noncausal_swap2]
            contrastive_loss = beta * contrastive_loss_supervised(z_causal_emb,z_noncausal_list,label)
            total_loss = ce_loss_causal + gce_loss_noncausal + contrastive_loss  + lambda_do * invariance_loss
            acc1 = accuracy(out, label, topk=(1,))
            losses.update(total_loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))

    return top1.avg, losses.avg


def compute_energy(logits):
    return -torch.logsumexp(logits, dim=-1)  # shape: (B,)

def pairwise_energy_divergence(energy_list):
    num_env = len(energy_list)
    total = 0.0
    count = 0
    for i in range(num_env):
        for j in range(i + 1, num_env):
            dist = F.mse_loss(energy_list[i], energy_list[j], reduction='mean')
            total += dist
            count += 1
    return total / count

def energy_based_diversity_regularization(logits_list):
    num_envs = len(logits_list)
    energy_list = [compute_energy(logits) for logits in logits_list]

    total = 0.0
    count = 0

    for j in range(num_envs):
        for k in range(j + 1, num_envs):
            diff = energy_list[j] - energy_list[k]
            pairwise_dist = torch.mean(diff ** 2)
            total += pairwise_dist
            count += 1
    diversity_loss = (2.0 / (num_envs * (num_envs - 1))) * total if count > 0 else torch.tensor(0.0)
    return diversity_loss

class SoftTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(SoftTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T)  # [B, B]

        # Mask matrices
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        identity_mask = torch.eye(features.size(0), device=features.device).bool()

        pos_mask = label_matrix & ~identity_mask  # exclude self-pairs
        neg_mask = ~label_matrix

        pos_sim = []
        neg_sim = []

        for i in range(features.size(0)):
            pos = sim_matrix[i][pos_mask[i]]  # same class, not self
            neg = sim_matrix[i][neg_mask[i]]  # different class

            if len(pos) > 0:
                pos_sim.append(pos.mean())
            else:
                pos_sim.append(torch.tensor(0.0, device=features.device))

            if len(neg) > 0:
                neg_sim.append(neg.max())
            else:
                neg_sim.append(torch.tensor(0.0, device=features.device))

        pos_sim = torch.stack(pos_sim)
        neg_sim = torch.stack(neg_sim)

        loss = F.relu(self.margin + neg_sim - pos_sim).mean()
        return loss


def contrastive_loss_supervised(z_causal, z_noncausal_list, label, temp=0.1):
    """
    - z_causal: (B, D)
    - z_noncausal_list: list of (B, D), e.g., [cf1, cf2, ..., noncausal_emb]
    """
    # Normalize
    z_causal = F.normalize(z_causal, dim=1)
    z_noncausal_list = [F.normalize(z, dim=1) for z in z_noncausal_list]
    batch_size = z_causal.size(0)

    # === Positive pairs (same label): supervised contrastive loss ===
    sim_matrix = torch.matmul(z_causal, z_causal.T) / temp
    logits_mask = torch.ones_like(sim_matrix) - torch.eye(batch_size, device=sim_matrix.device)
    label_mask = torch.eq(label.view(-1, 1), label.view(1, -1)).float().to(z_causal.device)

    sim_matrix = sim_matrix * logits_mask
    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)

    mean_log_prob_pos = (label_mask * log_prob).sum(1) / (label_mask.sum(1) + 1e-6)
    contrastive_loss_causal = - mean_log_prob_pos.mean()

    # === Negative pairs: causal vs. noncausal ===
    push_losses = []
    for z_non in z_noncausal_list:
        # cosine similarity between causal and noncausal
        sim = torch.sum(z_causal * z_non, dim=1)  # (B,)
        push_losses.append(sim.mean())

    contrastive_loss_cf = torch.stack(push_losses).mean()

    loss = contrastive_loss_causal + contrastive_loss_cf
    return loss


if __name__ == '__main__':
    main()
