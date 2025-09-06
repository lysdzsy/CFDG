import torch
import torch.utils.data
from torch.utils.data import TensorDataset


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def log_msg(message, log_file):
    with open(log_file, 'a') as f:
        print(message, file=f)


def min_max_normalize(data, min_val, max_val):
    # Min-Max Scaling
    return 2 * (data - min_val) / (max_val - min_val) - 1


def get_default_train_val_test_loader(args):
    # get dataset-id
    dsid = args.dataset

    # get dataset from .pt
    data_train = torch.load(f'E:/domain generalization/data/UCR/public/{dsid}/X_train.pt')
    data_val = torch.load(f'E:/domain generalization/data/UCR/public/{dsid}/X_valid.pt')
    data_test = torch.load(f'E:/domain generalization/data/UCR/public/{dsid}/X_test.pt')
    data_test_ood = torch.load(f'E:/domain generalization/data/UCR/public/{dsid}/X_test_ood.pt')
    label_train = torch.load(f'E:/domain generalization/data/UCR/public/{dsid}/y_train.pt')
    label_val = torch.load(f'E:/domain generalization/data/UCR/public/{dsid}/y_valid.pt')
    label_test = torch.load(f'E:/domain generalization/data/UCR/public/{dsid}/y_test.pt')
    label_test_ood = torch.load(f'E:/domain generalization/data/UCR/public/{dsid}/y_test_ood.pt')

    num_sensors = data_train.size(2)
    min_vals = []
    max_vals = []

    for i in range(num_sensors):
        sensor_data = data_train[:, 0, i, :]
        min_vals.append(sensor_data.min())
        max_vals.append(sensor_data.max())

    min_vals = torch.tensor(min_vals).view(1, 1, num_sensors, 1)
    max_vals = torch.tensor(max_vals).view(1, 1, num_sensors, 1)

    data_train = min_max_normalize(data_train, min_vals, max_vals)

    data_val = min_max_normalize(data_val, min_vals, max_vals)
    data_test = min_max_normalize(data_test, min_vals, max_vals)
    data_test_ood = min_max_normalize(data_test_ood, min_vals, max_vals)

    num_nodes = data_val.size(-2)

    seq_length = data_val.size(-1)

    num_classes = len(torch.bincount(label_val.type(torch.int)))

    # convert data & labels to TensorDataset
    train_dataset = TensorDataset(data_train, label_train)
    val_dataset = TensorDataset(data_val, label_val)
    test_dataset = TensorDataset(data_test, label_test)
    test_ood_dataset = TensorDataset(data_test_ood, label_test_ood)

    # data_loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.val_batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True, drop_last=True)
    test_ood_loader = torch.utils.data.DataLoader(test_ood_dataset,
                                                  batch_size=args.val_batch_size,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True, drop_last=True)
    return train_loader, val_loader, test_loader, test_ood_loader, num_nodes, seq_length, num_classes






