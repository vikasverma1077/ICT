import torch
from torch.autograd import Variable
import os, errno
import numpy as np
from scipy import linalg


import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from itertools import repeat, cycle



def to_var(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad=requires_grad)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)


def make_dir_if_not_exists(path):
    """Make directory if doesn't already exists"""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

def apply_zca(data, zca_mean, zca_components):
        temp = data.numpy()
        shape = temp.shape
        temp = temp.reshape(-1, shape[1]*shape[2]*shape[3])
        temp = np.dot(temp - zca_mean, zca_components.T)
        temp = temp.reshape(-1, shape[1], shape [2], shape[3])
        data = torch.from_numpy(temp).float()
        return data
        #print (temp)


def to_one_hot(inp):
    y_onehot = torch.FloatTensor(inp.size(0), 10)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    return Variable(y_onehot.cuda(),requires_grad=False)

"""
def mixup_data(input, target, lam):
    
    lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
    lam = Variable(lam)
    indices = np.random.permutation(input.size(0))
    input = input*lam.expand_as(input) + input[indices]*(1-lam.expand_as(input))
    target = target* lam.expand_as(target) + target*(1 - lam.expand_as(target))
    return input, target
"""

def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_labelled_unlabelled(input_l, input_u, target_l, target_u, mixup_alpha):
    
    if mixup_alpha > 0.:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
    else:
        lam = 1.
    
    lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
    lam = Variable(lam)
    #lam = torch.max(lam, 1-lam)
    #indices = np.random.permutation(out.size(0))
    out = input_l*lam.expand_as(input_l) + input_u*(1-lam.expand_as(input_u))
    target_l = to_one_hot(target_l)
    target = target_l* lam.expand_as(target_l) + target_u*(1 - lam.expand_as(target_u))
    return out, target


def mixup_data_hidden(input, target,  mixup_alpha):
    if mixup_alpha > 0.:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
    else:
        lam = 1.
    lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
    lam = Variable(lam)
    indices = np.random.permutation(input.size(0))
    #target = to_one_hot(target)
    output = input*lam.expand_as(input) + input[indices]*(1-lam.expand_as(input))
    target_a, target_b = target ,target[indices]
    
    return output, target_a, target_b, lam



def load_data_subset(data_aug, batch_size,workers,dataset, data_target_dir, labels_per_class=100, valid_labels_per_class = 500):
    ## copied from GibbsNet_pytorch/load.py
    import numpy as np
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
        
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    elif dataset == 'mnist':
        pass 
    else:
        assert False, "Unknow dataset : {}".format(dataset)
    
    if data_aug==1:
        print ('data aug')
        if dataset == 'svhn':
            train_transform = transforms.Compose(
                                             [ transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        elif dataset == 'mnist':
            hw_size = 24
            train_transform = transforms.Compose([
                                transforms.RandomCrop(hw_size),                
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
            test_transform = transforms.Compose([
                                transforms.CenterCrop(hw_size),                       
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
        else:    
            train_transform = transforms.Compose(
                                                 [transforms.RandomHorizontalFlip(),
                                                  transforms.RandomCrop(32, padding=2),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        print ('no data aug')
        if dataset == 'mnist':
            hw_size = 28
            train_transform = transforms.Compose([
                                transforms.ToTensor(),       
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
            test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
                
        else:   
            train_transform = transforms.Compose(
                                                 [transforms.ToTensor(),
                                                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif dataset == 'svhn':
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'mnist':
        train_data = datasets.MNIST(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.MNIST(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    #print ('svhn', train_data.labels.shape)
    elif dataset == 'stl10':
        train_data = datasets.STL10(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.STL10(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

        
    n_labels = num_classes
    
    def get_sampler(labels, n=None, n_valid= None):
        # Only choose digits in n_labels
        # n = number of labels per class for training
        # n_val = number of lables per class for validation
        #print type(labels)
        #print (n_valid)
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        
        indices_valid = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in range(n_labels)])
        indices_train = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid+n] for i in range(n_labels)])
        indices_unlabelled = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:] for i in range(n_labels)])
        #print (indices_train.shape)
        #print (indices_valid.shape)
        #print (indices_unlabelled.shape)
        indices_train = torch.from_numpy(indices_train)
        indices_valid = torch.from_numpy(indices_valid)
        indices_unlabelled = torch.from_numpy(indices_unlabelled)
        sampler_train = SubsetRandomSampler(indices_train)
        sampler_valid = SubsetRandomSampler(indices_valid)
        sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
        return sampler_train, sampler_valid, sampler_unlabelled
    
    #print type(train_data.train_labels)
    
    # Dataloaders for MNIST
    if dataset == 'svhn':
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.labels, labels_per_class, valid_labels_per_class)
    elif dataset == 'mnist':
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.train_labels.numpy(), labels_per_class, valid_labels_per_class)
    else: 
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.train_labels, labels_per_class, valid_labels_per_class)
    
    labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = train_sampler,  num_workers=workers, pin_memory=True)
    validation = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = valid_sampler,  num_workers=workers, pin_memory=True)
    unlabelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = unlabelled_sampler,  num_workers=workers, pin_memory=True)
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return labelled, validation, unlabelled, test, num_classes

if __name__ == '__main__':
    labelled, validation, unlabelled, test, num_classes  = load_data_subset(data_aug=1, batch_size=32,workers=1,dataset='cifar10', data_target_dir="/u/vermavik/data/DARC/cifar10", labels_per_class=100, valid_labels_per_class = 500)
    for (inputs, targets), (u, _) in zip(cycle(labelled), unlabelled):
        print (input)
