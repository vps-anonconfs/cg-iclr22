import os
import torch
from PIL import Image
import pandas as pd
import pickle
from torchvision import datasets

import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import MSE, Accuracy
from wilds.common.metrics.loss import Loss
from wilds.common.metrics.loss import ElementwiseLoss

"""
0.1 minority
[1.4557068347930908, 1.707288384437561, 1.5363376140594482, 1.5088559389114382, 1.6053923368453982, 1.425696849822998]
groupDRO | 1.540(0.094)
[1.3999165296554563, 1.5905730724334717, 1.4600753784179688, 1.4331107139587402, 1.5091383457183838, 1.367109775543213]
CG | 1.460(0.074)
"""

def make_environment(num, frac, train=True):
    # first feature is common and second is spurious
    gs = torch.from_numpy(np.random.choice(3, p=frac, size=num))
    DIM = 1000
    X = torch.randn([num, DIM])
    n = num
    Y = (X[:, :2].sum(dim=-1)>0).type(torch.float32)
    p1, p2, p3 = 1, 0.5, 0
#     rand = torch.tensor(np.random.binomial(1, 0.9, [len(Y)]))
#     Y = rand*Y + (1-rand)*(1-Y)
    rand1, rand3 = np.random.binomial(1, p1, [len(Y)]), np.random.binomial(1, p3, [len(Y)])
    rand2 = np.random.binomial(1, p2, [len(Y)])
    rand1, rand2, rand3 = torch.tensor(rand1), torch.tensor(rand2), torch.tensor(rand3)
    Y1 = rand1*Y + (1-rand1)*(1-Y)
    Y2 = rand2*Y + (1-rand2)*(1-Y)
    Y3 = rand3*Y + (1-rand3)*(1-Y)
    X[:, DIM-1] = torch.where(gs==0, Y1, X[:, DIM-1])
    X[:, DIM-1] = torch.where(gs==1, Y2, X[:, DIM-1])
    X[:, DIM-1] = torch.where(gs==2, Y3, X[:, DIM-1])
    print (X[:10], Y[:10], gs[:10])
    
    print ("label stats:", np.unique(Y.numpy(), return_counts=True))
    print ("g stats:", np.unique(gs.numpy(), return_counts=True))
    return {'X': X, 'y': Y, 'g': gs}

def make_environmen3(num, frac, train=True):
    # first feature is common and second is spurious
    gs = torch.from_numpy(np.random.choice(3, p=frac, size=num))
    DIM = 10
    X = torch.randn([num, DIM])
    n = num
    Y = (X[:, :-1].sum(dim=-1)).type(torch.float32)
    X[:, DIM-1] = torch.where(gs==0, Y, X[:, DIM-1])
    X[:, DIM-1] = torch.where(gs==2, -Y, X[:, DIM-1])
    X[:, DIM-1] = torch.where(gs==1, torch.zeros_like(Y), X[:, DIM-1])
    print (X[:10], Y[:10], gs[:10])
    
    print ("label stats:", np.unique(Y.numpy(), return_counts=True))
    print ("g stats:", np.unique(gs.numpy(), return_counts=True))
    return {'X': X, 'y': Y, 'g': gs}

def make_environment2(num, frac, train=True):
    # first feature is common and second is spurious
    n = num
    gs = torch.from_numpy(np.random.choice(3, p=frac, size=num))
    DIM = 3
    sigma_spu, sigma_core = 1e-2, 0.707
    X = torch.randn([num, DIM])
    Y = torch.from_numpy(np.random.binomial(1, 0.5, num)).type(torch.float32)
    X[:, :2] = torch.unsqueeze(Y, dim=-1)*(1 + torch.randn([num, DIM-1])*sigma_core)
    X[:, DIM-1] = torch.where(gs==0, Y + torch.randn(len(Y))*sigma_spu, X[:, DIM-1])
    X[:, DIM-1] = torch.where(gs==2, (1-Y) + torch.randn(len(Y))*sigma_spu, X[:, DIM-1])
    print (X[:10], Y[:10], gs[:10])
    
    print ("label stats:", np.unique(Y.numpy(), return_counts=True))
    print ("g stats:", np.unique(gs.numpy(), return_counts=True))
    return {'X': X, 'y': Y, 'g': gs}

def make_environment4(num, frac, train=True):
    # first feature is common and second is spurious
    gs = torch.from_numpy(np.random.choice(3, p=frac, size=num))
    DIM = 300
    sigma_spu, sigma_core = 1e-2, 1e-2
    X = torch.randn([num, DIM])
    n = num
    Y = (X[:, :-1].sum(dim=-1) > 0).type(torch.float32)
    X[:, 0] += torch.randn([n])*sigma_core
    X[:, 1] += torch.randn([n])*sigma_core
#     rnd = torch.distributions.Binomial(1, torch.tensor([0.5, 0.5])).sample([n])[:, 0].type(torch.bool)
#     X[:, DIM-1] = torch.where(rnd, Y, 1-Y) + torch.randn(len(Y))*sigma_spu
    X[:, DIM-1] = torch.where(gs==0, Y + torch.randn(len(Y))*sigma_spu, X[:, DIM-1])
    X[:, DIM-1] = torch.where(gs==1, 1 + torch.randn(len(Y))*sigma_spu, X[:, DIM-1])
    X[:, DIM-1] = torch.where(gs==2, (1-Y) + torch.randn(len(Y))*sigma_spu, X[:, DIM-1])
    print (X[:10], Y[:10], gs[:10])
    
    print ("label stats:", np.unique(Y.numpy(), return_counts=True))
    print ("g stats:", np.unique(gs.numpy(), return_counts=True))
    return {'X': X, 'y': Y, 'g': gs}

class SpuSimpleDataset(WILDSDataset):
    def __init__(self, root_dir='data', download=False, split_scheme='official'):
        required_attrs = ['_dataset_name', '_data_dir',
                          '_split_scheme', '_split_array',
                          '_y_array', '_y_size',
                          '_metadata_fields', '_metadata_array']

        self._dataset_name = "spu_2feature"
        self._data_dir = os.path.join(root_dir, self._dataset_name)

        train_data = make_environment(1000, [0.49, 0.49, 0.02], train=True)
        val_data = make_environment(1000, [0.34, 0.33, 0.33], train=False)
        test_data = make_environment(10000, [0.34, 0.33, 0.33], train=False)
        
        _x_array, _y_array, _split_array, _g_array = [], [], [], []
        i = 0
        for di, d in enumerate([train_data, val_data, test_data]):
            x, y = d['X'], d['y']
            g = d['g']
            for j in range(len(y)):
                _x_array.append(x[j])
                _y_array.append(y[j])
                _g_array.append(g[j])
            _split_array += [di]*len(y)
        
        _y_array = np.array(_y_array)
        _g_array = np.array(_g_array)
        self._input_array = _x_array
        self._y_array = torch.LongTensor(_y_array)
        self._split_array = np.array(_split_array)
        # partition the train in to val and test
        self._split_scheme = split_scheme
        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.stack(
            (torch.LongTensor(_g_array), self._y_array),
            dim=1
        )
        self._metadata_fields = ['group', 'y']
        self._metadata_map = {
            'group': ['       majority', ' clean majority', '       minority'], 
            'y': [' 0', '1']
        }
                        
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['group']))
        self._metric = Loss(loss_fn=torch.nn.CrossEntropyLoss())
        
        super().__init__(root_dir, download, split_scheme)
    
    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        return self._input_array[idx]
        
    def eval(self, y_pred, y_true, metadata):
        return self.standard_group_eval(
            self._metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
    
if __name__ == '__main__':
    dset = SpuSimpleDataset('data')
    train, val, test = dset.get_subset('train'), dset.get_subset('val'), dset.get_subset('test')
    print ("Train, val, test sizes:", len(train), len(val), len(test))