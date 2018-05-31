import os,sys
import shutil
import time
import traceback
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F


# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

    
class PixelWiseFlowLoss(nn.modules.loss._Loss):
    def __init__(self,size_average=False, reduce=False, minval=5):
        super(PixelWiseFlowLoss,self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.minval = minval
        self.SmoothL1 = nn.SmoothL1Loss(self.size_average, self.reduce)
        
    def forward(self,predict,target,visibility):
        """
        predict: (b,c,h,w) tensor with output from flow deconv
        target:  (b,h,w) tensor with correct flow
        visibility: (b,h,w) tensor with true matchability for each pixel
        """
        _assert_no_grad(target)
        _assert_no_grad(visibility)
        '''
        minval_np = np.ones( predict.size() , dtype=np.float32)*self.minval*self.minval
        minval_var = torch.autograd.Variable(torch.from_numpy(minval_np).cuda())        
        _assert_no_grad(minval_var)

        #old loss  
        pixelloss = target - predict
        pixelloss = pixelloss*pixelloss
        pixelloss = torch.min(pixelloss, minval_var)
        '''
        pixelloss = self.SmoothL1(predict, target );
        pixelloss *= visibility
        s = pixelloss.sum()
        s1 = visibility.sum()
        if s1 is None:
            s1=1
            s=0
        s = s/s1
        return s


class PixelWiseNLLLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,weight=None, size_average=True, ignore_index=-100 ):
        super(PixelWiseNLLLoss,self).__init__(weight,size_average)
        self.ignore_index = ignore_index
        self.reduce = False
        nweight = np.ones((2), dtype=np.float32)
        nweight[1] = 100.0
        self.weight = torch.from_numpy(nweight)
        
    def forward(self,predict,target):
        """
        predict: (b,c,h,w) tensor with output from logsoftmax
        target:  (b,h,w) tensor with correct class
        """
        _assert_no_grad(target)
        # reduce for below is false, so returns (b,h,w)
        pixelloss = F.nll_loss(predict,target, self.weight, self.size_average, self.ignore_index, self.reduce)
        return torch.mean(pixelloss)


def save_checkpoint(state, is_best, p, filename='checkpoint.pth.tar'):
    if p>0:
        filename = "checkpoint.%dth.tar"%(p)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = lr * (0.5 ** (epoch // 300))
    lr = lr
    #lr = lr*0.992
    #print "adjust learning rate to ",lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy_vis(output, target):

    batch_size = target.size(0)
    _, pred = output.max( 1, keepdim=False) #take class with max prob. 
    targetex = target.resize_( pred.size() ) # expanded view, should not include copy
    correct = pred.eq( targetex ) #returns 1 for accurate pred

    # we want counts for elements wise
    num_per_class = {}
    corr_per_class = {}
    total_corr = 0
    total_pix  = 0

    for c in range(output.size(1)):
        # loop over classes
        classmat = targetex.eq(int(c)) # elements where class is labeled
        num_per_class[c] = classmat.sum()
        corr_per_class[c] = (correct*classmat).sum() # mask by class matrix, then sum
        total_corr += corr_per_class[c]
        total_pix  += num_per_class[c]

    # make result vector
    res = []
    for c in range(output.size(1)):
        if num_per_class[c]>0:
            res.append( corr_per_class[c]/float(num_per_class[c])*100.0 )
        else:
            res.append( 0.0 )
                
    # totals
    res.append( 100.0*float(total_corr)/total_pix )

    #print res
    return res

def accuracy_flow(output, target, visibility, pix):

    batch_size = target.size(0)
    pred = output.resize_( target.size() ) #resize to match pred
    correct = torch.abs( pred - target )
    correct = correct.le( pix ) #one below thresh, zero above thresh
    correct = (correct.float())*visibility #mask zero pixels

    # we want counts for elements wise
    total_corr = 0
    total_pix  = 0
    
    total_corr = correct.sum()
    total_pix  = visibility.sum()

    # make result vector
    if total_pix==0:
        total_pix=1.0
        total_corr=0.0
    res = 100.0*float(total_corr)/total_pix 
    #print res
    return res

def dump_lr_schedule( startlr, numepochs ):
    for epoch in range(0,numepochs):
        lr = startlr*(0.5**(epoch//300))
        if epoch%10==0:
            print "Epoch [%d] lr=%.3e"%(epoch,lr)
    print "Epoch [%d] lr=%.3e"%(epoch,lr)
    return
