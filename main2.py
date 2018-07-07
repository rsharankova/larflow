import os,sys
import shutil
import time
import traceback
import numpy as np
import cv2 as cv

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
from torch.autograd import Variable

# tensorboardX
from tensorboardX import SummaryWriter

from larcvdataset import LArCVDataset
import network
import myfunc

# global variables
best_prec1_vis= 0.0
best_prec1_flow= 0.0
writer = SummaryWriter()

torch.cuda.device( 1 )

    
def main():

    global best_prec1_vis
    global best_prec1_flow
    global writer
    
    model = network.mymodel( num_classes=1, input_channels=1, showsizes=False)
    model.cuda()
    #print "Loaded model: ",model

    # define loss function (criterion) and optimizer
    criterion1 = myfunc.PixelWiseFlowLoss(minval=4).cuda()
    criterion2 = myfunc.PixelWiseNLLLoss().cuda()
    
    # training parameters
    lmbd = 0.5
    lr = 1.0e-4 #-3 
    momentum = 0.9
    weight_decay = 1.0e-3
    batchsize_train = 8
    batchsize_valid = 8
    start_epoch = 0
    epochs      = 50 #1500
    nbatches_per_iter = 25
    
    if len(sys.argv)>1:
        epochs = int(sys.argv[1])
    print "Number of epochs: ", epochs
    print "Train batch: ", batchsize_train
    print "# batch per iter: ", nbatches_per_iter

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    cudnn.benchmark = True

    # dataset
    #iotrain = LArCVDataset("train_dataloader.cfg", "ThreadProcessor", loadallinmem=True)
    iotrain = LArCVDataset("train_dataloader.cfg", "ThreadProcessor")
    iovalid = LArCVDataset("valid_dataloader.cfg", "ThreadProcessorTest")
    
    iotrain.start(batchsize_train)
    iovalid.start(batchsize_valid)

    #nbatch per epoch
    NENTRIES = iotrain.io.fetch_n_entries()
    
    #NENTRIES=0;
    if NENTRIES>0:
            nbatches_per_epoch = NENTRIES/batchsize_train
            nbatches_per_valid = NENTRIES/batchsize_valid
    else:
            nbatches_per_epoch = 1
            nbatches_per_valid = 1
                

    iter_per_epoch = nbatches_per_epoch/nbatches_per_iter
    iter_per_valid = 5
    iter_per_checkpoint = 150
    num_iters = iter_per_epoch*epochs
    print "Iterations: ", num_iters
    # Resume training option
    if False:
        checkpoint = torch.load( "checkpoint.pth.p01.tar" )
        best_prec1 = checkpoint["best_prec1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer'])


    for ii in range(0, num_iters):
        
        myfunc.adjust_learning_rate(optimizer, ii, lr)
        print "Iter:%d Epoch:%d.%d "%(ii,ii/iter_per_epoch,ii%iter_per_epoch),
        for param_group in optimizer.param_groups:
            print "lr=%.3e"%(param_group['lr']),
            print
            
            # train for one epoch
            try:
                train_ave_loss, train_ave_acc_vis, train_ave_acc_flow = train(iotrain, model, criterion1, criterion2, lmbd, optimizer, nbatches_per_iter, ii, 10)
                
            except Exception,e:
                print "Error in training routine!"
                print e.message
                print e.__class__.__name__
                traceback.print_exc(e)
                break
            print "Iter:%d Epoch [%d.%d] train aveloss=%.3f aveacc_vis=%.3f aveacc_flow=%.3f"%(ii,ii/iter_per_epoch,ii%iter_per_epoch,
                                                                                               train_ave_loss,train_ave_acc_vis,train_ave_acc_flow)             
            # evaluate on validation set
            if ii%iter_per_valid==0:
                try:
                    prec1_vis, prec1_flow = validate(iovalid, model, criterion1, criterion2, lmbd, nbatches_per_iter, ii, 10)
                except Exception,e:
                    print "Error in validation routine!"
                    print e.message
                    print e.__class__.__name__
                    traceback.print_exc(e)
                    break
                
                # remember best prec@1 and save checkpoint
                is_best_flow = prec1_flow > best_prec1_flow
                best_prec1_flow = max(prec1_flow, best_prec1_flow)
                is_best_vis = prec1_vis > best_prec1_vis
                best_prec1_vis = max(prec1_vis, best_prec1_vis)
                
                # check point for best model
                if is_best_flow:
                    print "Saving best model"
                    myfunc.save_checkpoint({
                        'iter':ii,
                        'epoch': ii/iter_per_epoch,
                        'state_dict': model.state_dict(),
                        'best_prec1_vis': best_prec1_vis,
                        'best_prec1_flow': best_prec1_flow,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best_flow, -1)
                    
            # periodic checkpoint
            if ii>0 and ii%iter_per_checkpoint==0:
                print "saving periodic checkpoint"
                myfunc.save_checkpoint({
                    'iter':ii,
                    'epoch': ii/iter_per_epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1_vis': best_prec1_vis,
                    'best_prec1_flow': best_prec1_flow,
                    'optimizer' : optimizer.state_dict(),
                }, False, ii)
                
        # end of profiler context
        print "saving last state"
        myfunc.save_checkpoint({
            'iter':num_iters,
            'epoch': num_iters/iter_per_epoch,
            'state_dict': model.state_dict(),
            'best_prec1_vis': best_prec1_vis,
            'best_prec1_flow': best_prec1_flow,
            'optimizer' : optimizer.state_dict(),
        }, False, num_iters)

    '''
    for epoch in range(start_epoch, epochs):

        myfunc.adjust_learning_rate(optimizer, epoch, lr)
        print "Epoch [%d]: "%(epoch),
        for param_group in optimizer.param_groups:
            print "lr=%.3e"%(param_group['lr']),
        print
            
        try:
            train_ave_loss, train_ave_acc_vis, train_ave_acc_flow = train(iotrain, model, criterion1, criterion2, lmbd, optimizer, nbatches_per_epoch, epoch, 100)
            #train_ave_loss, train_ave_acc_vis, train_ave_acc_flow = train(data, model, criterion1, criterion2, lmbd, optimizer, nbatches_per_epoch, epoch, 50)
        except Exception,e:
            print "Error in training routine!"            
            print e.message
            print e.__class__.__name__
            traceback.print_exc(e)
            break
        print "Epoch [%d] train aveloss=%.3f aveacc_vis=%.3f aveacc_flow=%.3f"%(epoch,train_ave_loss,train_ave_acc_vis,train_ave_acc_flow)

        # evaluate on validation set
        try:
            prec1_vis, prec1_flow = validate(iovalid, model, criterion1, criterion2, lmbd, nbatches_per_valid, epoch, 100)
            #prec1_vis, prec1_flow = validate(data2, model, criterion1, criterion2, lmbd, nbatches_per_valid, epoch, 50)
        except Exception,e:
            print "Error in validation routine!"            
            print e.message
            print e.__class__.__name__
            traceback.print_exc(e)
            break

        
        # remember best prec@1 and save checkpoint
        is_best_vis = prec1_vis > best_prec1_vis
        best_prec1_vis = max(prec1_vis, best_prec1_vis)
        is_best_flow = prec1_flow > best_prec1_flow
        best_prec1_flow = max(prec1_flow, best_prec1_flow)
        myfunc.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1_vis': best_prec1_vis,
            'best_prec1_flow': best_prec1_flow,
            'optimizer' : optimizer.state_dict(),
        }, is_best_vis, -1)

        if epoch%1==0:
            myfunc.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1_vis': best_prec1_vis,
                'best_prec1_flow': best_prec1_flow,
                'optimizer' : optimizer.state_dict(),
            }, False, epoch)
            
    '''
    writer.close()
    
    iotrain.stop()
    iovalid.stop()



def train(train_loader, model, criterion1, criterion2, lmbd, optimizer, nbatches, epoch, print_freq):

    global writer
    
    batch_time = myfunc.AverageMeter()
    data_time = myfunc.AverageMeter()
    format_time = myfunc.AverageMeter()
    train_time = myfunc.AverageMeter()
    losses = myfunc.AverageMeter()
    top1_vis = myfunc.AverageMeter()
    top1_flow = myfunc.AverageMeter()

    acc_list_vis = []
    for i in range(5):
        acc_list_vis.append( myfunc.AverageMeter() )
    acc_list_flow = []
    for i in range(5):
        acc_list_flow.append( myfunc.AverageMeter() )

    # switch to train mode
    model.train()

    for i in range(0,nbatches):
        #print "epoch ",epoch," batch ",i," of ",nbatches
        batchstart = time.time()
    
        end = time.time()        
        data = train_loader[i]
        #data = train_loader #debug

        # measure data loading time
        data_time.update(time.time() - end)
        end = time.time()
        img  = data["imageY"]
        img2 = data["imageU"]
        lbl  = data["label"]
        vis  = data["match"]
        img_np  = np.zeros( (img.shape[0], 1, 512, 512), dtype=np.float32 )
        img2_np = np.zeros( (img.shape[0], 1, 512, 512), dtype=np.float32 )
        lbl_np  = np.zeros( (lbl.shape[0], 1, 512, 512 ), dtype=np.float32 )
        vis_np  = np.zeros( (vis.shape[0], 512, 512 ), dtype=np.int )
        fvis_np  = np.zeros( (vis.shape[0], 1, 512, 512 ), dtype=np.float32 )

        # batch loop
        for j in range(img.shape[0]):
            img_np[j,0,:,:]  = img[j].reshape( (512,512) )
            img2_np[j,0,:,:] = img2[j].reshape( (512,512) )
            lbl_np[j,0,:,:]  = lbl[j].reshape( (512,512) )
            vis_np[j,:,:]    = vis[j].reshape( (512,512) )
            fvis_np[j,0,:,:] = vis[j].reshape( (512,512) )

        input1      = torch.from_numpy(img_np).cuda()
        input2      = torch.from_numpy(img2_np).cuda()
        target_flow = torch.from_numpy(lbl_np).cuda()
        target_vis  = torch.from_numpy(vis_np).cuda()
        floatvis    = torch.from_numpy(fvis_np).cuda()

        #print "train: ", input1.sum(), input2.sum()
        # measure data formatting time
        format_time.update(time.time() - end)
        
        input1_var = torch.autograd.Variable(input1)
        input2_var = torch.autograd.Variable(input2)
        target_flow_var = torch.autograd.Variable(target_flow)
        target_vis_var = torch.autograd.Variable(target_vis)
        floatvis_var= torch.autograd.Variable(floatvis)
        
        # compute output
        end = time.time()
        output_flow, output_vis = model.forward(input1_var, input2_var)
        loss1 = criterion1(output_flow, target_flow_var, floatvis_var)
        loss2 = criterion2(output_vis, target_vis_var)
        loss = loss1 + lmbd*loss2
        #loss = loss2 + lmbd*loss1

        # measure accuracy and record loss
        prec1_vis = myfunc.accuracy_vis(output_vis.data, target_vis)
        prec1_flow = []
        prec1_flow.append( myfunc.accuracy_flow(output_flow.data, target_flow, floatvis, 2)  )
        prec1_flow.append( myfunc.accuracy_flow(output_flow.data, target_flow, floatvis, 5) )
        prec1_flow.append( myfunc.accuracy_flow(output_flow.data, target_flow, floatvis, 10) )
        prec1_flow.append( myfunc.accuracy_flow(output_flow.data, target_flow, floatvis, 15) )

        losses.update(loss.data[0], input1.size(0))
        top1_vis.update(prec1_vis[2], input1.size(0))
        top1_flow.update(prec1_flow[0], input1.size(0))
        for k,acc in enumerate(prec1_vis):
            acc_list_vis[k].update( acc )
        for k,acc in enumerate(prec1_flow):
            acc_list_flow[k].update( acc )

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_time.update(time.time()-end)

        # measure elapsed time
        batch_time.update(time.time() - batchstart)
        '''
        if(i*(1+epoch)%1000==0):
            for name, param in model.named_parameters():
                if 'bn' not in name:
                    name = name.replace('.', '/')
                    writer.add_histogram(name, param.data, i*(1+epoch), 25)
                    #writer.add_histogram(name+'/grad', param.grad.data, epoch)
        '''
        
        if i%print_freq==0:
            status = (epoch,i,nbatches,
                      batch_time.val,batch_time.avg,
                      data_time.val,data_time.avg,
                      format_time.val,format_time.avg,
                      train_time.val,train_time.avg,
                      losses.val,losses.avg,
                      top1_vis.val,top1_vis.avg,
                      top1_flow.val,top1_flow.avg)
            print "Iter: [%d][%d/%d]\tTime %.3f (%.3f)\tData %.3f (%.3f)\tFormat %.3f (%.3f)\tTrain %.3f (%.3f)\tLoss %.3f (%.3f)\tvisPrec@1 %.3f (%.3f)\tflowPrec@1 %.3f (%.3f)"%status

    writer.add_scalar('data/train_loss', losses.avg, epoch )
    writer.add_scalars('data/train_accuracy_vis', {'vis 0': acc_list_vis[0].avg,
                                                   'vis 1': acc_list_vis[1].avg,
                                                   'vis tot': acc_list_vis[2].avg}, epoch )
    writer.add_scalars('data/train_accuracy_flow', {'2pix': acc_list_flow[0].avg,
                                                    '5pix': acc_list_flow[1].avg,
                                                    '10pix': acc_list_flow[2].avg,
                                                    '15pix': acc_list_flow[3].avg}, epoch )
    
    return losses.avg,top1_vis.avg, top1_flow.avg


def validate(val_loader, model, criterion1, criterion2, lmbd, nbatches, epoch, print_freq):

    global writer
    
    batch_time = myfunc.AverageMeter()
    losses = myfunc.AverageMeter()
    top1_vis = myfunc.AverageMeter()
    top1_flow = myfunc.AverageMeter()

    acc_list_vis = []
    for i in range(4):
        acc_list_vis.append( myfunc.AverageMeter() )
    acc_list_flow = []
    for i in range(4):
        acc_list_flow.append( myfunc.AverageMeter() )
    
    # switch to evaluate mode
    model.eval()
    #debug: change to train
    #model.train()

    end = time.time()
    for i in range(0,nbatches):
        data = val_loader[i]
        #data = val_loader #debug
        img  = data["imageYtest"]
        img2 = data["imageUtest"]
        lbl  = data["labeltest"]
        vis  = data["matchtest"]
        #img  = data["imageY"]
        #img2 = data["imageU"]
        #lbl  = data["label"]
        #vis  = data["match"]

        img_np  = np.zeros( (img.shape[0], 1, 512, 512), dtype=np.float32 )
        img2_np = np.zeros( (img2.shape[0], 1, 512, 512), dtype=np.float32 )
        lbl_np  = np.zeros( (lbl.shape[0], 1, 512, 512 ), dtype=np.float32 )
        vis_np  = np.zeros( (vis.shape[0], 512, 512 ), dtype=np.int )
        fvis_np  = np.zeros( (vis.shape[0], 1, 512, 512 ), dtype=np.float32 )

        for j in range(img.shape[0]):
            img_np[j,0,:,:]  = img[j].reshape( (512,512) )
            img2_np[j,0,:,:] = img2[j].reshape( (512,512) )
            lbl_np[j,0,:,:]  = lbl[j].reshape( (512,512) )
            vis_np[j,:,:]    = vis[j].reshape( (512,512) )
            fvis_np[j,0,:,:] = vis[j].reshape( (512,512) )
            
        input1      = torch.from_numpy(img_np).cuda()
        input2      = torch.from_numpy(img2_np).cuda()
        target_flow = torch.from_numpy(lbl_np).cuda()
        target_vis  = torch.from_numpy(vis_np).cuda()
        floatvis    = torch.from_numpy(fvis_np).cuda()

        #print "valid: ", input1.sum(), input2.sum()
        input1_var = torch.autograd.Variable(input1, volatile=True)
        input2_var = torch.autograd.Variable(input2, volatile=True)
        target_flow_var = torch.autograd.Variable(target_flow, volatile=True)
        target_vis_var = torch.autograd.Variable(target_vis, volatile=True)
        floatvis_var = torch.autograd.Variable(floatvis, volatile=True)
        
        # compute output
        output_flow, output_vis = model.forward(input1_var, input2_var)
        loss1 = criterion1(output_flow, target_flow_var, floatvis_var)
        loss2 = criterion2(output_vis, target_vis_var)
        loss = loss1 + lmbd*loss2
        #loss = loss2 + lmbd*loss1
        
        # measure accuracy and record loss
        prec1_vis = myfunc.accuracy_vis(output_vis.data, target_vis)
        prec1_flow = []
        prec1_flow.append( myfunc.accuracy_flow(output_flow.data, target_flow, floatvis, 2)  )
        prec1_flow.append( myfunc.accuracy_flow(output_flow.data, target_flow, floatvis, 5) )
        prec1_flow.append( myfunc.accuracy_flow(output_flow.data, target_flow, floatvis, 10) )
        prec1_flow.append( myfunc.accuracy_flow(output_flow.data, target_flow, floatvis, 15) )

        losses.update(loss.data[0], input1.size(0))
        top1_vis.update(prec1_vis[2], input1.size(0))
        top1_flow.update(float(prec1_flow[0]), input1.size(0))
        for k,acc in enumerate(prec1_vis):
            acc_list_vis[k].update( acc )
        for k,acc in enumerate(prec1_flow):
            acc_list_flow[k].update( acc )
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            status = (i,nbatches,batch_time.val,batch_time.avg,losses.val,losses.avg,top1_vis.val,top1_vis.avg,top1_flow.val,top1_flow.avg)
            print "Test: [%d/%d]\tTime %.3f (%.3f)\tLoss %.3f (%.3f)\tvisPrec@1 %.3f (%.3f)\tflowPrec@1 %.3f (%.3f)"%status

    writer.add_scalar( 'data/valid_loss', losses.avg, epoch )
    writer.add_scalars('data/valid_accuracy_vis', {'vis 0': acc_list_vis[0].avg,
                                                   'vis 1': acc_list_vis[1].avg,
                                                   'vis tot': acc_list_vis[2].avg}, epoch )
    writer.add_scalars('data/valid_accuracy_flow', {'2pix': acc_list_flow[0].avg,
                                                    '5pix': acc_list_flow[1].avg,
                                                    '10pix': acc_list_flow[2].avg,
                                                    '15pix': acc_list_flow[3].avg}, epoch )

    print "Test:Result* visPrec@1 %.3f\tflowPrec@1 %.3f\tLoss %.3f"%(top1_vis.avg, top1_flow.avg, losses.avg)

    return float(top1_vis.avg), float(top1_flow.avg)


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
