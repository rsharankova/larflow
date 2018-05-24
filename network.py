import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


class BasicBlockResNet(nn.Module):

    def __init__(self, inplanes, planes, stride=1, pad=1):
        super(BasicBlockResNet, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.bypass = None
        self.bnpass = None
        if inplanes!=planes or stride>1:
            self.bypass = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bnpass = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
            
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
                    
        if self.bypass is not None:
                outbp = self.bypass(x)
                outbp = self.bnpass(outbp)
                out += outbp
        else:
                out += x
            
        out = self.relu(out)

        return out
class DoubleResNet(nn.Module):
    def __init__(self,inplanes,planes,stride=1):
        super(DoubleResNet,self).__init__()
        self.res1 = BasicBlockResNet(inplanes,planes,stride)
        self.res2 = BasicBlockResNet(  planes,planes,     1)
        
    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        return out
                                                                                
    
class BasicBlockConv(nn.Module):

    def __init__(self, inplanes, planes, stride=1, pad=1):
        super(BasicBlockConv, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        #self.relu2 = nn.ReLU(inplace=True)

            
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        return out

class BasicBlockDeconv(nn.Module):

    def __init__(self, d_inplanes, d_planes, stride=1, pad=1):
        super(BasicBlockDeconv, self).__init__()
        self.stride = stride
        self.deconv1 = nn.ConvTranspose2d( d_inplanes, d_planes, kernel_size=3, stride=stride, padding=pad,  bias=False )
        self.bn1 = nn.BatchNorm2d(d_planes)
        self.relu1 = nn.ReLU(inplace=True)
        #self.res    = DoubleResNet(res_outplanes+deconv_outplanes,res_outplanes,stride=1)
        
    def forward(self, x, outsize):
        out = self.deconv1(x, output_size=outsize.size())
        out = self.bn1(out)
        out = self.relu1(out)
        #out = torch.cat( [out,outsize], 1 )
        #out = self.res(out)
        return out



class TestNet(nn.Module):

    def __init__(self, blockconv, blockdeconv,  num_classes=1000, input_channels=1, showsizes=False):
        self.inplanes = 16
        super(TestNet, self).__init__()
        self._showsizes = showsizes # print size at each layer

        #encoding
        self.layer0 = self._make_clayer(blockconv, input_channels, self.inplanes*1,  stride=1, pad=1) #1->16
        self.layer1 = self._make_clayer(blockconv, self.inplanes*1, self.inplanes*2,  stride=2, pad=1) #16->32
        self.layer2 = self._make_clayer(blockconv, self.inplanes*2, self.inplanes*2,  stride=1, pad=1) #32->32
        self.layer3 = self._make_clayer(blockconv, self.inplanes*2, self.inplanes*4,  stride=2, pad=1) #32->64
        self.layer4 = self._make_clayer(blockconv, self.inplanes*4, self.inplanes*4,  stride=1, pad=1) #64->64
        self.layer5 = self._make_clayer(blockconv, self.inplanes*4, self.inplanes*8,  stride=2, pad=1) #64->128
        self.layer6 = self._make_clayer(blockconv, self.inplanes*8, self.inplanes*8,  stride=1, pad=1) #128->128
        self.layer7 = self._make_clayer(blockconv, self.inplanes*8, self.inplanes*16,  stride=2, pad=1)#128->256

        # decoding flow
        self.layer8  = self._make_clayer(blockconv, self.inplanes*32, self.inplanes*32,  stride=1, pad=1)#256+256->512
        self.layer9  = self._make_dlayer(blockdeconv, self.inplanes*32, self.inplanes*16,  stride=2, pad=1)#512->256
        self.layer10 = self._make_clayer(blockconv, self.inplanes*16, self.inplanes*16,  stride=1, pad=1)#256->256
        self.layer11 = self._make_dlayer(blockdeconv, self.inplanes*16, self.inplanes*8,  stride=2, pad=1)#256->128
        self.layer12 = self._make_clayer(blockconv, self.inplanes*8, self.inplanes*8,  stride=1, pad=1)#128->128
        self.layer13 = self._make_dlayer(blockdeconv, self.inplanes*8, self.inplanes*4,  stride=2, pad=1)#128->64
        self.layer14 = self._make_clayer(blockconv, self.inplanes*4, self.inplanes*4,  stride=1, pad=1)#64->64
        self.layer15 = self._make_dlayer(blockdeconv, self.inplanes*4, self.inplanes*2,  stride=2, pad=1)#64->32
        #self.layer16 = self._make_clayer(blockconv, self.inplanes*2, num_classes,  stride=1, pad=1)#32->1

        # decoding vis        
        self.layer17 = self._make_clayer(blockconv, self.inplanes*32, self.inplanes*16,  stride=1, pad=1)#256+256->256
        self.layer18 = self._make_dlayer(blockdeconv, self.inplanes*16, self.inplanes*8,  stride=2, pad=1)#256->128
        self.layer19 = self._make_clayer(blockconv, self.inplanes*8, self.inplanes*8,  stride=1, pad=1)#128->128
        self.layer20 = self._make_dlayer(blockdeconv, self.inplanes*8, self.inplanes*4,  stride=2, pad=1)#128->64
        self.layer21 = self._make_clayer(blockconv, self.inplanes*4, self.inplanes*4,  stride=1, pad=1)#64->64
        self.layer22 = self._make_dlayer(blockdeconv, self.inplanes*4, self.inplanes*2,  stride=2, pad=1)#64->32
        self.layer23 = self._make_clayer(blockconv, self.inplanes*2, self.inplanes*2,  stride=1, pad=1)#32->32
        self.layer24 = self._make_dlayer(blockdeconv, self.inplanes*2, self.inplanes*1,  stride=2, pad=1)#32->16
        #self.layer25 = self._make_clayer(blockconv, self.inplanes, 2,  stride=1, pad=1)#16->2
        
        #1x1 decode
        self.deconv1 = nn.Conv2d( self.inplanes*2, 1, kernel_size=1, stride=1, padding=0, bias=False )
        self.deconv2 = nn.Conv2d( self.inplanes, 2, kernel_size=1, stride=1, padding=0, bias=False )
        self.softmax = nn.LogSoftmax(dim=1)

        #unused
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.avgpool = nn.AvgPool2d(3, stride=2)
        #self.dropout = nn.Dropout2d(p=0.5,inplace=True)
        #print "block.expansion=",block.expansion
        #self.fc = nn.Linear(16 , num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_clayer(self, block, inplanes, planes, stride=1, pad=1):

        layers = []
        for i in range(0, 1):
            layers.append(block(inplanes, planes, stride, pad))

        #return nn.Sequential(*layers)
        return block(inplanes, planes, stride, pad)
    
    def _make_dlayer(self, block, inplanes, planes, stride=1, pad=1):

        layers = []
        for i in range(0, 1):
            layers.append(block(inplanes, planes, stride, pad))

        #return nn.Sequential(*layers)
        return block(inplanes, planes, stride, pad)

    def forward(self, x, y):

        x0 = self.layer0(x)    
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        '''
        if self._showsizes:
            print "input: ", x.size()
            print "after encoding: "
            print "  x0: ",x0.size()
            print "  x1: ",x1.size()
            print "  x2: ",x2.size()
            print "  x3: ",x3.size()
            print "  x4: ",x4.size()
            print "  x5: ",x5.size()
            print "  x6: ",x6.size()
            print "  x7: ",x7.size()
        '''
        
        y0 = self.layer0(y)
        y1 = self.layer1(y0)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        y5 = self.layer5(y4)
        y6 = self.layer6(y5)
        y7 = self.layer7(y6)
        '''
        if self._showsizes:
            print "after encoding: "
            print "  y0: ",y.size()
            print "  y1: ",y1.size()
            print "  y2: ",y2.size()
            print "  y3: ",y3.size()
            print "  y4: ",y4.size()
            print "  y5: ",y5.size()
            print "  y6: ",y6.size()
            print "  y7: ",y7.size()
        '''

        z = torch.cat( (x7,y7) , 1)
        zz = z    #for vis
        if self._showsizes:
            print "after concat, z: ",z.size()

        z = self.layer8(z)
        z = self.layer9(z,x6)
        z = self.layer10(z)
        z = self.layer11(z,x4)
        z = self.layer12(z)
        z = self.layer13(z,x2)
        z = self.layer14(z)
        z = self.layer15(z,x0)
        #z = self.layer16(z)        
        z = self.deconv1(z)
        #print "z after decoding: ",z.size()
        #z = z.view(z.size(0),-1)
        #print z.size()

        zz1 = self.layer17(zz)
        zz2 = self.layer18(zz1,x6)
        zz3 = self.layer19(zz2)
        zz4 = self.layer20(zz3,x4)
        zz5 = self.layer21(zz4)
        zz6 = self.layer22(zz5,x2)
        zz7 = self.layer23(zz6)
        zz8 = self.layer24(zz7,x0)
        #zz9 = self.layer25(zz8)
        zz = self.deconv2(zz8)
        zz = self.softmax(zz)
        
        '''
        if self._showsizes:
            print "after decoding: "
            print "  z1: ",zz1.size()
            print "  z2: ",zz2.size()
            print "  z3: ",zz3.size()
            print "  z4: ",zz4.size()
            print "  z5: ",zz5.size()
            print "  z6: ",zz6.size()
            print "  z7: ",zz7.size()
            print "  z8: ",zz8.size()
            print "  z9: ",zz9.size()
        '''

        return z, zz


def mymodel( **kwargs):

    model = TestNet(BasicBlockConv, BasicBlockDeconv, **kwargs)

    return model

