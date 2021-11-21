import torch
import torch.nn as nn


class IncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size = 15, stride = 1, padding = 7):
        super(IncBlock,self).__init__()
        
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias = False)
        
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = size, stride = stride, padding = padding ),
                                   nn.BatchNorm1d(out_channels//4))
        
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
                                   nn.BatchNorm1d(out_channels//4),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size +2 , stride = stride, padding = padding + 1),
                                   nn.BatchNorm1d(out_channels//4))
        
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
                                   nn.BatchNorm1d(out_channels//4),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size + 4 , stride = stride, padding = padding + 2),
                                   nn.BatchNorm1d(out_channels//4))
        
        
        self.conv4 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
                                   nn.BatchNorm1d(out_channels//4),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size + 6 , stride = stride, padding = padding + 3),
                                   nn.BatchNorm1d(out_channels//4))
        self.relu = nn.ReLU()
    def forward(self,x):
        res = self.conv1x1(x)
#         print (res.size())

        
        c1 = self.conv1(x)
#         print (c1.size())
        
        c2 = self.conv2(x)
#         print (c2.size())
                
        c3 = self.conv3(x)
#         print (c3.size())
        
        c4 = self.conv4(x)
#         print (c4.size())
        
        concat = torch.cat((c1,c2,c3,c4),dim = 1)
        
        concat+=res
#         print (concat.shape)
        return self.relu(concat)



        
class InterAxialBlock(nn.Module):
        #3
  def __init__(self,in_channels = 1, out_channels = 1):
    
    super(InterAxialBlock, self).__init__()
    
    self.conv1 = nn.Conv1d(in_channels,8,3)
    self.bn1 = nn.BatchNorm1d(8)
    
    self.conv2 = nn.Conv1d(8,16,3)
    self.bn2 =nn.BatchNorm1d(16)
    
    self.conv3 = nn.Conv2d(1,1,(3,3), 2)
    self.bn3 = nn.BatchNorm2d(1)
    
    self.conv4 = nn.Conv2d(1, 1, (3,15), padding = (0,7))
    self.bn4 = nn.BatchNorm2d(1)
    
    self.conv5 = nn.Conv1d(1,out_channels,3, padding = 1)
    self.bn5 = nn.BatchNorm1d(out_channels)
    self.relu1 = nn.LeakyReLU(0.2)
    
    
    self.mp1 = nn.MaxPool1d(2)
    self.mp2 = nn.MaxPool2d((2,2))
    

    

  def forward(self, x):
    
#     print("in Inter",x.shape)
    x = self.relu1(self.bn1(self.conv1(x)))

    x = self.relu1(self.bn2(self.conv2(x)))
#3d -> 4d 
    x = x.view(x.shape[0],1,x.shape[1],x.shape[2])

    x = self.relu1(self.bn3(self.conv3(x)))

    x = self.mp2(x)

    
    x = self.relu1(self.bn4(self.conv4(x)))

    
    x = torch.squeeze(x, dim = 1)
    x = self.relu1(self.bn5(self.conv5(x)))

   
    return x

class Unet(nn.Module):
    def __init__(self, shape):
        super(Unet, self).__init__()
        #1
        in_channels = 1
        
        self.inter = nn.Sequential(InterAxialBlock())
        
        self.en1 = nn.Sequential(nn.Conv1d(in_channels, 32, 3, padding = 1), 
                                nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2),
                                nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
                                IncBlock(32,32))
        
        self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2),
                                 nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
                                IncBlock(64,64))
        
              
        self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
                                IncBlock(128,128))
        
        self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(256, 256, 5, stride = 2, padding = 1),
                                IncBlock(256,256))
        
        
        self.en5 = nn.Sequential(nn.Conv1d(256,512, 3, padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(512,512))
        
        
        self.de1 = nn.Sequential(nn.ConvTranspose1d(512,256,1),
                               nn.BatchNorm1d(256),
                               nn.LeakyReLU(0.2),
                                IncBlock(256,256))
        
        self.de2 =  nn.Sequential(nn.Conv1d(512,256,3, padding = 1),
                               nn.BatchNorm1d(256),
                               nn.LeakyReLU(0.2),
                                  nn.ConvTranspose1d(256,128,3, stride = 2),
                                IncBlock(128,128))
        
        self.de3 =  nn.Sequential(nn.Conv1d(256,128,3, stride = 1, padding = 1),
                               nn.BatchNorm1d(128),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(128,64,3, stride = 2),
                                IncBlock(64,64))
        
        self.de4 =  nn.Sequential(nn.Conv1d(128,64,3, stride = 1, padding = 1),
                               nn.BatchNorm1d(64),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(64,32,3, stride = 2),
                                IncBlock(32,32))
        
        self.de5 = nn.Sequential(nn.Conv1d(64,32,3, stride = 1, padding = 1),
                               nn.BatchNorm1d(32),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(32,16,3, stride = 2),
                                IncBlock(16,16))
                               
        self.de6 = nn.Sequential(nn.ConvTranspose1d(16,8,2,stride =2),
                                nn.BatchNorm1d(8),
                                nn.LeakyReLU(0.2))

        self.de7 = nn.Sequential(nn.ConvTranspose1d(8,4,2,stride =2),
                                nn.BatchNorm1d(4),
                                nn.LeakyReLU(0.2))
        
        self.de8 = nn.Sequential(nn.ConvTranspose1d(4,2,1,stride =1),
                                nn.BatchNorm1d(2),
                                nn.LeakyReLU(0.2))
        
        self.de9 = nn.Sequential(nn.ConvTranspose1d(2,1,1,stride =1),
                                nn.BatchNorm1d(1),
                                nn.LeakyReLU(0.2))
        
        
    def forward(self,x):
        
#         print("Before inter ",x.shape)
        x = self.inter(x)
#         print(" After Inter",x.shape)
        
        x = nn.ConstantPad1d((1,1),0)(x)
#         print ("After ConstantPad1d",x.shape)
        e1 = self.en1(x)
#         print ("After e1 ",e1.shape)
        
        e2 = self.en2(e1)
#         print ("After e2 ",e2.shape)
        
        e3 = self.en3(e2)
#         print ("After e3 ",e3.shape)
        
        e4 = self.en4(e3)
#         print ("After e4  ",e4.shape)
        
        e5 = self.en5(e4)
#         print ("After e5 ",e5.shape)
#         print ("-----------------------------------------------------------------------------")
        d1 = self.de1(e5)
#         print ("After d1", d1.shape)
        
#         print("Before cat d1 e4 {} {}".format(d1.shape,e4.shape))
        cat = torch.cat([d1,e4],1)
#         print("After cat d1 e4 {}".format(cat.shape))
        
        d2 = self.de2(cat)
#         print ("After d2 ",d2.shape)
        
#         print ("Before cat d2 e3 {} {}  ".format(d2.shape,e3.shape))
        cat = torch.cat([d2,e3[:,:,:-1]],1)
#         print("After cat d2 e3 {}".format(cat.shape))
        
        
        
        d3 = self.de3(cat)
        
#         print ("After d3 ",d3.shape)
#         print ("Before cat d3 e2 {} {}  ".format(d3.shape,e2.shape))
#         print("-1 being done on d3")
        cat = torch.cat([d3,e2[:,:,:]],1) #MADE A CHANGE HERE, ADDED -1
#         print("After cat d3 e2 {}".format(cat.shape))
        
        d4 = self.de4(cat)
#         print ("After d4 ",d4.shape)
        
#         print ("Before cat d4 e1 {} {}  ".format(d4.shape,e1.shape))
        cat = torch.cat([d4[:,:,:-2],e1],1) #MADE A CHANGE HERE, ([d4[:,:,:-2],e1],1) this is the original one
#         print("After cat d4 e1 {}".format(cat.shape))
        
        d5 = self.de5(cat)[:,:,:-2]
#         print ("After d5 ", d5.shape)
    
        d6 = self.de6(d5)[:,:,:-1]
        
#         print(d6.shape)
        
        d7 = self.de7(d6)
#         print("d7 ", d7.shape)
        d8 = self.de8(d7)
#         print(d8.shape)
        d9 = self.de9(d8)
#         print(d9.shape)
        return d9