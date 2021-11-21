import os
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from pytorchtools import EarlyStopping
from model import Unet
import dataloader

bs = 256
length = 1250
epochs = 300

train1 = torch.utils.data.DataLoader(BPdatasetv1(4, train = True), batch_size=bs)
val1 = torch.utils.data.DataLoader(BPdatasetv1(4, val = True), batch_size=bs)

model = Unet((256, 1, 1250)).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,200], gamma=0.1)
scaler = torch.cuda.amp.GradScaler()
early_stopping = EarlyStopping(patience=100, verbose=True)

best_loss = 1000

for epoch in range(epochs):
    model.train()
    print('epochs {}/{} '.format(epoch+1,epochs))

    running_loss = 0.0
    running_loss_v = 0.0

    for idx,(inputs, output) in tqdm(enumerate(train1),total=len(train1)):
        inputs = inputs.cuda()
        output = output.cuda()

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pred = model(inputs)
            loss = criterion(pred, output)

        scaler.scale(loss).backward()
        running_loss += loss
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()

    #VALIDATION
    model.eval()
    with torch.no_grad():
        for idx,(inputs_v,labels_v) in tqdm(enumerate(val1),total=len(val1)):
            inputs_v = inputs_v.cuda()
            labels_v = labels_v.cuda()
            outputs_v= model(inputs_v).cuda()
            loss_v = criterion(outputs_v,labels_v)
            running_loss_v += loss_v

    path = 'model/ssl.pt'

    if (running_loss_v/len(val1)) < best_loss:
        best_loss = running_loss_v/len(val1)
        out = torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_loss,
            'exp_dir':'model'
        }, f=path)
    print('loss : {:.4f}   val_loss : {:.4f}'.format((running_loss/len(train1)),(running_loss_v/len(val1))))  


    early_stopping(running_loss_v/len(val1), model)

    if early_stopping.early_stop:
        print("Early stopping")
        break