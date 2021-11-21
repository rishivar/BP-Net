import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from model import Unet
import re
import dataloader

bs = 256

model = Unet((256,1,1250)).cuda()
path = 'model/final.pt'
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model'])

pick_path = 'output.p'

test = torch.utils.data.DataLoader(BPdatasetv2(0, train = False, val = False,  test = True), batch_size=bs)

temp1 = []
model.eval()
with torch.no_grad():
    for idx,(inputs,labels) in tqdm(enumerate(test),total=len(test),  disable=True):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs_v= model(inputs).cuda()

        temp1.extend(outputs_v)

temp1 = torch.stack(temp1)    
with open(pick_path,'wb') as f:
    pickle.dump(temp1.cpu().detach().numpy(), f)