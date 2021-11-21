import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from model import Unet
import onnx
from onnx.defs import onnx_opset_version

model = Unet((256,1,1250))
path = 'model/final.pt'
checkpoint = torch.load(path, map_location='cpu')
model.load_state_dict(checkpoint['model'])


torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "model/onnx.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['inter.0.conv1'],   # the model's input names
                  output_names = ['de9.2'])