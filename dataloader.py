import pickle
import numpy as np

class BPdatasetv1(Dataset):
    def __init__(self, i, train = False, val = False):
        if train == True:
            dt = pickle.load(open(os.path.join('data','train4.p'),'rb'))
            self.input = np.swapaxes(dt['X_train'],1,2).astype('float32')
            self.output = np.swapaxes(dt['X_train'],1,2).astype('float32')
        elif val == True:
            dt = pickle.load(open(os.path.join('data','val4.p'),'rb'))
            self.input = np.swapaxes(dt['X_val'],1,2).astype('float32')
            self.output = np.swapaxes(dt['X_val'],1,2).astype('float32')
            
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        inp = self.input[idx]
        out = self.output[idx]
        return inp, out
    
class BPdatasetv2(Dataset):
    def __init__(self, i, train = False, val = False, test=False):
        if train == True:
            dt = pickle.load(open(os.path.join('data','train4.p'),'rb'))
            self.input = np.swapaxes(dt['X_train'],1,2).astype('float32')
            self.output = np.swapaxes(dt['Y_train'],1,2).astype('float32')
        elif val == True:
            dt = pickle.load(open(os.path.join('data','val4.p'),'rb'))
            self.input = np.swapaxes(dt['X_val'],1,2).astype('float32')
            self.output = np.swapaxes(dt['Y_val'],1,2).astype('float32')
        elif test == True:
            dt = pickle.load(open(os.path.join('data','test.p'),'rb'))
            self.input = np.swapaxes(dt['X_test'],1,2).astype('float32')
            self.output = np.swapaxes(dt['Y_test'],1,2).astype('float32')
            
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        inp = self.input[idx]
        out = self.output[idx]
        return inp, out 



