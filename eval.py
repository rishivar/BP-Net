import pickle
import os
from tqdm import tqdm
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
sns.set()
import dataloader


def evaluate_BHS_Standard(filename):
    """
        Evaluates PPG2ABP based on
        BHS Standard Metric
    """

    def BHS_metric(err):
        """
        Computes the BHS Standard metric

        Arguments:
            err {array} -- array of absolute error

        Returns:
            tuple -- tuple of percentage of samples with <=5 mmHg, <=10 mmHg and <=15 mmHg error
        """

        leq5 = 0
        leq10 = 0
        leq15 = 0

        for i in range(len(err)):

            if(abs(err[i]) <= 5):
                leq5 += 1
                leq10 += 1
                leq15 += 1

            elif(abs(err[i]) <= 10):
                leq10 += 1
                leq15 += 1

            elif(abs(err[i]) <= 15):
                leq15 += 1

        return (leq5*100.0/len(err), leq10*100.0/len(err), leq15*100.0/len(err))

    def calcError(Ytrue, Ypred, max_abp, min_abp, max_ppg, min_ppg):
        """
        Calculates the absolute error of sbp,dbp,map etc.

        Arguments:
            Ytrue {array} -- ground truth
            Ypred {array} -- predicted
            max_abp {float} -- max value of abp signal
            min_abp {float} -- min value of abp signal
            max_ppg {float} -- max value of ppg signal
            min_ppg {float} -- min value of ppg signal

        Returns:
            tuple -- tuple of abs. errors of sbp, dbp and map calculation
        """

        sbps = []
        dbps = []
        maps = []
        maes = []
        gt = []

        hist = []
    
       
        for i in (range(len(Ytrue))):
            y_t = Ytrue[i].ravel()
            y_p = Ypred[i].ravel()
            
            y_t = y_t * (max_abp - min_abp)
            y_p = y_p * (max_abp - min_abp) 
            
            dbps.append(abs(min(y_t)-min(y_p)))
            sbps.append(abs(max(y_t)-max(y_p)))
            maps.append(abs(np.mean(y_t)-np.mean(y_p)))

        return (sbps, dbps, maps)

    dt = pickle.load(open(os.path.join('data', 'test.p'), 'rb'))				# loading test data
    X_test = dt['X_test']
    Y_test = dt['Y_test']

    dt = pickle.load(open(os.path.join('data', 'meta.p'), 'rb'))				# loading meta data
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']
    
    Y_pred = pickle.load(open(filename, 'rb'))							# loading prediction

    (sbps, dbps, maps) = calcError(Y_test, Y_pred, max_abp, min_abp, max_ppg, min_ppg)   # compute errors

    sbp_percent = BHS_metric(sbps)											# compute BHS metric for sbp
    dbp_percent = BHS_metric(dbps)											# compute BHS metric for dbp
    map_percent = BHS_metric(maps)											# compute BHS metric for map

    print('----------------------------')
    print('|        BHS-Metric        |')
    print('----------------------------')

    print('----------------------------------------')
    print('|     | <= 5mmHg | <=10mmHg | <=15mmHg |')
    print('----------------------------------------')
    print('| DBP |  {} %  |  {} %  |  {} %  |'.format(round(dbp_percent[0], 2), round(dbp_percent[1], 2), round(dbp_percent[2], 2)))
    print('| MAP |  {} %  |  {} %  |  {} %  |'.format(round(map_percent[0], 2), round(map_percent[1], 2), round(map_percent[2], 2)))
    print('| SBP |  {} %  |  {} %  |  {} %  |'.format(round(sbp_percent[0], 2), round(sbp_percent[1], 2), round(sbp_percent[2], 2)))
    print('----------------------------------------')  


def evaluate_AAMI_Standard(filename):
    """
        Evaluate PPG2ABP using AAMI Standard metric	
    """

    def calcErrorAAMI(Ypred, Ytrue, max_abp, min_abp, max_ppg, min_ppg):
        """
        Calculates error of sbp,dbp,map for AAMI standard computation

        Arguments:
            Ytrue {array} -- ground truth
            Ypred {array} -- predicted
            max_abp {float} -- max value of abp signal
            min_abp {float} -- min value of abp signal
            max_ppg {float} -- max value of ppg signal
            min_ppg {float} -- min value of ppg signal

        Returns:
            tuple -- tuple of errors of sbp, dbp and map calculation
        """

        sbps = []
        dbps = []
        maps = []

        for i in (range(len(Ytrue))):
            y_t = Ytrue[i].ravel()
            y_p = Ypred[i].ravel()

            y_t = y_t * (max_abp - min_abp) 
            y_p = y_p * (max_abp - min_abp) 

            dbps.append(min(y_p)-min(y_t))
            sbps.append(max(y_p)-max(y_t))
            maps.append(np.mean(y_p)-np.mean(y_t))

        return (sbps, dbps, maps)

    dt = pickle.load(open(os.path.join('data', 'test.p'), 'rb'))			# loading test data
    X_test = dt['X_test']
    Y_test = dt['Y_test']

    dt = pickle.load(open(os.path.join('data', 'meta.p'), 'rb'))			# loading metadata
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    Y_pred = pickle.load(open(filename, 'rb'))						# loading prediction

    (sbps, dbps, maps) = calcErrorAAMI(Y_test, Y_pred, max_abp, min_abp, max_ppg, min_ppg)		# compute error

    print('---------------------')
    print('|   AAMI Standard   |')
    print('---------------------')

    print('-----------------------')
    print('|     |  ME   |  STD  |')
    print('-----------------------')
    print('| DBP | {} | {} |'.format(round(np.mean(dbps), 3), round(np.std(dbps), 3)))
    print('| MAP | {} | {} |'.format(round(np.mean(maps), 3), round(np.std(maps), 3)))
    print('| SBP | {} | {} |'.format(round(np.mean(sbps), 3), round(np.std(sbps), 3)))
    print('-----------------------')

def evaluate_metrics(filename, i):   
    def calcError(Ytrue, Ypred, max_abp, min_abp, max_ppg, min_ppg):
        sbp_t = []
        sbp_p = []
        dbp_t = []
        dbp_p = []
        map_t = []
        map_p = []
        
        x = 0
        y = 0
        
        
        for i in (range(len(Ytrue))):
            y_t = Ytrue[i].ravel()
            y_p = Ypred[i].ravel()
            
            y_t = y_t * (max_abp - min_abp) 
            y_p = y_p * (max_abp - min_abp) 
            
            sbp_p.append(abs(max(y_p)))
            dbp_p.append(abs(min(y_p)))
            map_p.append(abs(np.mean(y_p)))
            sbp_t.append(abs(max(y_t)))
            dbp_t.append(abs(min(y_t)))
            map_t.append(abs(np.mean(y_t)))
            
        
        print("SBP")
        print("Mean Absolute Error : ", round(mean_absolute_error(sbp_t, sbp_p), 3))
        print("Root Mean Squared Error : ", round(mean_squared_error(sbp_t, sbp_p, squared=False),3))
        print("R2 : ", r2_score(sbp_t, sbp_p))
              
        print("")
        
        print("DBP")
        print("Mean Absolute Error : ", round(mean_absolute_error(dbp_t, dbp_p),3))
        print("Root Mean Squared Error : ", round(mean_squared_error(dbp_t, dbp_p, squared=False),3))
        print("R2 : ", r2_score(dbp_t, dbp_p))
        
        print("")
        
        print("MAP")
        print("Mean Absolute Error : ", mean_absolute_error(map_t, map_p))
        print("Root Mean Squared Error : ", round(mean_squared_error(map_t, map_p, squared=False), 2))
        print("R2 : ", r2_score(map_t, map_p))
        
        print("------------------------------------------------------------------------")
        
    dt = pickle.load(open(os.path.join('data', 'test.p'), 'rb'))				# loading test data
    X_test = dt['X_test']
    Y_test = dt['Y_test']
    
    dt = pickle.load(open(os.path.join('data', 'meta.p'), 'rb'))				# loading meta data
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']
    

    Y_pred = pickle.load(open(filename, 'rb'))							# loading prediction
    calcError(Y_test, Y_pred, max_abp, min_abp, max_ppg, min_ppg)


evaluate_BHS_Standard('model/final.p')
evaluate_AAMI_Standard('model/final.p')
evaluate_metrics('model/final.p', 4)