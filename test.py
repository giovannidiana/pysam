import pysam

import numpy as np 
import scipy.stats as stat
import matplotlib.pyplot as plt
import math as math

class powerlaw(pysam.Model):
    def __init__(self,a,b):
        self.a=a
        self.b=b
        #self.data=None
   
    def LoadData(self,X):
        self.data=X
        self.M=X.shape[0]

    def LogPrior(self,x):
        return(stat.norm.logpdf(x,1,3))

    def rPrior(self,n):
        return(np.random.normal(1,3,n))
    
    def LogLikelihood(self,lam):
        zetamat=np.power.outer(1.0/np.arange(self.a,self.b+1),lam)
        zeta=np.sum(zetamat,0)
        nprod=-lam*np.sum(np.log(self.data))
        norm=-self.M*np.log(zeta)
        loglik=nprod+norm
        return(loglik) 

data=np.load("/home/giovanni/workspace/Dominic/PTZ-WILDTYPE-02_2photon_sess-01-6dpf_BLN_run-01_0.590bin0.10nnbav.npy")
sizes=data[0,:]

pl=powerlaw(min(sizes),max(sizes))
pl.LoadData(sizes)

I = pysam.Inference(pl)
