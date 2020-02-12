import numpy as np 
import scipy.stats as stat
import matplotlib.pyplot as plt
import math as math

class Model:
    # vectorized likelihood. X is a numpy array where each column corresponds
    # to a model parameter

    def __init__(self): raise NotImplementedError
    def LogLikelihood(self,X): raise NotImplementedError
    def LogPrior(self,X): raise NotImplementedError
    def rPrior(self,n): raise NotImplementedError
    def Density(self,X): raise NotImplementedError

    
class Inference :

    def __init__(self,model):
        self.model = model

    def move_part(self,v,h):
        # v is an array Nsamples x par
        nparams = v.ndim
        npart = v.shape[0]
 
        ll0=h*self.model.LogLikelihood(v) + self.model.LogPrior(v)
        fac=np.random.gamma(1000,0.001,npart*nparams)

        #note: reshape by default is row-major.
        #      sum(X,axis=0) is colsum
        np.reshape(fac,(npart,nparams))
        v_new = v*fac 
        ll1=h*self.model.LogLikelihood(v_new) + self.model.LogPrior(v_new)
        alpha = ll1-ll0 + np.sum(np.reshape(stat.gamma.logpdf(1.0/fac,1000,scale=0.001) - stat.gamma.logpdf(fac,1000,scale=0.001),(npart,nparams)),1)
        u=np.random.uniform(size=npart)
        v2=v
        mask=np.log(u)<alpha
        v2[mask]=v_new[mask]
        return(v2)

    def IS(self,npart):
        sample=self.model.rPrior(npart)
        weights=self.model.LogLikelihood(sample)
        maxw=np.max(weights)

        w2 = np.exp(weights-maxw)
        w2_sum = np.sum(w2)

        ESS=1.0/(np.sum((w2/w2_sum)**2))

        posterior_means = np.average(sample,0,w2/w2_sum)

        self.weighted_samples = [sample, w2/w2_sum]
        self.LML = maxw + np.log(np.sum(np.exp(weights-maxw)))-np.log(npart)
        self.posterior_means = posterior_means
        self.ESS = ESS

    def SMC(self,npart,steps=11):

        protocol=np.linspace(0,1,steps)
        LogML=0

        sample=self.model.rPrior(npart)

        W = np.repeat(1.0/npart,npart)
        logW=-np.log(npart)

        k=0

        ESS=npart

        for h in protocol[1:]:
        
            ESS=1.0/np.sum(W**2)
            if ESS<npart/2.0:
            #if True:
                if sample.ndim==1 :
                    sample=np.random.choice(sample,npart,True,W)
                else:
                    ancestors = np.random.choice(np.arange(npart),npart,True,W)
                    sample=sample[ancestors,:]
            
                W=np.repeat(1.0/npart,npart)
                logW=-np.log(npart)

            delta=protocol[k+1]-protocol[k]
            log_w_inc = delta*self.model.LogLikelihood(sample)  
            log_w_un = log_w_inc + logW

            lwun_max=np.max(log_w_un)

            W = np.exp(log_w_un-lwun_max)
            W = W/np.sum(W)

            logML_inc = lwun_max+np.log(np.sum(np.exp(log_w_un-lwun_max)))
            logW=log_w_un-logML_inc

            LogML=LogML+logML_inc
        
        
            sample=self.move_part(sample,h)

            k=k+1
            
        posterior_means = np.average(sample,0,W)

        self.weighted_samples = [sample, W]
        self.LML = LogML 
        self.posterior_means = posterior_means
        self.ESS = ESS

    def plot_samples(self,Range=None):

        if Range is None:
            plt.hist(self.weighted_samples[0],
                weights=self.weighted_samples[1],
                density=1)
        else :
            bins=np.linspace(Range[0],Range[1],20)
            plt.hist(self.weighted_samples[0],
                weights=self.weighted_samples[1],
                density=1,
                bins=bins)

        plt.show()


