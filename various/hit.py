###
# Code found in https://github.com/rayohauno/hierpart
###

import numpy as np
import dill
from collections import defaultdict
from sklearn import metrics

def check(hp):
    """Checks if a hierarchical partition represented by nested lists is well formed or not. Namely, it ensures that each list either contain other list or elements, but not lists and elements at the same time.
    
    Examples:
    >>> hp=[[1],[2,3]]
    >>> check(hp)
    True
    >>> hp=[1,[2,3]]
    >>> check(hp)
    False
    """
    flag=None
    for chp in hp:
        if isinstance(chp,list):
            if flag is None:
                flag=True
            elif not flag:
                return False
            if not check(chp):
                return False            
        else:
            if flag is None:
                flag=False
            elif flag:
                return False
    return True

def flattenator(newick):
    """Takes a hierarchical partition represented by nested lists and return a list of all its elements.
    
    Example
    >>> hp = [[3, 4, 5, 6], [[0], [1, 2]], [[7], [8, 9]]]
    >>> sorted(flattenator(hp))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    for e in newick:
        if isinstance(e,list):
            for ee in flattenator(e):
                yield ee
        else:
            yield e

def xlnx(x):
    """Returns x*log(x) for x > 0 or returns 0 otherwise."""
    if x <= 0.:
        return 0.
    return x*np.log(x)

def HMI(Ut,Us):
    """
    Computes the hierarchical mutual information between two hierarchical partitions.
    
    Returns
    n_ts,HMI(Ut,Us) : where n_ts is the number of common elements between the hierarchical partitions Ut and Us.
    
    NOTE: We label by u,v the children of t,s respectively.
    
    Examples
    >>>"""
    if isinstance(Ut[0],list):
        if isinstance(Us[0],list):
            # Ut and Us are both internal nodes since they contain other lists.
            n_ts=0.
            H_uv=0.
            H_us=0.
            H_tv=0.
            mean_I_ts=0.0
            n_tv=defaultdict(float)            
            for Uu in Ut:
                n_us=0.
                for v,Uv in enumerate(Us):
                    n_uv,I_uv=HMI(Uu,Uv)
                    n_ts+=n_uv
                    n_tv[v]+=n_uv
                    n_us+=n_uv                    
                    H_uv+=xlnx(n_uv)
                    mean_I_ts+=n_uv*I_uv
                H_us+=xlnx(n_us)
            for _n_tv in n_tv.values():
                H_tv+=xlnx(_n_tv)
            if n_ts>0.:
                local_I_ts=np.log(n_ts)-(H_us+H_tv-H_uv)/n_ts
                mean_I_ts=mean_I_ts/n_ts
                I_ts=local_I_ts+mean_I_ts
                #print("... Ut =",Ut,"Us =",Us,"n_ts =",n_ts,"I_ts =",I_ts,"local_I_ts =",local_I_ts,"mean_I_ts =",mean_I_ts)
                return n_ts,I_ts
            else:
                #print("... Ut =",Ut,"Us =",Us,"n_ts =",0.0,"I_ts =",0.0)
                return 0.,0.
        else:
            # Ut is internal node and Us is leaf
            return len(set(flattenator(Ut))&set(Us)),0.
    else:
        if isinstance(Us,list):
            # Ut is leaf and Us internal node
            return len(set(flattenator(Us))&set(Ut)),0.          
        else:
            # Both Ut and Us are leaves
            return len(set(Ut)&set(Us)),0.
        

def HH(hp):
    """Returns the hierarchical entropy of a hierarchical partition.
    
    Note: this is not the most efficient implementation."""
    return HMI(hp,hp)[1]


def HJH(hp1,hp2):
    """Returns the hierarchical joint entropy between two hierarchical partitions."""
    return HH(hp1)+HH(hp2)-HMI(hp1,hp2)[1]

def mean_arit(x,y):
    return .5*(x+y)

def mean_geom(x,y):
    return np.sqrt(x*y)

def mean_max(x,y):
    return max(x,y)

def mean_min(x,y):
    return min(x,y)

def NHMI(hp1,hp2,generalized_mean=mean_arit):
    """Returns the normalized hierarchical mutual information.
    
    By default, it uses the arithmetic mean for normalization. However, another generalized mean can be provided if desired."""
    gm = generalized_mean(HH(hp1),HH(hp2))
    if gm > 0.:
        return HMI(hp1,hp2)[1]/gm
    return 0.

def HCH(hp1,hp2):
    """Returns the hierarchical conditional entropy HCH(hp1|hp2)."""
    return HJH(hp1,hp2)-HH(hp2)

def HVI(hp1,hp2):
    """Returns the hierarchical variation of information."""
    return HH(hp1)+HH(hp2)-2.0*HMI(hp1,hp2)[1]

class RunningMeanStd:
    """To compute mean, variance, etc. online as values are obtained without the need to store them. 
    This method is also numerically stable.
    See Donald Knuth’s Art of Computer Programming, Vol 2, page 232, 3rd edition."""
    def __init__(self):
        self._M=0.
        self._S=0.
        self._n=0
    def push(self,x):
        self._n+=1
        oldM=self._M
        self._M=self._M+(x-self._M)/float(self._n)
        self._S=self._S+(x-oldM)*(x-self._M)
    def mean(self):
        if self._n>0:
            return self._M
        return None
    def variance(self):
        if self._n>1:
            return self._S/float(self._n-1)
        return None
    def std(self):
        v=self.variance()
        if v is None:
            return None
        return np.sqrt(v)
    def sem(self):
        if self._n>1:
            return self.std()/np.sqrt(self._n)
        return None
    def rel_err(self,tol_std=.05):
        assert tol_std>0.
        if self._n>1:
            return self.sem()/(abs(self.mean())+tol_std)
        return None
    def clear(self):
        self._M=0.
        self._S=0.
        self._n=0
    def __len__(self):
        return self._n
    def __repr__(self):
        return "RunningMeanStd{n="+str(len(self))+" mean="+str(self.mean())+" std="+str(self.std())+" sem="+str(self.sem())+"}"

def replicate_hierarchical_partition(hp,d):
    """Replicates a hierarchical partition hp into a new one whose elements are interachanged by others as specified by the list or dictionary d."""
    if isinstance(hp,list):
        return [replicate_hierarchical_partition(chp, d) for chp in hp]
    else:
        return d[hp]

def EHMI(hp1,hp2,min_num_shufflings=36,rel_err_tol=0.01,verbose=0):
    """Returns the expected hierarchical mutual information using the permutation model as the reference null model.
    When verbose=0, then the function works silently. For verbose=1,2,3 some information is shown as the process run. The larger the value, the more information."""
    assert min_num_shufflings>0
    # np.random.seed(10)
    runms=RunningMeanStd()    
    rel_err=2.*rel_err_tol
    d=sorted(flattenator(hp1))
    while rel_err>rel_err_tol or len(runms)<min_num_shufflings:
        np.random.shuffle(d)
        rhp1=replicate_hierarchical_partition(hp1,d)
        HMI_sample=HMI(rhp1,hp2)[1]
        runms.push(HMI(rhp1,hp2)[1])        
        if len(runms)>1:
            rel_err=runms.rel_err()  
        if verbose>0:
            if verbose==1:
                print(HMI_sample)
            elif verbose==2:
                print(HMI_sample,runms.mean(),rel_err)
            elif verbose==3:
                print(HMI_sample,runms.mean(),rel_err)                
                print(rhp1)
    return runms