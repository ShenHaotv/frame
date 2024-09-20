import numpy as np
import gc
from copy import deepcopy
from sklearn.model_selection import KFold
from .lyapunov_helper import modified_singular_lyapnov

def getcontrast(o,b):
    contrast=np.zeros((o-1,o))
    for i in range (0,b-1):
        contrast[i,i]=1
        contrast[i,b-1]=-1
    for i in range (b-1,o-1):
        contrast[i,i+1]=1
        contrast[i,b-1]=-1
    return(contrast)

def run_cv(
    sp_digraph,
    lamb_grid,
    lamb_warmup=None,
    n_folds=None,
    factr=1e10,
    random_state=500,
    outer_verbose=True,
    inner_verbose=False,
    node_train_idxs=None,
    ):
    """Run cross-validation."""
    o=sp_digraph.S.shape[0]

    # default is None i.e., leave-one-out CV
    if n_folds is None:
       n_folds = o
        
    # setup cv indicies
    if node_train_idxs is None:
       node_train_idxs=setup_k_fold_cv(o=o,n_splits=n_folds,random_state=random_state)
       
    else:
        node_train_idxs=node_train_idxs

    if lamb_warmup is None:
       lamb_warmup=float(np.max(lamb_grid))
    # CV error
    n_lamb = len(lamb_grid)

    errs=np.zeros((n_folds,n_lamb))
    
    for fold in range(n_folds):      
        if outer_verbose:
            print("\n fold=", fold+1)
        
        node_train_idx=node_train_idxs[fold].astype(int)
                                  
        sp_digraph_warmup=copy_spatial_digraph(sp_digraph, node_train_idx)      #Warm up
        sp_digraph_warmup.fit(lamb=lamb_warmup,
                              factr=factr,
                              lb=-np.Inf,
                              ub=np.Inf,
                              verbose=inner_verbose)
        
        logm_init=np.log(sp_digraph_warmup.m)
        logc_init=np.log(sp_digraph_warmup.c)
             
        for i, lamb in enumerate(lamb_grid):                                    #Formal cv
            if outer_verbose:
               print("\riteration lambda={}/{}".format(i + 1, n_lamb),end="",)
               # fit on train set
            lamb= float(lamb)
            (errs[fold,i])=error(sp_digraph,
                                 node_train_idx,
                                 lamb=lamb,                                    
                                 factr=factr,
                                 verbose=inner_verbose,
                                 logm_init=logm_init,
                                 logc_init=logc_init,
                                 )           
             
    cv_err=np.sum(errs,axis=0)/o
     
    return (cv_err,node_train_idxs)

def setup_k_fold_cv(o,n_splits, random_state):
    """Setup cross-validation indicies.0

    Args:
        o:number of observed nodes
        n_splits (:obj:`int`): number of CV folds
        random_state (:obj:`int`): random seed

    Returns:
        node_train_idxs: the indexes of indexes of rows of h that are selected for
        training
    """     
    
    kf = KFold(
        n_splits=n_splits, random_state=random_state, shuffle=True
    )  # k-fold cv object

    node_train_idxs = [train_index for train_index,_ in kf.split(np.arange(o))]
    return  node_train_idxs


def copy_spatial_digraph(sp_digraph, node_train_idx):
    """Copy SpatialDiGraph object

    Args:
        sp_digraph (:obj:`SpatialDiGraph`): SpatialDiGraph class
        node_idx (:obj:`numpy.ndarray`): indexes of rows of h that are selected

    Returns:
        sp_digraph_copy (:obj:`SpatialGraph`): SpatialDiGraph class
    """
    sp_digraph_copy=deepcopy(sp_digraph)
    h0_copy=sp_digraph.h[0][node_train_idx]
    h1_copy=sp_digraph.h[1][node_train_idx]
    sp_digraph_copy.h=(h0_copy,h1_copy)
    sp_digraph_copy.frequencies=sp_digraph.frequencies[node_train_idx,:]
    sp_digraph_copy.S=sp_digraph_copy.frequencies@sp_digraph_copy.frequencies.T/sp_digraph.n_snps

    return sp_digraph_copy


def error(sp_digraph,
          node_train_idx,
          lamb,
          factr,
          verbose,
          logm_init=None,
          logc_init=None,
          ):
    """compute the error of the predict snps of the prediction"""
  
    sp_digraph_train=copy_spatial_digraph(sp_digraph, node_train_idx)    
    sp_digraph_train.fit(lamb=lamb,
                         factr=factr,
                         lb=-np.Inf,
                         ub=np.Inf,
                         verbose=verbose,
                         logm_init=logm_init,
                         logc_init=logc_init,
                         )

    L=sp_digraph_train.L
    d=L.shape[0]
    gamma=sp_digraph_train.gamma
    h=sp_digraph.h                                                             
    h0=h[0]                                                                    #Obseved nodes index in the total set
    h1=h[1]                                                                    #Number of samples in the total set
    o=len(h0)                                                                  #Number of observed nodes   
    n_snps=sp_digraph.n_snps
    
    Time=modified_singular_lyapnov(L,gamma,np.ones((d,d)),False)               #Pairwise coalesce time   
    T_ob=Time[h0[:, np.newaxis],h0]                                            #Pairwise coalesce time of the observed demes
    T_bar=T_ob                                                                 #This corresponds to T_bar in the paper
    for i in range (o):
        T_bar[i,i]=T_bar[i,i]*(1-1/h1[i])
        
    s=sorted(node_train_idx, key=lambda k: (h1[k], k), reverse=True)  
    b=s[0]                                                                    
    con=getcontrast(o,b)
    Cov=-con@T_bar@con.T                                                       #Covariance matrix 
    node_train_idx_con=node_train_idx[node_train_idx!=b]                                                                  
    node_train_idx_con=np.where(node_train_idx_con>b,                          #Index of observed nodes in the training set, after constrasting
                                node_train_idx_con-1,node_train_idx_con)
    node_test_idx_con=np.setdiff1d(np.arange(o-1),node_train_idx_con)
    Cov_train=Cov[node_train_idx_con[:, np.newaxis],node_train_idx_con]        #Covariance matrix of the training data
    Cov_test_train=Cov[node_test_idx_con[:, np.newaxis],node_train_idx_con]    #Covariance matrix between test data and training data
     
    frequencies_con=con@sp_digraph.frequencies                                 #Frequencies after contrasting
   
    frequencies_train_con=frequencies_con[node_train_idx_con,:]                #Frequencies of the training data
    frequencies_test_con=frequencies_con[node_test_idx_con,:]                  #Frequencies of the testing data

    term=np.linalg.solve(Cov_train,frequencies_train_con)
    prediction=Cov_test_train@term
     
    difference=prediction-frequencies_test_con
    n_snps = sp_digraph.n_snps
    err=np.sum(difference**2)/n_snps

    gc.collect()
              
    return err
