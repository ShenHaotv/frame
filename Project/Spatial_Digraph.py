from __future__ import absolute_import, division, print_function
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import fmin_l_bfgs_b
from Loss import getlaplacian,getcoalesce,loss_wrapper
from discreteMarkovChain import markovChain

class SpatialDiGraph(nx.DiGraph):
    def __init__(self,sp_graph):
        """Represents the spatial network which the data is defined on and
        stores relevant matrices / performs linear algebra routines needed for
        the model and optimization. Inherits from the networkx Graph object."""

        # inherits from networkx Graph object
        super(SpatialDiGraph, self).__init__()

        # init digraph
        self._init_digraph(sp_graph) 

        
    def _init_digraph(self,sp_graph):
        """Initialize the digraph and related digraph objects"""

        #Copy nodes from sp_graph
        self.add_nodes_from(sp_graph.nodes())

        self.sample_pos=sp_graph.sample_pos
        self.node_pos=sp_graph.node_pos
        self.frequencies=sp_graph.frequencies
        
        self.heterozygosity=2*self.frequencies*(1-self.frequencies)
        self.average_heterozygosity=np.mean(self.heterozygosity,axis=1)
       
        d=self.number_of_nodes()
        
        #Copy nodes attributes from sp_graph
        for i in range(d):
            self.nodes[i]["n_samples"]=sp_graph.nodes[i]["n_samples"]
               
        #Add directed edges
        for a, b in sp_graph.edges():
           self.add_edge(a, b)
           self.add_edge(b, a)
   
        self.adj=nx.adjacency_matrix(self)
        self.adj_base=sp.triu(nx.adjacency_matrix(self), k=1)
       
        #Copy the number of snps and the sample covariance matrix
        self.n_snps=sp_graph.n_snps
        self.S = sp_graph.S      
        
        o=self.S.shape[0]
        diag=np.diag(self.S).reshape((1,o))
        self.distance=np.ones((o,1))@diag+diag.T@np.ones((1,o))-2*self.S
        #Transform and initialize the model parameters (migration rate, coalesce rate,)
        
        self.nsample=np.zeros(d)
        for i in range (d):
            self.nsample[i]=2*self.nodes[i]["n_samples"]                        #The number of haplotypes in each observed deme,which is twice the number of individuals
        
        observed_nodes_indices=np.nonzero(self.nsample)[0]
        observed_nodes_nsample=self.nsample[observed_nodes_indices]
        self.h=(observed_nodes_indices,observed_nodes_nsample)
          
        #The degree of each node
        self.deg=np.array(list(self.degree()))[:,1]
       
        #Initilizing the migration rates and coalescent rates
        mtriu=sp_graph.w
        M=self.adj_base.copy()
        M.data=mtriu
        self.M0=M+M.T
        self.M0.data=np.ones(len(self.M0.data))
        self.m0=self.M0.data           
        self.L0=getlaplacian(self.m0,self.M0)    
        self.gamma0=np.ones(d)

        Ttotal=getcoalesce(self.L0,self.gamma0,self.S,self.h)[0]
        self.M0=Ttotal*self.M0
        self.m0=self.M0.data           
        self.L0=getlaplacian(self.m0,self.M0)
        self.gamma0=Ttotal*self.gamma0

        self.k=1
        self.c0=np.zeros(2)
        self.c0[0]=self.gamma0[0]/np.sqrt(d)
        self.c0[1]=0.5
     
        self.M=self.M0.copy()
        self.m=self.m0.copy()
        self.L=self.L0.copy()
        self.gamma=self.gamma0.copy()
        self.c=self.c0.copy()
                   
    # ------------------------- Optimizers -------------------------



    def fit(
        self,       
        lamb,
        maxls=50,
        factr=1e7,
        m=10,
        lb=-np.Inf,
        ub=np.Inf,
        maxiter=15000,
        verbose=True,
        logm_init=None,
        logc_init=None,
        ):
        if  logm_init is None:
            logm_init=np.log(self.m0)
        
        if logc_init is None:
           logc_init=np.zeros(2)
           logc_init[0]=np.log(self.c0[0])
           logc_init[1]=-self.k*np.log((1/self.c0[1])-1)
                     
        """Estimates the edge weights of the full model holding the residual
        variance fixed using a quasi-newton algorithm, specifically L-BFGS.

        Args:
            lamb_m (:self:`float`): penalty strength on migration rate difference
            lamb_gamma(:(:self:`float`): penalty strength on coalescence rate difference
            factr (:self:`float`): tolerance for convergence
            maxls (:self:`int`): maximum number of line search steps
            m (:self:`int`): the maximum number of variable metric corrections
            lb (:self:`int`): lower bound of parameters
            ub (:self:`int`): upper bound of parameters
            maxiter (:self:`int`): maximum number of iterations to run L-BFGS
            verbose (:self:`Bool`): boolean to print summary of results
            logm_init(:self:`float`):initial value of log edge weights
            loggamma_init(:self:`float`):initial value of log coalescence rates
            gamma_init(:self:`float`):initial value of coalescence rates"""
      
        # check inputs
        assert lamb >= 0.0, "lambda_w must be non-negative"
        assert type(lamb) == float, "lambda_w must be float"
        assert type(factr) == float, "factr must be float"
        assert maxls > 0, "maxls must be at least 1"
        assert type(maxls) == int, "maxls must be int"
        assert type(m) == int, "m must be int"
        assert type(lb) == float, "lb must be float"
        assert type(ub) == float, "ub must be float"
        assert lb < ub, "lb must be less than ub"
        assert type(maxiter) == int, "maxiter must be int"
        assert maxiter > 0, "maxiter be at least 1"
   
        # run l-bfgs
        x0 =  np.append(logm_init,logc_init)
        
        res = fmin_l_bfgs_b(
              func=loss_wrapper,
              x0=x0,
              args=[self.M0,self.S,self.h,self.n_snps,lamb,self.deg,self.k],
              factr=factr,              
              m=m,
              maxls=maxls,
              maxiter=maxiter,
              approx_grad=False,
              bounds=[(lb, ub) for _ in range(x0.shape[0])],)
        
        if maxiter >= 100:
           assert res[2]["warnflag"] == 0, "did not converge"
        
        #Update and store the attributes
        d=self.number_of_nodes()
        o=self.S.shape[0]
        nnzm=len(self.m0)  
        self.m = np.exp(res[0][0:nnzm])
        self.M.data=self.m                                                      #Migration rate matrix
        self.c[0]=np.exp(res[0][nnzm])
        self.c[1]=1/(1+np.exp(-res[0][nnzm+1]/self.k))
        self.L=getlaplacian(self.m,self.M)                                      #Laplacian
        mc=markovChain(np.identity(d)-self.L)
        mc.computePi('linear')
        self.y=mc.pi.reshape(d)                                                 #Stationary distribution  
        if np.min(self.y)<=0:
           mc.computePi('power') 
        self.y=mc.pi.reshape(d)   
        self.y=self.y/np.sum(self.y)     
        self.gamma=self.c[0]/(self.y**self.c[1])                                #Coalescent rates
        coalesce_wrapper=getcoalesce(self.L, self.gamma, self.S, self.h)
        self.T=coalesce_wrapper[1]                                              # Expected pairwise coalescence time
        self.T_bar=coalesce_wrapper[2]
        diag=np.diag(self.T_bar).reshape((1,o))
        self.distance_fit=2*self.T_bar-np.ones((o,1))@diag-diag.T@np.ones((1,o)) # Fitted genetic distances
        self.train_loss=res[1]                                                  # Train loss
        
        
        if verbose:
           sys.stdout.write(
               (
                   "lambda={:.7f}, "
                   "converged in {} iterations, "
                   "train_loss={:.7f}\n"
               ).format(lamb,res[2]["nit"], self.train_loss)
           )
