from __future__ import absolute_import, division, print_function
import sys
import networkx as nx
import numpy as np
import numbers
from scipy.optimize import fmin_l_bfgs_b
from .loss import getlaplacian,getcoalesce,loss_wrapper
from discreteMarkovChain import markovChain

def query_node_attributes(digraph, name):
    """Query the node attributes of a nx digraph. This wraps get_node_attributes
    and returns an array of values for each node instead of the dict
    """
    d = nx.get_node_attributes(digraph, name)
    arr = np.array(list(d.values()))
    return arr

class SpatialDiGraph(nx.DiGraph):
    def __init__(self, genotypes, sample_pos, node_pos, edges, sample_plo=None, preassignment=None):
        """Represents the spatial network which the data is defined on and
        stores relevant matrices / performs linear algebra routines needed for
        the model and optimization. Inherits from the networkx Graph object.

        Args:
            genotypes (:obj:`numpy.ndarray`): genotypes for samples
            sample_pos (:obj:`numpy.ndarray`): spatial positions for samples
            node_pos (:obj:`numpy.ndarray`):  spatial positions of nodes
            edges (:obj:`numpy.ndarray`): edge array(in undirected format)
            sample_plo (:obj:`numpy.ndarray`): ploidies of the samples
            preassignment (:obj:`numpy.ndarray`): artificial preassignment of samples to nodes before the
                                                  automatic assignment based on closest distance
        """
        # Check inputs
        assert len(genotypes.shape) == 2
        assert len(sample_pos.shape) == 2
        assert np.all(~np.isnan(genotypes)), "no missing genotypes are allowed"
        assert np.all(~np.isinf(genotypes)), "non inf genotypes are allowed"
        assert (
            genotypes.shape[0] == sample_pos.shape[0]
        ), "genotypes and sample positions must be the same size"

        if preassignment is not None:
           assert len(preassignment.shape) == 2
           
        # Inherits from networkx Graph object
        super(SpatialDiGraph, self).__init__()
        self._init_digraph(node_pos, edges)  # init graph

        # Inputs
        self.sample_pos = sample_pos
        self.node_pos = node_pos
        
        # Number of nodes
        d=self.number_of_nodes()
        
        # Degree of each node
        self.deg=np.array(list(self.degree()))[:,1]
        
        #preassignment of samples to nodes
        self.preassignment=preassignment    
        
        if sample_plo is None:
           self.sample_plo=2*np.ones(sample_pos.shape[0])                      # The default ploidy of each sample is set to 2
        else:
            self.sample_plo=sample_plo
        
        self.n_hap=np.zeros(d)                                                 # Number of haplotypes in each deme, will be updated in the sample assignment step
        
        # Assign samples to nodes
        if self.preassignment is not None:
           for i in range(self.preassignment.shape[0]):
               sample_idx=self.preassignment[i][0]
               node_idx = self.preassignment[i][1]
               self.nodes[node_idx]["n_haps"]+=self.sample_plo[sample_idx]
               self.nodes[node_idx]["sample_idx"].append(sample_idx)
        
        self._assign_samples_to_nodes(sample_pos, node_pos)                    

        self.n_hap=np.zeros(d)
        for i in range (d):
            self.n_hap[i]=self.nodes[i]["n_haps"]                              
    
        observed_nodes_indices=np.nonzero(self.n_hap)[0]                       # Indices of observed demes
        observed_nodes_n_hap=self.n_hap[observed_nodes_indices]                # Number of haplotypes in each observed deme
        self.h=(observed_nodes_indices,observed_nodes_n_hap)                   
         
        # estimate allele frequencies at observed locations 
        self.genotypes = genotypes
        self._estimate_allele_frequencies()
        
        #Estimate heterozygosities
        self.heterozygosity=2*self.frequencies*(1-self.frequencies)
        self.average_heterozygosity=np.mean(self.heterozygosity,axis=1)

        # estimate sample covariance matrix
        self.S=self.frequencies@(self.frequencies).T/self.n_snps   
        o=self.S.shape[0]
        diag=np.diag(self.S).reshape((1,o))
        self.distance=np.ones((o,1))@diag+diag.T@np.ones((1,o))-2*self.S
        
        #Get the adjacency matrix
        self.adj=nx.adjacency_matrix(self)
       
        #Initilizing the migration rates and coalescent rates
        M=self.adj.copy()
        M.data=np.ones(len(M.data))
        self.M0=M
        self.m0=self.M0.data           
        self.L0=getlaplacian(self.m0,self.M0)    
        self.gamma0=np.ones(d)

        Ttotal=getcoalesce(self.L0,self.gamma0,self.S,self.h)[0]
        self.M0=Ttotal*self.M0
        self.m0=self.M0.data           
        self.L0=getlaplacian(self.m0,self.M0)
        self.gamma0=Ttotal*self.gamma0
      
        self.c0=self.gamma0[0]/np.sqrt(d)
        self.alpha0=0.5
      
        self.M=self.M0.copy()
        self.m=self.m0.copy()
        self.L=self.L0.copy()
        self.gamma=self.gamma0.copy()
        self.c=self.c0
        self.alpha=self.alpha0
  
    def _init_digraph(self, node_pos, edges):
        """Initialize the digraph and related digraph objects

         Args:
        node_pos (:obj:`numpy.ndarray`): spatial positions of nodes
        edges (:obj:`numpy.ndarray`): edge array (0-based indexing)
         """
        self.add_nodes_from(np.arange(node_pos.shape[0]))
    
        # Create a list to hold both original and reversed edges (0-based)
        all_edges = []
        for edge in edges:
            all_edges.append(tuple(edge))
            all_edges.append((edge[1], edge[0]))  # Add the reversed edge
    
        # Add edges to the graph
        self.add_edges_from(all_edges)

        # Add spatial coordinates to node attributes
        for i in range(len(self)):
            self.nodes[i]["idx"] = i
            self.nodes[i]["pos"] = node_pos[i, :]
            self.nodes[i]["n_haps"] = 0
            self.nodes[i]["sample_idx"] = []
        
    def _assign_samples_to_nodes(self, sample_pos, node_pos):
       """Assigns each sample to a node on the graph by finding the closest
       node to that sample
       
       Args:
           sample_pos:sample positions
           node_pos:node positions        
       """
       if self.preassignment is not None:         
          samples = np.delete(np.arange(sample_pos.shape[0]),self.preassignment[0])
       else:
           samples=np.arange(sample_pos.shape[0])
       for i in samples:
           dist = (sample_pos[i, :] - node_pos) ** 2
           idx = np.argmin(np.sum(dist, axis=1))
           self.nodes[idx]["n_haps"]+=self.sample_plo[i]
           self.nodes[idx]["sample_idx"].append(i)
       n_haps_per_node = query_node_attributes(self, "n_haps")
       self.n_observed_nodes = np.sum(n_haps_per_node != 0)
    
    def _estimate_allele_frequencies(self):
        """Estimates allele frequencies by maximum likelihood on the observed
        nodes of the spatial digraph
        """
        self.n_snps = self.genotypes.shape[1]

         # create the data matrix of means
        self.frequencies = np.empty((self.n_observed_nodes, self.n_snps))

        # get indicies
        sample_idx = nx.get_node_attributes(self, "sample_idx")

        for i, node_id in enumerate(self.h[0]):

            # find the samples assigned to the ith node
            s = sample_idx[node_id]

            # compute mean at each node
            frequencies = np.sum(self.genotypes[s, :], axis=0)/np.sum(self.sample_plo[s])
            self.frequencies[i, :] = frequencies
                               
    # ------------------------- Optimizers -------------------------
    def fit(self,   
            lamb_m,
            maxls=50,
            factr=1e7,
            m=10,
            lb=-np.Inf,
            ub=np.Inf,
            maxiter=15000,
            verbose=True,
            alpha_lb=0.0,
            alpha_ub=1.0,
            trans_alpha_init=0.0,
            alpha_scale=1.0,
            logm_init=None,
            logc_init=None,):
        """Estimates model parameters with L-BFGS for either compact ('cp') or full ('fp') parameterization model.

           Args:
               lamb_m (real): Penalty strength on migration-rate differences.
               maxls (int): Maximum number of line search steps.
               factr (real): Tolerance for convergence (passed to L-BFGS-B).
               m (int): The maximum number of variable metric corrections.
               lb (real): Lower bound for parameters (applied to all elements).
               ub (real): Upper bound for parameters (applied to all elements).
               maxiter (int): Maximum number of L-BFGS iterations.
               verbose (bool): Print a brief optimization summary if True.
               alpha_lb (real): Lower bound of alpha.
               alpha_ub (real): Upper bound of alpha.
               trans_alpha_init (real): Initial value of the unconstrained (transformed) parameter corresponding
                                        to alpha. This parameter lives on ℝ and is mapped to alpha via an logistic transformation.
               alpha_scale (real): Scale parameter of the logistic transformation. Controls how
                                   rapidly α transitions between its lower and upper bounds as a function
                                   of the transformed parameter. Larger values produce a flatter transition;
                                   smaller values produce a steeper transition.
               logm_init (np.ndarray): Initial values for log edge weights (len == nnz of M).
               logc_init (np.ndarray): Initial values for log c (len ==1). """
      
        assert isinstance(lamb_m, numbers.Real) and lamb_m >= 0.0, "lamb_m must be non-negative real number"
        assert isinstance(maxls, int) and maxls > 0, "maxls must be int >= 1"
        assert isinstance(factr, numbers.Real) and factr>0, "factr must be positive real number"
        assert isinstance(m, int), "m must be int"
        assert isinstance(lb, numbers.Real), "lb must be real number"
        assert isinstance(ub, numbers.Real), "ub must be real number"
        assert lb < ub, "lb must be less than ub"
        assert isinstance(maxiter, int) and maxiter > 0, "maxiter must be int >= 1"
        assert isinstance(alpha_lb, numbers.Real) and 0 <= alpha_lb <= 1, "alpha_lb must be real number in [0,1]"
        assert isinstance(alpha_ub, numbers.Real) and 0 <= alpha_ub <= 1, "alpha_ub must be real number in [0,1]"
        assert alpha_lb < alpha_ub, "alpha_lb must be less than alpha_ub"
        assert isinstance(trans_alpha_init, numbers.Real), "trans_alpha_init must be real number"
        assert isinstance(alpha_scale, numbers.Real) and alpha_scale > 0, "alpha_scale must be positive real number"
        
        if logm_init is None:
           logm_init = np.log(self.m0)
           
        alpha_init=alpha_lb+(alpha_ub-alpha_lb) / (1 + np.exp(-trans_alpha_init/alpha_scale))

        if logc_init is None:
           d=self.number_of_nodes()
           c_init=self.gamma0[0]/np.power(d,alpha_init)
           logc_init=np.log(c_init)
         
        # run l-bfgs
        x0 = np.concatenate([logm_init,np.atleast_1d(logc_init),np.atleast_1d(trans_alpha_init)])
        res = fmin_l_bfgs_b(func=loss_wrapper,
                            x0=x0,
                            args=[self.M0,self.S,self.h,self.n_snps,lamb_m,self.deg,alpha_lb,alpha_ub,alpha_scale],
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
        self.M.data=self.m                                                     #Migration rate matrix
        self.L=getlaplacian(self.m,self.M)
        mc=markovChain(np.identity(d)-self.L)
        mc.computePi('linear')
        self.pi=mc.pi.reshape(d)                                               #Stationary distribution, represented as a column vector  
        if np.min(self.pi)<=0:
           mc.computePi('power') 
        self.pi=mc.pi.reshape(d)   
        self.pi=self.pi/np.sum(self.pi) 
                                                        
        self.c=np.exp(res[0][nnzm])
        self.alpha=alpha_lb+(alpha_ub-alpha_lb)/(1+np.exp(-res[0][nnzm+1]/alpha_scale))       
        self.gamma=self.c/(self.pi**self.alpha)                                #Coalescent rates
            
        coalesce_wrapper=getcoalesce(self.L, self.gamma, self.S, self.h)
        self.T=coalesce_wrapper[1]                                             #Expected pairwise coalescence time
        self.T_bar=coalesce_wrapper[2]
        diag=np.diag(self.T_bar).reshape((1,o))
        self.distance_fit=2*self.T_bar-np.ones((o,1))@diag-diag.T@np.ones((1,o)) # Fitted genetic distances
        self.train_loss=res[1]                                                   # Train loss
               
        if verbose:
           sys.stdout.write(("\nlambda={:.7f}, "
                             "converged in {} iterations, "
                             "train_loss={:.7f}").format(lamb_m,res[2]["nit"], self.train_loss))