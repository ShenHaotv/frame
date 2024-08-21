from __future__ import absolute_import, division, print_function
import sys
import networkx as nx
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from Loss import getlaplacian,getcoalesce,loss_wrapper
from discreteMarkovChain import markovChain

def query_node_attributes(digraph, name):
    """Query the node attributes of a nx digraph. This wraps get_node_attributes
    and returns an array of values for each node instead of the dict
    """
    d = nx.get_node_attributes(digraph, name)
    arr = np.array(list(d.values()))
    return arr

class SpatialDiGraph(nx.DiGraph):
    def __init__(self, genotypes, sample_pos, node_pos, edges):
        """Represents the spatial network which the data is defined on and
        stores relevant matrices / performs linear algebra routines needed for
        the model and optimization. Inherits from the networkx Graph object.

        Args:
            genotypes (:obj:`numpy.ndarray`): genotypes for samples
            sample_pos (:obj:`numpy.ndarray`): spatial positions for samples
            node_pos (:obj:`numpy.ndarray`):  spatial positions of nodes
            edges (:obj:`numpy.ndarray`): edge array
        """
        # Check inputs
        assert len(genotypes.shape) == 2
        assert len(sample_pos.shape) == 2
        assert np.all(~np.isnan(genotypes)), "no missing genotypes are allowed"
        assert np.all(~np.isinf(genotypes)), "non inf genotypes are allowed"
        assert (
            genotypes.shape[0] == sample_pos.shape[0]
        ), "genotypes and sample positions must be the same size"

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
        
        # Assign samples to nodes
        self._assign_samples_to_nodes(sample_pos, node_pos)                    

        self.n_hap=np.zeros(d)
        for i in range (d):
            self.n_hap[i]=2*self.nodes[i]["n_samples"]                         #The number of haplotypes in each observed deme,which is twice the number of individuals
    
        observed_nodes_indices=np.nonzero(self.n_hap)[0]
        observed_nodes_n_hap=self.n_hap[observed_nodes_indices]
        self.h=(observed_nodes_indices,observed_nodes_n_hap)                   #Indices and number of haplotypes in observed demes
         
        # estimate allele frequencies at observed locations 
        self.genotypes = genotypes
        self._estimate_allele_frequencies()
        self.heterozygosity=2*self.frequencies*(1-self.frequencies)
        self.average_heterozygosity=np.mean(self.heterozygosity,axis=1)

        # estimate sample covariance matrix
        self.S=self.frequencies@(self.frequencies).T/self.n_snps   
        o=self.S.shape[0]
        diag=np.diag(self.S).reshape((1,o))
        self.distance=np.ones((o,1))@diag+diag.T@np.ones((1,o))-2*self.S
        
        #Get the adhacency matrix
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

        self.k=1
        self.c0=np.zeros(2)
        self.c0[0]=self.gamma0[0]/np.sqrt(d)
        self.c0[1]=0.5
     
        self.M=self.M0.copy()
        self.m=self.m0.copy()
        self.L=self.L0.copy()
        self.gamma=self.gamma0.copy()
        self.c=self.c0.copy()
  
    def _init_digraph(self, node_pos, edges):
        """Initialize the graph and related graph objects

        Args:
            node_pos (:obj:`numpy.ndarray`):  spatial positions of nodes
            edges (:obj:`numpy.ndarray`): edge array
        """
        self.add_nodes_from(np.arange(node_pos.shape[0]))
        # Subtract 1 from each edge to adjust from 1-based to 0-based indexing
        adjusted_edges = edges - 1
       
        # Create a list to hold both original and reversed edges
        all_edges = []
        for edge in adjusted_edges:
            all_edges.append(tuple(edge))
            all_edges.append((edge[1], edge[0]))  # Add the reversed edge
       
        # Add edges to the graph
        self.add_edges_from(all_edges)

        # add spatial coordinates to node attributes
        for i in range(len(self)):
            self.nodes[i]["idx"] = i
            self.nodes[i]["pos"] = node_pos[i, :]
            self.nodes[i]["n_samples"] = 0
            self.nodes[i]["sample_idx"] = []
    
    def _assign_samples_to_nodes(self, sample_pos, node_pos):
       """Assigns each sample to a node on the graph by finding the closest
       node to that sample
       """
       n_samples = sample_pos.shape[0]
       assned_node_idx = np.zeros(n_samples, "int")
       for i in range(n_samples):
           dist = (sample_pos[i, :] - node_pos) ** 2
           idx = np.argmin(np.sum(dist, axis=1))
           assned_node_idx[i] = idx
           self.nodes[idx]["n_samples"] += 1
           self.nodes[idx]["sample_idx"].append(i)
       n_samples_per_node = query_node_attributes(self, "n_samples")
       self.n_observed_nodes = np.sum(n_samples_per_node != 0)
       self.assned_node_idx = assned_node_idx
    
    def _estimate_allele_frequencies(self):
        """Estimates allele frequencies by maximum likelihood on the observed
        nodes of the spatial graph

        Args:
            genotypes (:obj:`numpy.ndarray`): array of diploid genotypes with
              no missing data
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
            allele_counts = np.mean(self.genotypes[s, :], axis=0)
            self.frequencies[i, :] = allele_counts
                               
    # ------------------------- Optimizers -------------------------



    def fit(
        self,       
        lamb,
        maxls=50,
        factr=1e7,
        m=10,
        lb=-np.inf,
        ub=np.inf,
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
            m   #The degree of each node
               self.deg=np.array(list(self.degree()))[:,1]axls (:self:`int`): maximum number of line search steps
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
