import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from .Sim import Sim
from .Spatial_Digraph import SpatialDiGraph
from .Cross_Validation import run_cv
from discreteMarkovChain import markovChain

def fitting(sp_digraph,
            lamb_grid,
            lamb_warmup=None,
            factr=1e10,
            factr_fine=1e7):
       
    l=len(lamb_grid)

    cv, node_train_idxs = run_cv(sp_digraph,
                                 lamb_grid,
                                 n_folds=10,
                                 lamb_warmup=lamb_warmup,
                                 factr=factr,
                                 random_state=500,
                                 outer_verbose=True,
                                 inner_verbose=False,)

    if np.argmin(cv) == 0:
       lamb_grid_fine = np.geomspace(lamb_grid[0], lamb_grid[1], 7)[::-1]

    elif np.argmin(cv) ==l-1:
         lamb_grid_fine = np.geomspace(lamb_grid[l-2], lamb_grid[l-1], 7)[::-1]

    else:
        lamb_grid_fine = np.geomspace(lamb_grid[np.argmin(cv)-1], lamb_grid[np.argmin(cv)+1], 7)[::-1]

    cv_fine, node_train_idxs_fine = run_cv(sp_digraph,
                                           lamb_grid_fine,
                                           n_folds=10,
                                           lamb_warmup=lamb_warmup,
                                           factr=factr,
                                           random_state=500,
                                           outer_verbose=True,
                                           inner_verbose=False,
                                           node_train_idxs=node_train_idxs)

    lamb_opt = lamb_grid_fine[np.argmin(cv_fine)]
    lamb_opt = float("{:.3g}".format(lamb_opt))
    
    if lamb_warmup is not None:
       sp_digraph.fit(lamb=lamb_warmup, factr=1e10)
       logm = np.log(sp_digraph.m)
       logc = np.log(sp_digraph.c)
       
    else:
        logm = None
        logc = None

    sp_digraph.fit(lamb=lamb_opt,
                   factr=factr_fine,
                   logm_init=logm,
                   logc_init=logc)

    sp_digraph.lamb_grid=lamb_grid
    sp_digraph.lamb_grid_fine=lamb_grid_fine
    sp_digraph.cv=cv
    sp_digraph.cv_fine=cv_fine
    
    return(sp_digraph)

def run_sim_migration(topology,
                      sample_mode,
                      n_e_mode):
    
    N_rows=11
    N_columns=9
    
    if topology=='boundary' or topology=='showcasing':
       boundary=[5]
    else:
        boundary=None
        
    if topology=='directional' or topology=='anisotropic':
       directional=[]
       for j in range(N_rows):
           directional.append(((0, j), N_columns-1, 'E'))
           if topology=='anisotropic':
              directional.append(((N_columns-1, j), N_columns-1, 'W'))
    elif topology=='showcasing':
         directional=[]
         r=int((N_columns-1)/2)
         directional.append(((r, 1), r, 'E'))
         directional.append(((r, 1), r, 'W'))
         directional.append(((r, 9), r, 'E'))
         directional.append(((r, 9), r, 'W'))
    else:
        directional=None
        
    if topology=='sink':
       sink=[((4,5),1)]
    elif topology=='radiation':
         sink=[((4,5),3)]
    elif topology=='showcasing':
         sink=[((2,3),1),((2,7),1)]
    else:
        sink=None
        
    if topology=='source':
       source=[((4,5),1)]
    elif topology=='admixture':
         source=[((4,5),3)]
    elif topology=='showcasing':
         source=[((6,3),1),((6,7),1)]
    else:
        source=None

    if topology=='circle':
       circle=[((4,4), (5,4), (5,5),(5,6),(4,6),(3,5))] 
    elif topology=='showcasing':
         circle=[((4,2), (5,2), (4,3)),((4,8), (4,7), (5,8))] 
    else:
        circle=None
       
    if topology=='showcasing':
       m_topo=10
    else:
        m_topo=3
        
    m_base=0.1
    m_low=0.3
    m_high=3
    
    n_samples_per_node=10
    node_sample_prob=1
    
    if sample_mode=='sparse':
       node_sample_prob=0.25  
    
    if sample_mode=='semi_random':
       semi_random_sampling=True
    else:
        semi_random_sampling=None
        
    Simulation = Sim(n_rows=N_rows,
                     n_columns=N_columns,
                     n_samples_per_node=n_samples_per_node,
                     node_sample_prob=node_sample_prob,
                     semi_random_sampling=semi_random_sampling,)
    
    Simulation.setup_digraph(m_base=m_base,
                             m_low=m_low,
                             m_high=m_high,
                             m_topo=m_topo,
                             boundary=boundary,
                             directional=directional,
                             sink=sink,
                             source=source,
                             circle=circle,
                             )
    
    Simulation.set_up_populations(n_e_mode=n_e_mode)

    genotypes = Simulation.simulate_genotypes(sequence_length=1,
                                              mu=1e-3,
                                              target_n_snps=100000,
                                              n_print=500)

    coord = Simulation.coord.copy()
    grid = Simulation.grid.copy()
    edges = Simulation.edges.copy()
    digraph = Simulation.digraph.copy()
    d = digraph.number_of_nodes()
    genotypes = genotypes.astype(np.float64)
    genotypes /= 2

    M=nx.adjacency_matrix(digraph, weight="weight").toarray()
    d=M.shape[0]
    for i in range(d):
        M[i,i]=1-np.sum(M[i,:])
    mc=markovChain(M)  
    mc.computePi('linear')
    y=mc.pi.reshape(d)                                                             #Get the stationary distribution   
    y=y/np.sum(y)
    
    ground_truth=SpatialDiGraph(genotypes, coord, grid, edges)
    ground_truth.M = csr_matrix(nx.adjacency_matrix(digraph, weight="weight").toarray())
    ground_truth.m = ground_truth.M.data
    if n_e_mode=='proportional':
       ground_truth.gamma=np.max(y)/y
    else:
        ground_truth.gamma=np.ones(d)
    ground_truth.y=y.copy()                                                     #Get the stationary distribution   

    lamb_grid = np.geomspace(1e-3, 1e3, 13)[::-1]
    lamb_warmup=1e3
    
    sp_digraph = SpatialDiGraph(genotypes, coord, grid, edges)
    sp_digraph=fitting(sp_digraph,
                       lamb_grid,
                       lamb_warmup)
    return(ground_truth,sp_digraph)

def run_sim_re(sample_mode,
               re_mode,):
    
    if re_mode=='radiation':
       N_rows=9
    elif re_mode=='directional':
         N_rows=11
         
    N_columns=9
          
    boundary=None
    directional=None
    sink=None
    source=None
    circle=None
       
    m_topo=3       
    m_base=0
    m_low=0.3
    m_high=3
    
    n_samples_per_node=10
    node_sample_prob=1
    
    if sample_mode=='sparse':
       node_sample_prob=0.25  
    
    if sample_mode=='semi_random':
       semi_random_sampling=True
    else:
        semi_random_sampling=None
        
    if re_mode=='radiation':
       reshape_origin=[(4,4)]
    else:
        reshape_origin=None
        
    Simulation = Sim(n_rows=N_rows,
                     n_columns=N_columns,
                     n_samples_per_node=n_samples_per_node,
                     node_sample_prob=node_sample_prob,
                     semi_random_sampling=semi_random_sampling,)
    
    Simulation.setup_digraph(reshape_origin=reshape_origin,
                             m_base=m_base,
                             m_low=m_low,
                             m_high=m_high,
                             m_topo=m_topo,
                             boundary=boundary,
                             directional=directional,
                             sink=sink,
                             source=source,
                             circle=circle,
                             )
    
    Simulation.set_up_populations()
    
    if re_mode=='radiation':
       re_origin=[(4,4)]
       re_dt=1e-3
       re_proportion=0.1
    
    elif re_mode=='directional':
         re_origin=[]
         for i in range(11):
             re_origin.append((4,i))
         re_dt=1e-3
         re_proportion=0.1
    
    Simulation.set_up_re(re_origin,
                         re_dt,
                         re_proportion,
                         re_mode,)

    genotypes = Simulation.simulate_genotypes(sequence_length=1,
                                              mu=1e-3,
                                              target_n_snps=100000,
                                              n_print=500)

    coord = Simulation.coord.copy()
    grid = Simulation.grid.copy()
    edges = Simulation.edges.copy()
    genotypes = genotypes.astype(np.float64)
    genotypes /= 2

    lamb_grid = np.geomspace(1e-6, 1e0, 13)[::-1]
    lamb_warmup=1e0
    
    sp_digraph = SpatialDiGraph(genotypes, coord, grid, edges)
    sp_digraph=fitting(sp_digraph,
                       lamb_grid,
                       lamb_warmup)
    
    return(sp_digraph)

def run_sim_mm(sample_mode,
               mm_mode):
    
    if mm_mode=='radiation':
       N_rows=9
       
    elif mm_mode=='directional':
         N_rows=11
    
    N_columns=9
    boundary=None
    directional=None
    sink=None
    source=None
    circle=None
       
    m_topo=3       
    m_base=0.1
    m_low=0.3
    m_high=3
    
    n_samples_per_node=10
    node_sample_prob=1
    
    if sample_mode=='sparse':
       node_sample_prob=0.25  
    
    if sample_mode=='semi_random':
       semi_random_sampling=True
    else:
        semi_random_sampling=None
        
    if mm_mode=='radiation':
       reshape_origin=[(4,4)]
    else:
        reshape_origin=None
        
    Simulation = Sim(n_rows=N_rows,
                     n_columns=N_columns,
                     n_samples_per_node=n_samples_per_node,
                     node_sample_prob=node_sample_prob,
                     semi_random_sampling=semi_random_sampling,)
    
    Simulation.setup_digraph(reshape_origin=reshape_origin,
                             m_base=m_base,
                             m_low=m_low,
                             m_high=m_high,
                             m_topo=m_topo,
                             boundary=boundary,
                             directional=directional,
                             sink=sink,
                             source=source,
                             circle=circle,
                             )
    
    Simulation.set_up_populations()
    
    if mm_mode=='radiation':
       mm_origin=[(4,4)]
       mm_dt=1e-1
       mm_proportion=0.2
    
    elif mm_mode=='directional':
         mm_origin=[]
         for i in range(11):
             mm_origin.append((4,i))
         mm_dt=1e-1
         mm_proportion=0.2
         
    Simulation.set_up_mm(mm_origin,
                         mm_dt,
                         mm_proportion,
                         mm_mode,
                         )

    genotypes = Simulation.simulate_genotypes(sequence_length=1,
                                              mu=1e-3,
                                              target_n_snps=100000,
                                              n_print=500)

    coord = Simulation.coord.copy()
    grid = Simulation.grid.copy()
    edges = Simulation.edges.copy()
    genotypes = genotypes.astype(np.float64)
    genotypes /= 2

    lamb_grid = np.geomspace(1e-3, 1e3, 13)[::-1]
    lamb_warmup=1e3
    
    sp_digraph = SpatialDiGraph(genotypes, coord, grid, edges)
    sp_digraph=fitting(sp_digraph,
                       lamb_grid,
                       lamb_warmup)
    return(sp_digraph)
