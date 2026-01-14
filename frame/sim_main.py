import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from .sim import Sim
from .spatial_digraph import SpatialDiGraph
from .cross_validation import run_cv
from discreteMarkovChain import markovChain

def fitting(sp_digraph,
            lamb_m_grid,
            lamb_m_warmup=None,
            factr=1e10,
            factr_fine=1e7):

    cv_errs, node_train_idxs = run_cv(sp_digraph=sp_digraph,
                                     lamb_m_grid=lamb_m_grid,
                                     n_folds=10,
                                     lamb_m_warmup=lamb_m_warmup,
                                     factr=factr,
                                     random_state=500,
                                     outer_verbose=True,
                                     inner_verbose=False,)
    
    if np.argmin(cv_errs)==0:
       lamb_m_grid_fine=np.geomspace(lamb_m_grid[0],lamb_m_grid[1],7)[::-1]

    elif np.argmin(cv_errs)==12:
         lamb_m_grid_fine=np.geomspace(lamb_m_grid[11],lamb_m_grid[12], 7)[::-1]
         
    else:
        lamb_m_grid_fine=np.geomspace(lamb_m_grid[np.argmin(cv_errs)-1],lamb_m_grid[np.argmin(cv_errs)+1], 7)[::-1]
        
    cv_errs_fine,node_train_idxs_fine=run_cv(sp_digraph, 
                                             lamb_m_grid=lamb_m_grid_fine,
                                             n_folds=10,
                                             lamb_m_warmup=lamb_m_warmup,
                                             factr=factr,
                                             random_state=500,
                                             outer_verbose=True,
                                             inner_verbose=False,
                                             node_train_idxs=node_train_idxs)

    lamb_m_opt=lamb_m_grid_fine[np.argmin(cv_errs_fine)]
    lamb_m_opt=float("{:.3g}".format(lamb_m_opt))

    sp_digraph.fit(lamb_m=lamb_m_warmup, factr=factr)
    logm = np.log(sp_digraph.m)
    logc = np.log(sp_digraph.c)
    trans_alpha=-np.log((1/sp_digraph.alpha)-1)

    sp_digraph.fit(lamb_m=lamb_m_opt,
                   logm_init=logm,
                   logc_init=logc,
                   trans_alpha_init=trans_alpha)

    sp_digraph.lamb_m_grid=lamb_m_grid
    sp_digraph.lamb_m_grid_fine=lamb_m_grid_fine
    sp_digraph.cv=cv_errs
    sp_digraph.cv_fine=cv_errs_fine
    
    return(sp_digraph)

def run_sim_migration(topology,
                      sample_mode,
                      n_e_mode,):
    
    N_rows=11
    N_columns=9
    
    if topology=='small_scale_patterns':
       boundary=[5]
    else:
        boundary=None
             
    if topology=='large_scale_directionally_migrating_lineages':
         directional=[]
         for j in range(N_rows):
             directional.append(((0, j), 8, 'E'))
                         
    elif topology=='large_scale_converging_directionally_migrating_lineages':
         directional=[]
         for j in range(N_rows):
             directional.append(((0, j), 4, 'E'))
             directional.append(((8, j), 4, 'W'))
    
    elif topology=='small_scale_patterns':
         directional=[]
         directional.append(((4, 1), 4, 'E'))
         directional.append(((4, 1), 4, 'W'))
         directional.append(((0, 9), 4, 'E'))
         directional.append(((8, 9), 4, 'W'))
                       
    if topology=='large_scale_spatially_converging_lineages':
       converging=[((4,5),3)]
    elif topology=='small_scale_patterns':
         converging=[((2,3),1)]
    else:
        converging=None
        
    if topology=='large_scale_spatially_diverging_lineages':
       diverging=[((4,5),3)]
    elif topology=='small_scale_patterns':
         diverging=[((6,3),1)]
    else:
        diverging=None

    if topology=='small_scale_patterns':
       circle=[((2,6),(3,6),(3,7),(3,8),(2,8),(1,7)),((5,6),(6,6),(7,6),(6,7),(6,8),(5,7))] 
    else:
        circle=None
       
    if topology=='small_scale_patterns':
       m_topo=10
    else:
        m_topo=3
         
    m_base=0.1
    m_low=0.3
    m_high=3
    
    n_samples_per_node=10
    node_sample_prob=1
    
    if sample_mode=='sparse':
       node_sample_prob=0.3
    
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
                             converging=converging,
                             diverging=diverging,
                             circle=circle,
                             )
    
    Simulation.set_up_populations(n_e_mode=n_e_mode)

    genotypes = Simulation.simulate_genotypes(sequence_length=1,
                                              mu=1e-3,
                                              target_n_snps=100000,
                                              n_print=5000)

    coord = Simulation.coord.copy()
    grid = Simulation.grid.copy()
    edges = Simulation.edges.copy()
    digraph = Simulation.digraph.copy()
    d = digraph.number_of_nodes()
    genotypes = genotypes.astype(np.float64)

    M=nx.adjacency_matrix(digraph, weight="weight").toarray()
    d=M.shape[0]
    for i in range(d):
        M[i,i]=1-np.sum(M[i,:])
    mc=markovChain(M)  
    mc.computePi('linear')
    pi=mc.pi.reshape(d)                                                             #Get the stationary distribution   
    pi=pi/np.sum(pi)
    
    ground_truth=SpatialDiGraph(genotypes, coord, grid, edges)
    ground_truth.M = csr_matrix(nx.adjacency_matrix(digraph, weight="weight").toarray())
    ground_truth.m = ground_truth.M.data
    ground_truth.gamma=1/Simulation.n_e
    ground_truth.pi=pi.copy()                                                     #Get the stationary distribution   

    lamb_m_grid = np.geomspace(1e-3, 1e3,13)[::-1]
    lamb_m_warmup=1e3
     
    sp_digraph = SpatialDiGraph(genotypes, coord, grid, edges)
    sp_digraph=fitting(sp_digraph=sp_digraph,
                       lamb_m_grid=lamb_m_grid,
                       lamb_m_warmup=lamb_m_warmup)
    return(ground_truth,sp_digraph)

def run_sim_re(sample_mode):

    N_rows=11         
    N_columns=9
          
    boundary=None
    directional=None
    converging=None
    diverging=None
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
                             converging=converging,
                             diverging=diverging,
                             circle=circle,
                             )
    
    Simulation.set_up_populations()
    
    re_origin=[]
    for i in range(11):
        re_origin.append((4,i))
    re_dt=1e-2
    re_proportion=0.1
    
    Simulation.set_up_re(re_origin,
                         re_dt,
                         re_proportion)

    genotypes = Simulation.simulate_genotypes(sequence_length=1,
                                              mu=1e-3,
                                              target_n_snps=100000,
                                              n_print=5000)

    coord = Simulation.coord.copy()
    grid = Simulation.grid.copy()
    edges = Simulation.edges.copy()
    genotypes = genotypes.astype(np.float64)

    sp_digraph = SpatialDiGraph(genotypes, coord, grid, edges)
    
    return(sp_digraph)

def run_sim_mm(sample_mode):
    
    N_rows=11   
    N_columns=9
    boundary=None
    directional=None
    converging=None
    diverging=None
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
                             converging=converging,
                             diverging=diverging,
                             circle=circle,
                             )
    
    Simulation.set_up_populations()
      
    mm_origin=[]
    for i in range(11):
        mm_origin.append((4,i))
    mm_dt=1e-1
    mm_proportion=0.5
         
    Simulation.set_up_mm(mm_origin,
                         mm_dt,
                         mm_proportion,
                         )

    genotypes = Simulation.simulate_genotypes(sequence_length=1,
                                              mu=1e-3,
                                              target_n_snps=100000,
                                              n_print=500)

    coord = Simulation.coord.copy()
    grid = Simulation.grid.copy()
    edges = Simulation.edges.copy()
    genotypes = genotypes.astype(np.float64)
    
    sp_digraph = SpatialDiGraph(genotypes, coord, grid, edges)
    return(sp_digraph)
