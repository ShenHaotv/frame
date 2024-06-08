import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import networkx as nx
from Sim import Sim
from feems import SpatialGraph
from Spatial_Digraph import SpatialDiGraph
import timeit
from Visualization import Vis
from Cross_Validation import run_cv
from discreteMarkovChain import markovChain
from Digraphstats import Digraphstats
import pickle

start = timeit.default_timer()

N_rows = 9
N_columns = 9
directional = []
directional_re = [(0,4)]
for j in range(N_rows):
    #directional.append(((0, j), N_columns-1, 'E'))
    # Directional.append(((N_columns-1- int(j/2),j), N_columns,'W'))
    directional.append((4, j))

Simulation = Sim(n_rows=N_rows,
                 n_columns=N_columns,
                 n_samples_per_node=10,
                 node_sample_prob=1,
                 individual_sample_prob=1,
                 semi_random_sampling=None,)

Simulation.setup_digraph(m_base=0.1,
                         m_low=0.3,
                         m_high=3,
                         # boundary=[4],
                         # directional=directional,
                         # sink=[((4,4),3)],
                         # source=[((4,4),3)],
                         # circle=[((4,2), (5,2), (4,3)),((4,6), (4,5), (5,6))]
                         )

# Simulation.set_up_populations(n_e_mode="proportional")
Simulation.set_up_populations()
"""Simulation.set_up_re(re_origin=directional_re,
                     re_dt=1e-3,
                     re_proportion=0.1,
                     re_mode='directional',)"""

Simulation.set_up_mm(mm_origin=[(4,4)],
                     mm_dt=1e-3,
                     mm_proportion=0.2,
                     mm_mode='radiation',)


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

"""
M=nx.adjacency_matrix(digraph, weight="weight").toarray()
d=M.shape[0]
for i in range(d):
    M[i,i]=1-np.sum(M[i,:])
       
mc=markovChain(M)  
mc.computePi('linear')
y=mc.pi.reshape(d)                                                             #Get the stationary distribution   
y=y/np.sum(y)

ground_truth=SpatialGraph(genotypes, coord, grid, edges, scale_snps=False)
ground_truth=SpatialDiGraph(ground_truth)
ground_truth.M = csr_matrix(nx.adjacency_matrix(digraph, weight="weight").toarray())
ground_truth.m = ground_truth.M.data
ground_truth.gamma=np.max(y)/y
ground_truth.y=y.copy()                                                     #Get the stationary distribution  
weights_truth = ground_truth.m

projection=ccrs.Mercator()
fig, axs = plt.subplots(2, 3, figsize=(10, 10), dpi=300,subplot_kw={'projection': projection})
v = Vis(axs[0,0], ground_truth, projection=projection, edge_width=0.5,
        edge_alpha=1, edge_zorder=100, sample_pt_size=20,
        obs_node_size=7.5, sample_pt_color="black",
        cbar_font_size=10, cbar_loc='lower center',cbar_ticklabelsize=6,
        cbar_bbox_to_anchor=(0.0, -0.2, 1, 1),   
        campass_bbox_to_anchor=(0.4, -0.35),mutation_scale=5)

# Show the figure
plt.show()  """

sp_graph = SpatialGraph(genotypes/2, coord, grid, edges, scale_snps=False)

sp_digraph = SpatialDiGraph(sp_graph)

print(np.mean(sp_digraph.frequencies))

print(sp_digraph.n_snps)

lamb_warmup = 1e3

lamb_grid = np.geomspace(1e-6, 1e3, 10)[::-1]

cv, node_train_idxs = run_cv(sp_digraph,
                             lamb_grid,
                             n_folds=10,
                             lamb_warmup=lamb_warmup,
                             factr=1e10,
                             random_state=500,
                             outer_verbose=True,
                             inner_verbose=False,)

if np.argmin(cv) == 0:
    lamb_grid_fine = np.geomspace(lamb_grid[0], lamb_grid[1], 7)[::-1]

elif np.argmin(cv) == 9:
    lamb_grid_fine = np.geomspace(lamb_grid[8], lamb_grid[9], 7)[::-1]

else:
    lamb_grid_fine = np.geomspace(
        lamb_grid[np.argmin(cv)-1], lamb_grid[np.argmin(cv)+1], 7)[::-1]

cv_fine, node_train_idxs_fine = run_cv(sp_digraph,
                                       lamb_grid_fine,
                                       n_folds=10,
                                       lamb_warmup=lamb_warmup,
                                       factr=1e10,
                                       random_state=500,
                                       outer_verbose=True,
                                       inner_verbose=False,
                                       node_train_idxs=node_train_idxs)

lamb_opt = lamb_grid_fine[np.argmin(cv_fine)]
lamb_opt = float("{:.3g}".format(lamb_opt))

sp_digraph.fit(lamb=lamb_warmup, factr=1e10)
logm = np.log(sp_digraph.m)
logc = np.log(sp_digraph.c)

sp_digraph.fit(lamb=1e-6,
               factr=1e7,
               logm_init=logm,
               logc_init=logc)

projection = ccrs.Mercator()
fig, axs = plt.subplots(2, 3, figsize=(10, 10), dpi=300,
                        subplot_kw={'projection': projection})
v = Vis(axs[0, 0], sp_digraph, projection=ccrs.Mercator(), edge_width=0.5,
        edge_alpha=1, edge_zorder=100, sample_pt_size=20,
        obs_node_size=7.5, sample_pt_color="black",
        cbar_font_size=8, cbar_loc='lower center', cbar_ticklabelsize=8,
        cbar_bbox_to_anchor=(0.0, -0.2, 1, 1),
        campass_bbox_to_anchor=(0.4, -0.35), mutation_scale=5)

v.digraph_wrapper(axs, node_scale=[1, 10, 10], draw_map=None, draw_nodes=None)

# Show the figure
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(np.log10(lamb_grid), cv, 'bo')
plt.plot(np.log10(lamb_grid_fine), cv_fine, 'bo')
plt.xlabel(r"$\mathrm{log}_{10}(\mathrm{\lambda})$")
plt.ylabel('CV Error')

print(cv, cv_fine)

digraphstats = Digraphstats(sp_digraph)

fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
digraphstats.fitting_wrapper(axs)
plt.show()
"""import pickle
with open('directional.pkl', 'wb') as f:
     pickle.dump(sp_digraph, f)
     
with open('directional.pkl', 'rb') as f:
     sp_digraph = pickle.load(f)"""
