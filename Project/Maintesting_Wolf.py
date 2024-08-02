import numpy as np
from sklearn.impute import SimpleImputer
from pandas_plink import read_plink
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from utils import prepare_graph_inputs
from Spatial_Digraph import SpatialDiGraph
from Visualization import Vis
from Cross_Validation import run_cv
from Digraphstats import Digraphstats

data_path="/Users/shenhao/Desktop/Project/data" 
(bim, fam, G) = read_plink("{}/wolvesadmix".format(data_path))
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
genotypes = imp.fit_transform((np.array(G)).T)/2
coord = np.loadtxt("{}/wolvesadmix.coord".format(data_path))
outer = np.loadtxt("{}/wolvesadmix.outer".format(data_path))
grid_path = "{}/grid250.shp".format(data_path)

outer, edges, grid, _ = prepare_graph_inputs(coord=coord,
                                             ggrid=grid_path,
                                             translated=True,
                                             buffer=0,
                                             outer=outer)

sp_digraph = SpatialDiGraph(genotypes, coord, grid, edges)

lamb_warmup = 1e3

lamb_grid = np.geomspace(1e-3, 1e3,13)[::-1]

cv,node_train_idxs=run_cv(sp_digraph,
                          lamb_grid,
                          lamb_warmup=lamb_warmup,
                          n_folds=10,
                          factr=1e10,
                          random_state=500,
                          outer_verbose=True,
                          inner_verbose=False,)

if np.argmin(cv)==0:
   lamb_grid_fine=np.geomspace(lamb_grid[0],lamb_grid[1],5)[::-1]

elif np.argmin(cv)==12:
     lamb_grid_fine=np.geomspace(lamb_grid[11],lamb_grid[12], 5)[::-1]
     
else:
    lamb_grid_fine=np.geomspace(lamb_grid[np.argmin(cv)-1],lamb_grid[np.argmin(cv)+1], 5)[::-1]

cv_fine,node_train_idxs_fine=run_cv(sp_digraph,
                                    lamb_grid_fine,
                                    lamb_warmup=lamb_warmup,
                                    n_folds=10,
                                    factr=1e10,
                                    random_state=500,
                                    outer_verbose=True,
                                    inner_verbose=False,
                                    node_train_idxs=node_train_idxs)

lamb_opt=lamb_grid_fine[np.argmin(cv_fine)]
lamb_opt=float("{:.3g}".format(lamb_opt))

sp_digraph.fit(lamb=lamb_warmup, factr=1e10)
logm = np.log(sp_digraph.m)
logc = np.log(sp_digraph.c)

sp_digraph.fit(lamb=lamb_opt,
               factr=1e7,
               logm_init=logm,
               logc_init=logc,
               )


projection = ccrs.EquidistantConic(
    central_longitude=-108.842926, central_latitude=66.037547)

fig, axs= plt.subplots(2, 4, figsize=(15, 6), dpi=300,
                        subplot_kw={'projection': projection})

v = Vis(axs[0,0], sp_digraph, projection=projection, edge_width=0.5,
        edge_alpha=1, edge_zorder=100, sample_pt_size=20,
        obs_node_size=5, sample_pt_color="black",
        cbar_font_size=10, cbar_ticklabelsize=8, mutation_scale=3)

v.digraph_wrapper(axs, node_scale=[5, 5, 5])

plt.show()

plt.figure(figsize=(8, 6))
plt.plot(np.log10(lamb_grid), cv, 'bo')  
plt.plot(np.log10(lamb_grid_fine), cv_fine, 'bo')  
plt.xlabel(r"$\mathrm{log}_{10}(\mathrm{\lambda})$")
plt.ylabel('CV Error')

digraphstats = Digraphstats(sp_digraph)

fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
digraphstats.fitting_wrapper(axs)
plt.show()


