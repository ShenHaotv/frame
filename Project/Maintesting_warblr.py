import numpy as np
import pkg_resources
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from utils import prepare_graph_inputs
from feems import SpatialGraph
from Spatial_Digraph import SpatialDiGraph
from Visualization import Vis
from Cross_Validation import run_cv
from Digraphstats import Digraphstats
import timeit

start = timeit.default_timer()
data_path = pkg_resources.resource_filename("feems", "data/")
genotypes = np.loadtxt('{}/genotypes_ESM13.csv'.format(data_path))/2
coord = np.loadtxt("{}/warblr.coord".format(data_path))  # sample coordinates
outer = np.loadtxt("{}/warblr.outer".format(data_path))  # outer coordinates
grid_path = "{}/grid250.shp".format(data_path)  # path to discrete global grid

outer, edges, grid, _ = prepare_graph_inputs(coord=coord,
                                             ggrid=grid_path,
                                             translated=False,
                                             buffer=0,
                                             outer=outer)

sp_graph = SpatialGraph(genotypes, coord, grid, edges, scale_snps=False)
sp_graph.fit(lamb=2.)

sp_digraph = SpatialDiGraph(sp_graph)

lamb_warmup = 1e3
"""lamb_grid = np.geomspace(1e-3, 1e3,13)[::-1]

cv,node_train_idxs=run_cv(sp_digraph,
                          lamb_grid,
                          lamb_warmup=lamb_warmup,
                          #n_folds=10,
                          factr=1e10,
                          random_state=500,
                          outer_verbose=True,
                          inner_verbose=False,)

if np.argmin(cv)==0:
   lamb_grid_fine=np.geomspace(lamb_grid[0],lamb_grid[2], 7)[::-1]

elif np.argmin(cv)==12:
     lamb_grid_fine=np.geomspace(lamb_grid[10],lamb_grid[12], 7)[::-1]
     
else:
    lamb_grid_fine=np.geomspace(lamb_grid[np.argmin(cv)-1],lamb_grid[np.argmin(cv)+1], 7)[::-1]

cv_fine,node_train_idxs_fine=run_cv(sp_digraph,
                                    lamb_grid_fine,
                                    lamb_warmup=lamb_warmup,
                                    #n_folds=10,
                                    factr=1e10,
                                    random_state=500,
                                    outer_verbose=True,
                                    inner_verbose=False,
                                    node_train_idxs=node_train_idxs)

lamb_opt=lamb_grid_fine[np.argmin(cv_fine)]
lamb_opt=float("{:.3g}".format(lamb_opt))"""

sp_digraph.fit(lamb=lamb_warmup, factr=1e10)
logm = np.log(sp_digraph.m)
logc = np.log(sp_digraph.c)

sp_digraph.fit(lamb=1e0,
               factr=1e7,
               logm_init=logm,
               logc_init=logc,)

projection = ccrs.AzimuthalEquidistant(
    central_longitude=85.5, central_latitude=42.54)
fig, axs = plt.subplots(5, 3, figsize=(15, 25), dpi=300,
                        subplot_kw={'projection': projection})

v = Vis(axs[0, 0], sp_digraph, projection=projection, edge_width=.5,
        edge_alpha=1, edge_zorder=100, sample_pt_size=20,
        obs_node_size=7.5, sample_pt_color="black",
        cbar_font_size=10, cbar_ticklabelsize=8,
        cbar_bbox_to_anchor=(0.15, 0.03, 1, 1), mutation_scale=6)

# v.ax.gridlines(xlocs=[60,90,120], ylocs=[30,45,60], draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, zorder=0)
v.digraph_wrapper(axs, node_scale=[1, 10, 5])

plt.show()
"""
plt.figure(figsize=(8, 6))
plt.plot(np.log10(lamb_grid), cv, 'bo')  
plt.plot(np.log10(lamb_grid_fine), cv_fine, 'bo')  
plt.xlabel(r"$\mathrm{log}_{10}(\mathrm{\lambda})$")
plt.ylabel('CV Error')

print(cv,cv_fine)"""

digraphstats = Digraphstats(sp_digraph)

fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
digraphstats.fitting_wrapper(axs)
plt.show()

fig, ax = plt.subplots(figsize=(5, 3), dpi=300, subplot_kw={
                       'projection': projection})
digraphstats.draw_outliers(v, ax, fdr=0.4)
plt.show()

digraphstats.distribution_wrapper()


print(np.max(sp_digraph.y))
print(np.min(sp_digraph.y))