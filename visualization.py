from __future__ import absolute_import, division, print_function

import cartopy.feature as cfeature
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import tril,csr_matrix
from matplotlib import ticker
from matplotlib.patches import FancyArrowPatch
from pyproj import Proj

""" Round a positive number to 2 significant figures using ceiling"""
def round_to_2_sig_fig_ceil(number):
    if number==0:
       result=0
    else:
        order_of_magnitude = np.floor(np.log10(number))-1
        scaled_number = number / (10**order_of_magnitude)
        rounded_scaled_number = np.ceil(scaled_number)
        result = rounded_scaled_number * (10**order_of_magnitude)
        result = round(result, -int(order_of_magnitude))
    return result
 
"""Project coordinates"""   
def project_coords(X, proj):
    P = np.empty(X.shape)
    for i in range(X.shape[0]):
        x, y = proj(X[i, 0], X[i, 1])
        P[i, 0] = x
        P[i, 1] = y
    return P

"""Function to add an arrowhead on the arc """
def add_arrowhead_on_arc(ax,start,end,arc_rad,color, mutation_scale):
    x1, y1 = start
    x2, y2 = end
    x12, y12 =0.4*x1+0.6*x2, 0.4*y1+0.6*y2
    dx, dy = x2 - x1, y2 - y1

    # Calculate the position for the arrowhead
    cx, cy = x12 + arc_rad* dy/2, y12 - arc_rad * dx/2

    # Create the arrowhead using FancyArrowPatch
    arrow = FancyArrowPatch(posA=(cx-dx*0.001,cy-dy*0.001), posB=(cx+dx*0.001, cy+dy*0.001), 
                            arrowstyle='fancy', mutation_scale=mutation_scale, color=color,linewidth=0.3)
    ax.add_patch(arrow)
    
class Vis(object):
    def __init__(
        self,
        ax,
        sp_digraph,
        projection=None,
        coastline_m="50m",
        coastline_linewidth=0.5,
        edge_color="#d9d9d9",
        edge_width=1,
        edge_alpha=1.0,
        edge_zorder=2,
        cbar_font_size=12,
        cbar_nticks=3,
        cbar_orientation="horizontal",
        cbar_ticklabelsize=12,
        cbar_width="20%",
        cbar_height="5%",
        cbar_bbox_to_anchor=(0.05, 0.2),
        ell_scaler=np.sqrt(3.0) / 6.0,
        ell_edgecolor="gray",
        ell_lw=0.2,
        ell_abs_max=0.5,
        target_dist_pt_size=10,
        target_dist_pt_linewidth=0.5,
        target_dist_pt_alpha=1.0,
        target_dist_pt_zorder=2,
        seed=1996,
        arc_rad =-0.2,
        mutation_scale=10,
        abs_max_full=None,
        scale=None,
    ):
        """A visualization module 

        Args:
        """
        # main attributes
        self.ax = ax
        self.ax.axis("off")
        self.sp_digraph = sp_digraph
        self.node_pos= sp_digraph.node_pos.copy()
        self.projection = projection
        self.seed = seed
        self.arc_rad=arc_rad
        self.mutation_scale=mutation_scale
        np.random.seed = self.seed

        # ------------------------- Attributes -------------------------
        self.coastline_m = coastline_m
        self.coastline_linewidth = coastline_linewidth

        # edge
        self.edge_width = edge_width
        self.edge_alpha = edge_alpha
        self.edge_zorder = edge_zorder
        self.edge_color = edge_color

        # colorbar
        self.cbar_font_size = cbar_font_size
        self.cbar_nticks = cbar_nticks
        self.cbar_orientation = cbar_orientation
        self.cbar_ticklabelsize = cbar_ticklabelsize
        self.cbar_width = cbar_width
        self.cbar_height = cbar_height
        self.cbar_bbox_to_anchor = cbar_bbox_to_anchor
        
        # target correlations
        self.target_dist_pt_si_radius=target_dist_pt_size
        self.target_dist_pt_linewidth = target_dist_pt_linewidth
        self.target_dist_pt_alpha = target_dist_pt_alpha
        self.target_dist_pt_zorder = target_dist_pt_zorder
        self.abs_max_full=abs_max_full
        
        # colors
        self.colors = ["#994000",
                       "#CC5800",
                       "#FF8F33",
                       "#FFAD66",
                       "#FFCA99",
                       "#FFE6CC",
                       "#FBFBFB",
                       "#CCFDFF",
                       "#99F8FF",
                       "#66F0FF",
                       "#33E4FF",
                       "#00AACC",
                       "#007A99",]  
        
        
        self.edge_cmap = clr.LinearSegmentedColormap.from_list(
             "colors", self.colors, N=256)
         
        # plotting maps
        if self.projection is not None:
           self.proj = Proj(projection.proj4_init)
           self.node_pos = project_coords(self.node_pos, self.proj)
        self.weights_assignment()
       
    # ------------------------- Helping functions -------------------------        
    
    def weights_assignment(self):
        # edge weights
        M=self.sp_digraph.M.copy()

        M=csr_matrix(M)
        self.weights=M.data
        self.log_weights=np.log10(self.weights)
        self.log_weights_mean=np.mean(np.log10(self.weights))
        self.norm_log_weights=np.log10(self.weights) - np.mean(np.log10(self.weights))
        self.n_params = int(len(self.weights) / 2)  
       
        # extract node positions on the lattice
        self.idx_full=self.sp_digraph.M.nonzero()   
        
        # Parameters for base graph and difference graph
        M_log=M.copy()
        M_log.data=np.log10(M.data)
        M_base=0.5*(M_log+M_log.T)
        M_diff=M_log-M_log.T
        M_base=tril(M_base, k=-1)
        M_base.eliminate_zeros()
        M_diff.data[M_diff.data<0]=0
        M_diff.eliminate_zeros()
        self.idx_base=M_base.nonzero()
        self.idx_diff=M_diff.nonzero()
        self.norm_log_weights_diff=M_diff.data
        
        if self.abs_max_full is None:
           self.range_full= round_to_2_sig_fig_ceil(np.max(np.abs(self.norm_log_weights)))
        else:
            self.range_full=self.abs_max_full
       
        self.vmin_full=-self.range_full
        self.vmax_full=self.range_full
            
        self.edge_norm_full=clr.Normalize(vmin=self.vmin_full,vmax=self.vmax_full,clip=True) 
        
    """Draws the underlying map projection"""        
    def draw_map(self):
        self.ax.add_feature(cfeature.LAND, facecolor="#f7f7f7", zorder=0)
        self.ax.coastlines(
            self.coastline_m,
            color="#636363",
            linewidth=self.coastline_linewidth,
            zorder=0,
        )
            
      
    """Draw the edges for different representations"""
    def draw_edges(self, mode=None):
        if mode=='Full':
           nx.draw(self.sp_digraph,
                   ax=self.ax,
                   node_size=0.1,
                   edge_cmap=self.edge_cmap,
                   alpha=self.edge_alpha,
                   pos=self.node_pos,
                   width=self.edge_width,
                   edgelist=list(np.column_stack(self.idx_full)),
                   edge_color=self.norm_log_weights,
                   edge_vmin=self.vmin_full,
                   edge_vmax=self.vmax_full,
                   connectionstyle=f'arc3, rad = {self.arc_rad}',
                   arrowstyle='-',
                   arrows=True, 
                   arrowsize=2,)
            
           for edge, weight in zip(np.column_stack(self.idx_full), self.norm_log_weights):               
               start_node, end_node = edge
               start_pos, end_pos = self.node_pos[start_node], self.node_pos[end_node]
               color=self.edge_cmap(self.edge_norm_full(weight))
               add_arrowhead_on_arc(self.ax,start_pos, end_pos, self.arc_rad,color,self.mutation_scale)
                 
        else:
             nx.draw(
                self.sp_digraph,
                ax=self.ax,
                node_size=0.0,
                alpha=self.edge_alpha,
                pos=self.node_pos,
                width=self.edge_width,
                edgelist=list(np.column_stack(self.idx_base)),
                edge_color=self.edge_color,
                arrowstyle='-', 
                arrowsize=2,)

    """Draw colorbar"""
    def draw_edge_colorbar(self):
        self.edge_norm_rep=clr.LogNorm(vmin=1,vmax=100)
        self.edge_sm = plt.cm.ScalarMappable(cmap=self.edge_cmap, norm=self.edge_norm_rep)
        
        self.edge_sm._A = []
        x0=self.cbar_bbox_to_anchor[0]
        y0=self.cbar_bbox_to_anchor[1]
        width = float(self.cbar_width.strip('%')) / 100
        height = float(self.cbar_height.strip('%')) / 100
        
        self.edge_axins = self.ax.inset_axes([x0, y0, width, height],transform=self.ax.transAxes)
        self.edge_cbar = plt.colorbar(self.edge_sm, cax=self.edge_axins, 
                                      orientation=self.cbar_orientation)
        self.edge_tick_locator = ticker.LogLocator(base=10,numticks=self.cbar_nticks)
        self.edge_cbar.locator = self.edge_tick_locator
        cbar_min, cbar_max = self.edge_sm.get_clim()
       
        self.edge_cbar.set_ticks([cbar_min,10, cbar_max])             
        self.edge_cbar.ax.tick_params(which="minor", length=0)
        
        v_mean=round(self.log_weights_mean,1)
        v_max=round(v_mean+self.vmax_full,1)
        v_min=round(v_mean+self.vmin_full,1)
              
        self.edge_cbar.set_ticklabels([f'${{{v_min}}}$',
                                       f'${{{v_mean}}}$',
                                       f'${{{v_max}}}$'],fontsize=self.cbar_ticklabelsize)
           
        self.edge_cbar.ax.set_title(r"$\mathrm{log}_{10}\left(\mathrm{m}\right)$",  loc="center", fontsize=self.cbar_font_size)              
        self.edge_cbar.ax.tick_params(labelsize=self.cbar_ticklabelsize)
     
    # ------------------------- Plotting Functions -------------------------             
    """Draw different representations of migration rates"""
    def draw_migration_rates(self,
                             ax,
                             mode,
                             draw_map=True,
                             set_title=True,
                             draw_colorbar=True,
                             draw_colorcompass=True,
                             title_font_size=10,
                             ):
        self.ax = ax
        
        if draw_map is True:
           self.draw_map()
           
        self.draw_edges(mode=mode)

        if draw_colorbar is True:           
           self.draw_edge_colorbar()
     
        if set_title is True:
           ax.set_title('Miration rate',pad=3,fontsize=title_font_size)
           
    def draw_ne(self,
                ax,
                node_scale,
                draw_map=True,
                set_title=True,
                title_font_size=10):
        
        self.ax=ax
        if draw_map is True:
           self.draw_map()
 
        ne=self.sp_digraph.ne
        ne_min=np.min(ne)
        node_size=node_scale*np.log2(ne/ne_min+1)    
            
        nx.draw(self.sp_digraph,
                ax=self.ax,
                node_size=node_size,
                alpha=self.edge_alpha,
                pos=self.node_pos,
                width=self.edge_width,
                edgelist=list(np.column_stack(self.idx_base)),
                edge_color=self.edge_color,
                arrows=True, 
                arrowstyle='-',
                arrowsize=2,)
       
        if set_title is True:
           ax.set_title('Ne',pad=3,fontsize=title_font_size)
           