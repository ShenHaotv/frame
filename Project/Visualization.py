from __future__ import absolute_import, division, print_function

import cartopy.feature as cfeature
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import tril,csr_matrix
from matplotlib import ticker
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Proj

def round_to_2_sig_fig_ceil(number):
    order_of_magnitude = np.floor(np.log10(number))-1
    scaled_number = number / (10**order_of_magnitude)
    rounded_scaled_number = np.ceil(scaled_number)
    result = rounded_scaled_number * (10**order_of_magnitude)
    result = round(result, -int(order_of_magnitude))
    return result
    
def project_coords(X, proj):
    """Project coordinates"""
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
    
"""Function to add an arrowhead on the line segment """
def add_arrowhead_on_linesegment(ax,start,end,color,alpha, mutation_scale):
    x1, y1 = start
    x2, y2 = end
    x12, y12 = 0.4*x1+0.6*x2, 0.4*y1+0.6*y2
    dx, dy = x2 - x1, y2 - y1

    # Calculate the position for the arrowhead
    cx, cy = x12 , y12

    # Create the arrowhead using FancyArrowPatch
    arrow = FancyArrowPatch(posA=(cx-dx*0.001,cy-dy*0.001), posB=(cx+dx*0.001, cy+dy*0.001), 
                            arrowstyle='fancy', mutation_scale=mutation_scale, color=color,alpha=alpha,linewidth=0.3)
    ax.add_patch(arrow)   
    
def add_arrow(ax,start,end,color,mutation_scale):
    # Create the arrowhead using FancyArrowPatch

    arrow = FancyArrowPatch(posA=start-(end-start)*0.1, posB=end, 
                            arrowstyle='Simple', mutation_scale=mutation_scale, color=color,linewidth=0.3)
    ax.add_patch(arrow)  

class Vis(object):
    def __init__(
        self,
        ax,
        sp_digraph,
        projection=None,
        coastline_m="50m",
        coastline_linewidth=0.5,
        sample_pt_size=1,
        sample_pt_linewidth=0.5,
        sample_pt_color="#d9d9d9",
        sample_pt_jitter_std=0.0,
        sample_pt_alpha=1.0,
        sample_pt_zorder=2,
        obs_node_size=10,
        obs_node_textsize=7,
        obs_node_linewidth=0.5,
        obs_node_color="#d9d9d9",
        obs_node_alpha=1.0,
        obs_node_zorder=2,
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
        cbar_loc="lower left",
        cbar_bbox_to_anchor=(0.05, 0.2,1,1),
        campass_font_size=10,
        campass_radius=0.15,
        campass_bbox_to_anchor=(0.05, 0.1),
        ell_scaler=np.sqrt(3.0) / 6.0,
        ell_edgecolor="gray",
        ell_lw=0.2,
        ell_abs_max=0.5,
        target_dist_pt_size=10,
        target_dist_pt_linewidth=0.5,
        target_dist_pt_alpha=1.0,
        target_dist_pt_zorder=2,
        seed=1996,
        arc_rad = 0.2,
        mutation_scale=10,
        abs_max=None,
        scale=None,
    ):
        """A visualization module 

        Args:
        """
        # main attributes
        self.ax = ax
        self.ax.axis("off")
        self.sp_digraph = sp_digraph
        self.grid = sp_digraph.node_pos.copy()
        self.coord = sp_digraph.sample_pos.copy()
        self.projection = projection
        self.seed = seed
        self.arc_rad=arc_rad
        self.mutation_scale=mutation_scale
        np.random.seed = self.seed

        # ------------------------- Attributes -------------------------
        self.coastline_m = coastline_m
        self.coastline_linewidth = coastline_linewidth

        # sample pt
        self.sample_pt_size = sample_pt_size
        self.sample_pt_linewidth = sample_pt_linewidth
        self.sample_pt_color = sample_pt_color
        self.sample_pt_zorder = sample_pt_zorder
        self.samplte_pt_alpha = sample_pt_alpha
        self.sample_pt_jitter_std = sample_pt_jitter_std
        self.sample_pt_alpha = sample_pt_alpha

        # obs nodes
        self.obs_node_size = obs_node_size
        self.obs_node_textsize = obs_node_textsize
        self.obs_node_alpha = obs_node_alpha
        self.obs_node_linewidth = obs_node_linewidth
        self.obs_node_color = obs_node_color
        self.obs_node_zorder = obs_node_zorder

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
        self.cbar_loc = cbar_loc
        self.cbar_bbox_to_anchor = cbar_bbox_to_anchor
        
        #color campass
        self.campass_font_size = campass_font_size
        self.campass_radius = campass_radius 
        self.campass_bbox_to_anchor =campass_bbox_to_anchor
        

        # target correlations
        self.target_dist_pt_si_radius=target_dist_pt_size
        self.target_dist_pt_linewidth = target_dist_pt_linewidth
        self.target_dist_pt_alpha = target_dist_pt_alpha
        self.target_dist_pt_zorder = target_dist_pt_zorder
        self.abs_max=abs_max
        
        self.weights_assignment()
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
        
        self.colors_summary= ["#CCFDFF",
                              "#99F8FF",
                              "#66F0FF",
                              "#33E4FF",
                              "#00AACC",
                              "#007A99",]  
        
        self.edge_cmap = clr.LinearSegmentedColormap.from_list(
             "colors", self.colors, N=256)
        
        self.edge_cmap_summary = clr.LinearSegmentedColormap.from_list(
             "colors_summary", self.colors_summary, N=256)
         
        theta = np.linspace(0, 2 * np.pi, 256)  
        red = np.abs(1 - theta / np.pi) + np.sin(theta) / 4
        blue = 1 + np.sin(theta) / 2 - red
        green = np.zeros_like(theta)  
 
        self.colors_diff= np.vstack([red, green, blue]).T
        self.edge_cmap_diff=clr.LinearSegmentedColormap.from_list(
             "colors", self.colors_diff, N=256)
                        
        self.dist_cmap = plt.get_cmap("viridis_r")

        # plotting maps
        if self.projection is not None:
           self.proj = Proj(projection.proj4_init)
           self.coord = project_coords(self.coord, self.proj)
           self.grid = project_coords(self.grid, self.proj)
           
        self.weights_assignment()
         
    # ------------------------- Helping functions -------------------------        
    
    def weights_assignment(self):
        # edge weights
        M=self.sp_digraph.M.copy()
        d=M.shape[0]

        M=csr_matrix(M)
        self.weights=M.data
                           
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
        self.norm_log_weights_base=M_base.data-np.mean(np.log10(self.weights))
        self.norm_log_weights_diff=M_diff.data

        
        if self.abs_max is None:
           absmax=np.max(np.abs(self.norm_log_weights))
           self.range=round_to_2_sig_fig_ceil(absmax)           
        else:
            self.range=self.abs_max
                   
        self.vmin=-self.range
        self.vmax=self.range
        
        self.edge_norm=clr.Normalize(vmin=self.vmin,vmax=self.vmax) 
        
        d=M.shape[0]
        self.edge_angle_radians_matrix=np.zeros((d,d))
        for s in range(d):
            for t in range(d):
                if M_diff[s,t]>0:
                   dx,dy=self.sp_digraph.node_pos[t]-self.sp_digraph.node_pos[s]
                   angle_radian = np.arctan2(dy, dx)
                   if angle_radian < 0:
                      angle_radian+= 2*np.pi
                   self.edge_angle_radians_matrix[s,t]=angle_radian+0.1
        self.edge_angle_radians_matrix=csr_matrix(self.edge_angle_radians_matrix)
        self.edge_angle_radians=self.edge_angle_radians_matrix.data-0.1
        
        if len(self.norm_log_weights_diff)>0:
           self.vmin_diff=0      
           self.vmax_diff=np.max(self.norm_log_weights_diff)
           self.vmax_diff=round_to_2_sig_fig_ceil(self.vmax_diff)
                               
           self.edge_norm_color_diff=clr.Normalize(vmin=0.0, vmax=2*np.pi)
           self.edge_norm_alpha_diff=clr.Normalize(vmin=self.vmin_diff, vmax=self.vmax_diff)
           self.alpha_diff=self.edge_norm_alpha_diff(self.norm_log_weights_diff)
        else:
            self.vmax_diff=0
        
        #Parameters for summary graph
        norm_vector=np.zeros(d)
        start_positions=[]
        end_positions=[]
          
        for node in self.sp_digraph.nodes():
            start_position=self.grid[node]
            start_positions.append(start_position)
              
            summary_vector = np.array([0, 0])
 
            neighbors = list(self.sp_digraph.neighbors(node))
            edge_length = np.zeros(len(neighbors))
 
            for i in range(len(neighbors)):
                neighbor=neighbors[i]
                coordinate_difference = self.grid[neighbor] - self.grid[node]
                edge_length[i] = np.linalg.norm(coordinate_difference)
                summary_vector=summary_vector+ (M[node, neighbor] * coordinate_difference) / edge_length[i]
 
            norm_vector[node] = np.linalg.norm(summary_vector)
              
            normalized_summary_vector = summary_vector / norm_vector[node]
            end_position = self.grid[node]+ normalized_summary_vector * np.mean(edge_length)*0.75
            end_positions.append(end_position)
        
        self.weight_summary=np.log10(norm_vector)
        self.summary_start_positions=start_positions
        self.summary_end_positions=end_positions
        self.vmax_summary=np.max(self.weight_summary)
        self.vmin_summary=np.min(self.weight_summary)
        self.edge_norm_color_summary=clr.Normalize(vmin=self.vmin_summary, vmax=self.vmax_summary)
            
    def draw_map(self):
        """Draws the underlying map projection"""
        self.ax.add_feature(cfeature.LAND, facecolor="#f7f7f7", zorder=0)
        self.ax.coastlines(
            self.coastline_m,
            color="#636363",
            linewidth=self.coastline_linewidth,
            zorder=0,
        )

    def draw_samples(self):
        """Draw the individual sample coordinates"""
        jit_coord = self.coord + np.random.normal(
            loc=0.0, scale=self.sample_pt_jitter_std, size=self.coord.shape
        )
        self.ax.scatter(
            jit_coord[:, 0],
            jit_coord[:, 1],
            edgecolors="black",
            linewidth=self.sample_pt_linewidth,
            s=self.sample_pt_size,
            alpha=self.sample_pt_alpha,
            color=self.sample_pt_color,
            marker=".",
            zorder=self.sample_pt_zorder,
        )

    def draw_obs_nodes(self, use_ids=None):
        """Draw the observed node coordinates"""     
        obs_ids = self.sp_digraph.h[0]
        obs_grid = self.grid[obs_ids, :]
        if use_ids:
           for i, j in enumerate(obs_ids):
               self.ax.text(
                    obs_grid[i, 0],
                    obs_grid[i, 1],
                    str(j),
                    horizontalalignment="center",
                    verticalalignment="center",
                    size=self.obs_node_textsize,
                    zorder=self.obs_node_zorder,)
        else:
            self.ax.scatter(
                 obs_grid[:, 0],
                 obs_grid[:, 1],
                 edgecolors="black",
                 linewidth=self.obs_node_linewidth,
                 s=self.obs_node_size * np.sqrt(self.sp_digraph.h[1]),
                 alpha=self.obs_node_alpha,
                 color=self.obs_node_color,
                 zorder=self.obs_node_zorder,)
                 
    def draw_edges(self, use_weights):
        """Draw the edges of the graph"""
        if use_weights=='Full':
           nx.draw(self.sp_digraph,
                   ax=self.ax,
                   node_size=0.1,
                   edge_cmap=self.edge_cmap,
                   alpha=self.edge_alpha,
                   pos=self.grid,
                   width=self.edge_width,
                   edgelist=list(np.column_stack(self.idx_full)),
                   edge_color=self.norm_log_weights,
                   edge_vmin=self.vmin,
                   edge_vmax=self.vmax,
                   connectionstyle=f'arc3, rad = {self.arc_rad}',
                   arrowstyle='-',
                   arrows=True, 
                   arrowsize=2,)
            
           for edge, weight in zip(np.column_stack(self.idx_full), self.norm_log_weights):               
               start_node, end_node = edge
               start_pos, end_pos = self.grid[start_node], self.grid[end_node]
               color=self.edge_cmap(self.edge_norm(weight))
               add_arrowhead_on_arc(self.ax,start_pos, end_pos, self.arc_rad,color,self.mutation_scale)
          
        elif use_weights=='Base':
             nx.draw(self.sp_digraph,
                     ax=self.ax,
                     node_size=0.1,
                     edge_cmap=self.edge_cmap,
                     alpha=self.edge_alpha,
                     pos=self.grid,
                     width=5*self.edge_width,
                     edgelist=list(np.column_stack(self.idx_base)),
                     edge_color=self.norm_log_weights_base,
                     edge_vmin=self.vmin,
                     edge_vmax=self.vmax,
                     arrowstyle='-', 
                     arrowsize=2,)      
                     
        elif use_weights=='Difference':
             if len(self.norm_log_weights_diff)>0:
                nx.draw(
                self.sp_digraph,
                ax=self.ax,
                node_size=0.1,
                edge_cmap=self.edge_cmap_diff,
                alpha=self.alpha_diff,
                pos=self.grid,
                width=2.5*self.edge_width,
                edgelist=list(np.column_stack(self.idx_diff)),
                edge_color=self.edge_angle_radians,
                edge_vmin=0.0,
                edge_vmax=2*np.pi,
                arrowstyle='-',
                arrowsize=2,
                )
                 
                for edge, weight,alpha in zip(np.column_stack(self.idx_diff),self.edge_angle_radians,self.alpha_diff):               
                    start_node, end_node = edge
                    start_pos, end_pos = self.grid[start_node], self.grid[end_node]
                    color=self.edge_cmap_diff(self.edge_norm_color_diff(weight))
                    add_arrowhead_on_linesegment(self.ax,start_pos,end_pos,color,alpha,2*self.mutation_scale)
                    
             else:
                 nx.draw(self.sp_digraph,
                         ax=self.ax,
                         node_size=0.1,
                         alpha=self.edge_alpha,
                         pos=self.grid,
                         width=self.edge_width,
                         edgelist=list(np.column_stack(self.idx_base)),
                         edge_color=self.edge_color,
                         arrows=True, 
                         arrowstyle='-',
                         arrowsize=2,)
                 
        elif use_weights=='Summary':
             nx.draw(
                self.sp_digraph,
                ax=self.ax,
                node_size=0.0,
                alpha=self.edge_alpha,
                pos=self.grid,
                width=self.edge_width,
                edgelist=list(np.column_stack(self.idx_base)),
                edge_color=self.edge_color,
                arrows=True, 
                arrowstyle='-',
                arrowsize=2,)
             
             for node in self.sp_digraph.nodes():
                 start_pos=self.summary_start_positions[node]
                 end_pos=self.summary_end_positions[node]
                 color =self.edge_cmap_summary(self.edge_norm_color_summary(self.weight_summary[node]))
                 add_arrow(self.ax,start_pos,end_pos,color,1.5*self.mutation_scale)


    def draw_edge_colorbar(self, use_weights):
        """Draw colorbar"""
        self.edge_norm_rep=clr.LogNorm(vmin=1,vmax=100)
        if use_weights=='Summary':
           self.edge_sm = plt.cm.ScalarMappable(cmap=self.edge_cmap_summary, norm=self.edge_norm_rep)
        else:
            self.edge_sm = plt.cm.ScalarMappable(cmap=self.edge_cmap, norm=self.edge_norm_rep)
        
        self.edge_sm._A = []
        self.edge_axins = inset_axes(
            self.ax,
            width=self.cbar_width,
            height=self.cbar_height,
            loc=self.cbar_loc,
            bbox_to_anchor=self.cbar_bbox_to_anchor,
            bbox_transform=self.ax.transAxes,
            borderpad=0,)
        
        self.edge_cbar = plt.colorbar(self.edge_sm, cax=self.edge_axins, 
                                      orientation=self.cbar_orientation)
        self.edge_tick_locator = ticker.LogLocator(base=10,numticks=self.cbar_nticks)
        self.edge_cbar.locator = self.edge_tick_locator
        cbar_min, cbar_max = self.edge_sm.get_clim()
       
        if use_weights=='Summary':
           self.edge_cbar.set_ticks([cbar_min,cbar_max])
           range_summary=round_to_2_sig_fig_ceil(self.vmax_summary-self.vmin_summary)
           self.edge_cbar.set_ticklabels([r'$0$',
                                          f'${{{range_summary}}}$'],fontsize=self.cbar_ticklabelsize)
           
        else:
            self.edge_cbar.set_ticks([cbar_min,10, cbar_max])
            self.edge_cbar.set_ticklabels([f'${{{-self.range}}}$',
                                           r'$0$', 
                                           f'${{{self.range}}}$'],fontsize=self.cbar_ticklabelsize)
             
        self.edge_cbar.ax.tick_params(which="minor", length=0)
        if use_weights=='Full':
           self.edge_cbar.ax.set_title("$\mathrm{log}_{10}(\mathrm{m})$", 
                                        loc="center", fontsize=self.cbar_font_size)

        elif use_weights=='Base':
             self.edge_cbar.ax.set_title(r"$\overline{\mathrm{log}_{10}(\mathrm{m})}$", 
                                         loc="center", fontsize=self.cbar_font_size)
             
        elif use_weights=='Summary':
             self.edge_cbar.ax.set_title(r"$\mathrm{log}_{10}(\mathrm{m_{s}})$", 
                                         loc="center", fontsize=self.cbar_font_size)
        
        self.edge_cbar.ax.tick_params(labelsize=self.cbar_ticklabelsize)
        
    def draw_edge_colorcampass(self):
        bbox = self.ax.get_position()
        inset_width = bbox.width * self.campass_radius  
        inset_height = bbox.height * self.campass_radius  
        
        inset_left = bbox.x0+self.campass_bbox_to_anchor[0]*bbox.width 
        inset_bottom = bbox.y0+self.campass_bbox_to_anchor[1]*bbox.height
        
        radius=1
        theta = np.linspace(0, 2 * np.pi, 256)
        r = np.linspace(0, radius, 256)
        theta, r = np.meshgrid(theta, r)

        red_theta=np.abs(1-theta/np.pi)+np.sin(theta)/4
        blue_theta=1+np.sin(theta)/2-red_theta
        green_theta=0

        # Calculate RGB components
        red = 1-(r/radius)+(r/radius)*red_theta
        blue =1-(r/radius)+(r/radius)*blue_theta
        green = 1-(r/radius)+(r/radius)*green_theta

        # Create the color array
        colors = np.stack((red, green, blue), axis=2)
        
        # Create polar axes at the calculated position
        self.campass_axins=self.ax.figure.add_axes([inset_left, inset_bottom, 
                                                    inset_width, inset_height], 
                                                    projection='polar')

        self.campass_axins.pcolormesh(theta, r, np.rad2deg(theta), 
                                   color=colors.reshape(-1, 3), shading='auto')
        
        self.campass_axins.set_yticklabels([])

        # Customizing ticks
        self.campass_axins.set_xticks(np.pi/180. * np.linspace(0,  360, 4, endpoint=False))
          
        self.campass_axins.set_xticklabels([f'${{{self.vmax_diff}}}$', 
                                            "$\mathrm{△log}_{10}(\mathrm{m})$",
                                            '', 'S'],fontsize=self.campass_font_size)
               
    # ------------------------- Plotting Functions -------------------------             
     
    def draw_migration_rates(self,
                             ax,
                             mode,
                             draw_map=True,
                             draw_nodes=True,):
        self.ax = ax
        if draw_map is True:
           self.draw_map()
        self.draw_edges(use_weights=mode)
        if mode=='Difference':
           self.draw_edge_colorcampass()
        else:
            self.draw_edge_colorbar(use_weights=mode,)
        if draw_nodes is True:
           self.draw_obs_nodes()
           
        ax.set_title(f"{mode} graph")

    def draw_migration_rates_wrapper(self,
                                     axs,
                                     draw_map=True,
                                     draw_nodes=True):
        mode=['Base','Full','Difference','Summary']
        for i in range(4):  
            self.draw_migration_rates(ax=axs[i],
                                      mode=mode[i],
                                      draw_map=draw_map,
                                      draw_nodes=draw_nodes)
           
    def draw_attributes(self,
                        ax,
                        node_scale,
                        attribute,
                        draw_map=True,):
        self.ax=ax
        if draw_map is True:
           self.draw_map()

        node_size=0.1
        
        if attribute=='Stationary Distribution':
             y=self.sp_digraph.y.copy()
             logy_min=np.log(np.min(y))
             logy_max=np.log(np.max(y))
             node_size=5*(1+node_scale*(np.log(y)-logy_min)/(logy_max-logy_min))
        elif attribute=='Coalescent Rate':  
             gamma=self.sp_digraph.gamma.copy()
             loggamma_min=np.log(np.min(gamma))
             loggamma_max=np.log(np.max(gamma))
             node_size=5*(1+node_scale*(np.log(gamma)-loggamma_min)/(loggamma_max-loggamma_min))
            
        nx.draw(self.sp_digraph,
                ax=self.ax,
                node_size=node_size,
                alpha=self.edge_alpha,
                pos=self.grid,
                width=self.edge_width,
                edgelist=list(np.column_stack(self.idx_base)),
                edge_color=self.edge_color,
                arrows=True, 
                arrowstyle='-',
                arrowsize=2,)
       
        if attribute=='Sample Size and Position':
           self.draw_obs_nodes()
           self.draw_samples()
           
        elif attribute=='Heterozygosity':
             het=self.sp_digraph.average_heterozygosity
             obs_ids = self.sp_digraph.h[0]
             obs_grid = self.grid[obs_ids, :]
             het_min=np.min(het)
             het_max=np.max(het)
             ax.scatter(obs_grid[:, 0],
                        obs_grid[:, 1],
                        s=5*(1+node_scale*(het-het_min)/(het_max-het_min)),
                        alpha=self.obs_node_alpha,)

        ax.set_title(f"{attribute}")
            
    def draw_attributes_wrapper(self,
                                axs,
                                node_scale=[5,5,5],
                                draw_map=True,):
         attribute=['Sample Size and Position',
                    'Heterozygosity',
                    'Stationary Distribution',
                    'Coalescent Rate']
         
         for i in range(4):
             if i==0:
                scale=None
             else:
                 scale=node_scale[i-1]
             self.draw_attributes(axs[i],
                                  scale,
                                  attribute[i],
                                  draw_map=draw_map,)
             
    def digraph_wrapper(self,
                        axs,
                        node_scale,
                        draw_map=True,
                        draw_nodes=True,):
                 
        self.draw_migration_rates_wrapper([axs[0, 0], axs[0, 1], axs[0, 2],axs[0,3]],
                                                  draw_map=draw_map,
                                                  draw_nodes=draw_nodes)
           
        self.draw_attributes_wrapper([axs[1, 0], axs[1, 1], axs[1, 2],axs[1,3]],
                                     node_scale=node_scale,
                                     draw_map=draw_map)
       