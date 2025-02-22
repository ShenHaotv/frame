import numpy as np
import scipy.stats as stats
from scipy.sparse import csr_matrix
from .loss import getindex

def benjamini_hochberg(p_values, fdr=0.2):
    """
    Apply the Benjamini-Hochberg procedure to a list of p-values to determine significance
    and the largest k such that p_(k) <= k/m * FDR.
    
    Args:
    p_values (list or array): Array of p-values from multiple hypothesis tests.
    fdr (float): False discovery rate threshold.
    
    Returns:
    tuple:
        array: Boolean array where True indicates the hypotheses that are accepted.
        int: The largest k for which p_(k) <= k/m * FDR.
    """
    m = len(p_values)  # total number of hypotheses
    sorted_p_values = np.sort(p_values)
    sorted_indices = np.argsort(p_values)
    critical_values = np.array([fdr * (i + 1) / (m) for i in range(m)])
    
    # Find the largest p-value that meets the Benjamini-Hochberg criterion
    is_significant = sorted_p_values <= critical_values
    if np.any(is_significant):
        max_significant = np.max(np.where(is_significant)[0])  # max index where condition is true
    else:
        max_significant = -1  # no significant results
    
    # All p-values with rank <= max_significant are significant
    significant_indices = sorted_indices[:max_significant + 1]
    
    # max_significant + 1 because indices are 0-based, but k should be 1-based
    return significant_indices

class Digraphstats(object):
      def __init__(self,sp_digraph):
          self.sp_digraph=sp_digraph
          self.D=csr_matrix(np.triu(self.sp_digraph.distance.copy(),k=1))
          self.D_fit=csr_matrix(np.triu(self.sp_digraph.distance_fit.copy(),k=1))
          self.D_data=self.D.data.copy()
          self.D_fit_data=self.D_fit.data.copy()
          logratio=np.log(self.D_data/self.D_fit_data)
          mean_logratio=np.mean(logratio)
          var_logratio=np.var(logratio,ddof=1)
          z_score=(logratio-mean_logratio)/np.sqrt(var_logratio) 
          self.z_score=z_score
     
      def distance_regression(self,ax,ms=3,labelsize=6, R2_fontsize=10,legend_fontsize=10,xticks=None,yticks=None):         
          coefficients = np.polyfit(self.D_fit_data,self.D_data,1)
          m, b = coefficients
          D_linear=m*self.D_fit_data + b
          
          ax.plot(self.D_fit_data,self.D_data,'ko', ms=ms)  
          ax.tick_params(axis='both', which='major', labelsize=labelsize)
          ax.plot(self.D_fit_data,D_linear,zorder=2, color="orange", linestyle='--', linewidth=1)
          corrcoef=np.corrcoef(self.D_data,self.D_fit_data)[0,1]
          ax.text(0.6, 0.3, "R²={:.3f}".format(corrcoef**2),fontsize=R2_fontsize,transform=ax.transAxes)
          ax.set_xlabel('Fitted genetic distance',fontsize=legend_fontsize)
          ax.set_ylabel('Empirical genetic distance',fontsize=legend_fontsize)
          if xticks is not None:
             ax.set_xticks(xticks)
             ax.set_xticklabels([f'{tick:.2f}' for tick in xticks])
          if yticks is not None:
             ax.set_yticks(yticks)
             ax.set_yticklabels([f'{tick:.2f}' for tick in yticks]) 
      
      def z_score_distribution(self,ax):
          ax.hist(self.z_score, bins='auto', color='blue', alpha=0.7)
          ax.set_xlabel('z-score',fontsize=15)
          
      def draw_heatmap(self,ax):
          abs_Z_triu=self.D.copy()
          abs_Z_triu.data=np.abs(self.z_score)
          abs_Z_matrix=(abs_Z_triu+abs_Z_triu.T).toarray()
          ax.imshow(abs_Z_matrix, cmap='viridis',interpolation='nearest')
          ax.set_xlabel('|z-score|',fontsize=15)
              
      def outlier_detection(self,fdr):
          p_value_neg=stats.norm.cdf(self.z_score)
          p_value_pos=1-p_value_neg
          p_values=np.minimum(p_value_pos,p_value_neg)
          indices=benjamini_hochberg(p_values, fdr=fdr)
          edge_index=getindex(self.D)
          observed_indices=edge_index[indices,:]                               #Indices of edge outlier in the observed nodes                
          h0=self.sp_digraph.h[0]
          outlier_indices=np.zeros((len(indices),2))                           #Indices of outliers in the original digraph
          for i in range(len(indices)):
              outlier_indices[i,0]=h0[int(observed_indices[i,0])]
              outlier_indices[i,1]=h0[int(observed_indices[i,1])]
              
          return(observed_indices, outlier_indices)
      
      def draw_outliers(self,
                        v,
                        ax,
                        indices=None,
                        fdr=None,
                        draw_map=True,
                        draw_nodes=True,):
        
          if fdr is None:
             fdr=0.2
             
          if indices is None:
             observed_indices,outlier_indices=self.outlier_detection(fdr)
          
          v.draw_migration_rates(ax,
                                 mode='Base',
                                 draw_map=draw_map,
                                 draw_nodes=draw_nodes,)
          
          for i in range(len(outlier_indices)):
              pair=outlier_indices[i,:]
              pair_observed=observed_indices[i,:]
              v.ax.plot([v.grid[int(pair[0])][0],v.grid[int(pair[1])][0]],
                        [v.grid[int(pair[0])][1],v.grid[int(pair[1])][1]],
                        color = 'gray', linewidth = 0.5)      
              v.ax.text(v.grid[int(pair[0])][0],v.grid[int(pair[0])][1], str(int(pair_observed[0])),
                        horizontalalignment="center", verticalalignment="bottom",
                        size=v.obs_node_textsize*0.8, zorder=v.obs_node_zorder,)           
              v.ax.text(v.grid[int(pair[1])][0],v.grid[int(pair[1])][1], str(int(pair_observed[1])),
                        horizontalalignment="center", verticalalignment="bottom",
                        size=v.obs_node_textsize*0.8, zorder=v.obs_node_zorder,)
   
