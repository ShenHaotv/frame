import numpy as np
import scipy.stats as stats
from scipy.sparse import csr_matrix
from .loss import getindex

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
                        size=v.obs_node_textsize*0.8, zorder=v.obs_node_zorder,)
   
