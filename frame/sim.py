import itertools as it
import networkx as nx
import numpy as np
import msprime
from discreteMarkovChain import markovChain

class Sim(object):
     
      def __init__(self,
                   n_rows=9,
                   n_columns=9,
                   n_samples_per_node=10,
                   node_sample_prob=1,
                   semi_random_sampling=None,):
          
        """
        Initializes a new Simulation instance with specific lattice dimensions and sampling parameters.

        Parameters:
           n_rows (int): Number of rows in the lattice, determining the vertical size of the grid.
           n_columns (int): Number of columns in the lattice, determining the horizontal size of the grid.
           n_samples_per_node (int): Total number of samples per node, specifying how many individuals are simulated in each node.
           node_sample_prob (float): Probability that a node will be included in the sample, affecting how nodes are randomly selected.
           semi_random_sampling (bool): If True, enables a semi-random sampling method where only nodes with even coordinates are sampled.

        Notes:
         - This method sets up the basic properties of the simulation object, including the structure of the spatial grid (lattice) and parameters for population sampling.
         
         """
        
        self.n_rows=n_rows
        self.n_columns=n_columns
        self.n_samples_per_node=n_samples_per_node
        self.node_sample_prob=node_sample_prob
        self.semi_random_sampling=semi_random_sampling
             
   
      def setup_digraph(self,
                        m_base=0.1,
                        m_low=0.3,
                        m_high=3,
                        m_topo=3,
                        boundary=None,
                        directional=None,
                        converging=None,
                        diverging=None,
                        circle=None,
                        ):  

          """
          Sets up a directed graph (digraph) for simulating migration patterns in a spatial population model.

          Parameters:
             m_base (float): Base migration rate.
             m_low (float): Multiplier for lower migration rate areas.
             m_high (float): Multiplier for higher migration rate areas. 
             m_topo (float): Mulyiplier for topological patterns.
             boundary (list): Specifies regions with different migration rates.
             directional (list): Specifies directionally migrating lineages.
             converging (list): Specifies areas acting as zone of spatially converging lineages.
             diverging (list): Specifies areas acting as zone of spatially converging lineages.
             circle (list): Specifies nodes forming zone of cyclic rotating lineages.

          Notes:
           - The function initializes a triangular lattice graph and converts it to a directed graph.
           - Migration rates are adjusted based on specified patterns.
           - Sample sizes for populations are set based on the sampling model.
           
           """       
          # Generate a base triangular lattice graph and convert to a directed graph                                          
          graph = nx.generators.lattice.triangular_lattice_graph(
           self.n_rows - 1, 2 * self.n_columns - 2, with_positions=True)
     
          digraph=nx.DiGraph(graph)
          
          # Set default migration weights and modify them based on boundary conditions
          for (u, v) in digraph.edges():
              digraph[u][v]['weight'] =m_base
              if boundary!=None:
                 a=max(boundary)
                 b=min(boundary)
                 if u[1]>a or v[1]>a:
                    digraph[u][v]['weight']*=m_low
                 if u[1]<b or v[1]<b:
                    digraph[u][v]['weight']*=m_high
     
          # Apply high migration rates to directional flows
          if directional!=None:
             for (startpoint,length,direction) in directional:
                 x=startpoint                                                  #Start point of the directional pattern
                 l=length                                                      #length of the directional pattern 
                 if direction=='E':                                            #East direction                        
                    for i in range(x[0],x[0]+l) :                                                                                                                   
                        digraph[(i,x[1])][(i+1,x[1])]['weight']*=m_topo 
                 elif direction=='W':                                          #West direction                                                    
                      for i in range(x[0]-l,x[0]) :                                                                                                                   
                          digraph[(i+1,x[1])][(i,x[1])]['weight']*=m_topo      
                          
          # Set up migration sinks                     
          if converging!=None:
             for (si,r) in converging:                                         #center and radius of the pattern of spatially converging lineages        
                 a={si}
                 b={si}
                 for i in range(r):
                     c=set()
                     for x in a:
                         for y in list(digraph.neighbors(x)):
                             if y not in b:
                                c.add(y)
                                digraph[y][x]['weight']*=m_topo
                     b=a.union(c)
                     a=c
            
          # Set up migration sources
          if diverging!=None:
             for (so,r) in diverging:                                          #center and radius of the pattern of spatially diverging lineages      
                 a={so} 
                 b={so}
                 for i in range(r):
                     c=set()
                     for x in a:
                         for y in list(digraph.neighbors(x)):
                             if y not in b:
                                c.add(y)
                                digraph[x][y]['weight']*=m_topo
                     b=a.union(c)
                     a=c
                     
          # Define cyclic migration routes    
          if circle!=None:
             for ci in circle:
                 for i in range(len(ci)):
                     if i<len(ci)-1:
                        a=ci[i]
                        b=ci[i+1]
                        digraph[a][b]['weight']*=m_topo
                     else:
                         a=ci[i]
                         b=ci[0]
                         digraph[a][b]['weight']*=m_topo
        
          # Set node sample sizes based on the sampling model
          for x in digraph.nodes():
              digraph.nodes[x]['sample_size']=0
              Nx=self.n_samples_per_node                                        #The number of individuals sampled in each sampled deme
              if self.semi_random_sampling is True:
                 if x[0]%2==0 and x[1]%2==0:                    
                    digraph.nodes[x]['sample_size']=Nx   
              else:     
                  Ix=np.random.binomial(1, self.node_sample_prob)              #Index of whether a node is sampled
                  digraph.nodes[x]['sample_size']=Ix*Nx  
   
          digraph = nx.convert_node_labels_to_integers(digraph)
          
          self.digraph=digraph
          self.new_index_map=None                                              
               
          graph = self.digraph.to_undirected()
          pos_dict = nx.get_node_attributes(self.digraph, 'pos')
    
          grid = np.array(list(pos_dict.values()))
          edges = np.array(graph.edges)

          # create sample coordinates array
          sample_sizes_dict = nx.get_node_attributes(self.digraph, 'sample_size')
          pops = [[i] * int(sample_sizes_dict[i]) for i in self.digraph.nodes]
          pops = list(it.chain.from_iterable(pops))
          coord = grid[pops, :]
                
          self.coord=coord
          self.grid=grid
          self.edges=edges
      
          return ()
      
      
      def compute_level_set(self,
                            origin):
            
          """
          Compute the level set and assign potential ancestors to nodes.

          Parameters:
             origin (list): The origin points of demographic events represented as (column, row) indices.
       
          Notes:
           - This function modifies the directed graph associated with the instance by updating nodes' 'level' 
             and 'ancestors' attributes and by storing level sets within the graph's metadata. 
              
          """
                  
          # Initialize a set to hold converted origin indices       
          origin_set=set()
           
          # Convert origin coordinates into graph indices and add to the origin set
          for x in origin:
              new_index=self.n_columns*x[1]+x[0]
              if self.new_index_map is not None:
                 new_index=self.new_index_map[new_index]
              origin_set.add(new_index)
           
          # Prepare the first level set with origin indices, and other sets for processing
          Level_sets = [set()]     
          Level_sets = [set()]
          boundary=origin_set
          existing_set=boundary.copy()
          Level_sets[0]=boundary.copy()
          instrumental_set=boundary.copy()
                     
          # Set the level of origin nodes to 0
          for i in origin_set:
              self.digraph.nodes[i]['level'] = 0
           
          # Initialize ancestors set for all nodes in the digraph
          for i in self.digraph.nodes():
              self.digraph.nodes[i]['ancestors']=set()
           
          # Process each level set based on the expansion mode
          while instrumental_set:
                instrumental_set = set()
                for deme in boundary:
                    if deme%self.n_columns!=0:
                       if deme-1 not in existing_set:
                          instrumental_set.add(deme-1)
                          self.digraph.nodes[deme-1]['ancestors'].add(deme)
                       if (deme+1)%self.n_columns!=0:
                           if deme+1 not in existing_set:
                              instrumental_set.add(deme+1)
                              self.digraph.nodes[deme+1]['ancestors'].add(deme)
                                 
                # Update the sets for the next iteration               
                existing_set = existing_set.union(instrumental_set)
                boundary = instrumental_set.copy()
                if boundary:
                   Level_sets.append(boundary)
                   for i in boundary:
                       self.digraph.nodes[i]['level'] = len(Level_sets)-1     # Set the level for each node in the new boundary
        
          self.digraph.graph['Level_sets']=Level_sets
          self.digraph.graph['max_level']=len(Level_sets)-1
           
          return()
      
      def set_up_populations(self,
                             n_e_mode=None
                             ):  
           
          """
          Setup the populations in msprime.

          Parameters:
             n_e_mode (str): 'equal' or 'proportional'. Defaults to 'equal'. This determines whether
             the effective population sizes are equal or proportional to the stationary
             distribution derived from the Markov chain transition matrix.

           Notes:
            - This function initializes demographic settings for a simulation, setting up populations
              and migration rates based on the graph structure.
            - Population sizes can be either uniform ('equal') or adjusted based on the stationary
              distribution of a Markov chain derived from the adjacency matrix of the graph.
              
          """
           
          d = len(self.digraph.nodes) 
          self.demography = msprime.Demography() 
           
          # Default: equal effective population sizes
          n_e=np.ones(d).tolist()
           
          # Proportional sizes based on stationary distribution
          if n_e_mode=="proportional":
             M=nx.adjacency_matrix(self.digraph, weight="weight").toarray()
             for i in range(d):
                 M[i,i]=1-np.sum(M[i,:])
                      
             mc=markovChain(M)  
             mc.computePi('linear')
             pi=mc.pi.reshape(d)                                               #Get the stationary distribution   
             pi=pi/np.sum(pi)
             n_e=((pi/np.max(pi))).tolist()                                     
            
          # Add populations to demography, setting initial sizes
          for i in range(d):
              self.demography.add_population(initial_size=n_e[i],
                                             initially_active=True)
               
          # Set migration rates between populations based on graph edges
          for x in range(d):
              for y in list(self.digraph.neighbors(x)):
                  self.demography.set_migration_rate(f"pop_{x}", f"pop_{y}", self.digraph[x][y]['weight'])
                   
          self.n_e=n_e
                   
          return()        
           
      def set_up_re(self,
                    re_origin,
                    re_dt,
                    re_proportion,
                    ):
            
          """
          Setup the simulation parameters for a range expansion model.

          Parameters:
             re_origin (list): The origin points for range expansion, specified as (column, row) indices.
             re_dt (float): The time interval between successive expansion events.
             re_proportion (float): The proportion of the population that moves to a new deme during expansion.

          Notes:
            - This function adjusts population parameters, sets migration rates to zero between demes, and handles
              population splits and admixture events based on the calculated level sets from `compute_level_set`. 
               
          """
            
          # Calculate the level sets based on the provided origin and mode
          
          self.compute_level_set(origin=re_origin)
            
          # Retrieve the maximum level from the graph properties
          lmax=self.digraph.graph['max_level']
          dt=re_dt                                                             #Time interval of range expansion
          k=np.log(1/re_proportion)/dt                                         #Growth rate of boundary population
             
          # Process each level set starting from the highest level
          for i in range(lmax):
              boundary=self.digraph.graph['Level_sets'][lmax-i].copy()
              for j in boundary:
                    
                  # Update population parameters for each node in the boundary
                  self.demography.add_population_parameters_change(time=(4*i+3)*dt, 
                                                                   population=f"pop_{j}", 
                                                                   initial_size=self.n_e[j],
                                                                   growth_rate=k)
                                   
                  # Retrieve the list of ancestors for each node in the current boundary
                  ancestors=list(self.digraph.nodes[j]['ancestors'])
                  ancestor_names=[f"pop_{num}" for num in self.digraph.nodes[j]['ancestors']]
                    
                  # Handle population splits for single ancestors or admixture for multiple ancestors
                  if len(ancestors)==1:
                     self.demography.add_population_split(time=4*(i+1)*dt, 
                                                          derived=[f"pop_{j}"], 
                                                          ancestral=ancestor_names[0])                
                  elif len(ancestors)>1:
                       proportions=np.ones(len(ancestors))/len(ancestors)
                       proportions=proportions.tolist()
                       self.demography.add_admixture(time=4*(i+1)*dt, 
                                                     derived=f"pop_{j}", 
                                                     ancestral=ancestor_names,
                                                     proportions=proportions)
                   
                  # Set migration rates to zero for the nodes after(backward in time) population split and admixture 
                  for k in list(self.digraph.neighbors(j)):
                      self.demography.set_migration_rate(f"pop_{j}", f"pop_{k}", 0)
                      self.demography.set_migration_rate(f"pop_{k}", f"pop_{j}", 0)
          return()
             
      def set_up_mm(self,
                    mm_origin,
                    mm_dt,
                    mm_proportion,
                    ):
          
          """
          Setup the simulation parameters for a mass migration model.

          Parameters:
             mm_origin (list): The origin points for mass migration, specified as (column, row) indices.
             mm_dt (float): The time interval between successive mass migration events.
             mm_proportion (float): The proportion of the population that migrates out of each deme during each event.

          Notes:
           - This function configures mass migration events based on level sets calculated from the origin points.
             It schedules migrations proportionally distributed among ancestors at defined intervals.
          """
           
          self.compute_level_set(mm_origin)
           
          lmax=self.digraph.graph['max_level']
          dt=mm_dt                                                            #Time interval of mass expansion
               
          for i in range(lmax):
              boundary=self.digraph.graph['Level_sets'][lmax-i].copy()
              for j in boundary:
                  ancestors=list(self.digraph.nodes[j]['ancestors'])
                  ancestor_names=[f"pop_{num}" for num in self.digraph.nodes[j]['ancestors']]
                   
                  # Schedule a mass migration event for each ancestor
                  for k in range(len(ancestors)):
                      proportion=mm_proportion/len(ancestors)
                      self.demography.add_mass_migration(time=4*(i+1)*dt, 
                                                         source=f"pop_{j}", 
                                                         dest=ancestor_names[k],
                                                         proportion=proportion)                    
          return()
          
          
      def simulate_genotypes(self,
                             sequence_length,
                             mu,
                             target_n_snps,
                             n_print):
          
         """
         Simulates a genotype matrix based on the specified demographic model and mutation rate.

         Parameters:
           sequence_length (float): Total length of the genome sequence to be simulated.
           mu (float): Mutation rate per base per generation.
           target_n_snps (int): Number of independent replicates to simulate, each aiming to generate at least one SNP.
           n_print (int): Frequency of progress updates during the simulation.

        Returns:
          genotypes (2D ndarray): The simulated genotype matrix with individuals as rows and SNPs as columns.
           
        """
 
         d = len(self.digraph.nodes)

         self. demography.sort_events()

         # sample sizes per node
         sample_sizes = list(nx.get_node_attributes(
               self.digraph, "sample_size").values())

         samples = {f"pop_{i}": sample_sizes[i] for i in range(d)}

         # tree sequences
         ts = msprime.sim_ancestry(samples=samples,
                                   demography=self.demography,
                                   sequence_length=sequence_length,
                                   num_replicates=target_n_snps)

         # simulate haplotypes
         haplotypes = []

         for i, tree_sequence in enumerate(ts):
             
             mutated_tree_sequence = msprime.sim_mutations(tree_sequence,
                                                           rate=mu,
                                                           model=msprime.BinaryMutationModel(),
                                                           discrete_genome=False)
             # extract haps from ts
             H = mutated_tree_sequence.genotype_matrix()
             p, n = H.shape

             # select a random marker per linked replicate
             if p == 0:
                 continue
             else:
                 idx = np.random.choice(np.arange(p), 1)
                 h = H[idx, :]
                 haplotypes.append(h)

             if i % n_print == 0:
                 print("Simulating ~SNP {}".format(i))

         # stack haplotypes over replicates
         H = np.vstack(haplotypes)

         # convert to genotype matrix: s/o to @aabiddanda
         genotypes = H[:, ::2] + H[:, 1::2]

         return genotypes.T
