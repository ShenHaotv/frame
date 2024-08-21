
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/NovembreLab/feems/main)

# feems

**F**ast **E**stimation of **E**ffective **M**igration **S**urfaces (`feems`) is a python package 
implementing a statistical method for inferring and visualizing gene-flow in 
spatial population genetic data.

The `feems` method and software was developed by Joe Marcus and Wooseok Ha and 
advised by Rina Foygel Barber and John Novembre. We also used code from Benjamin M. Peter 
to help construct the spatial graphs. 

For details on the method see our [pre-print](https://www.biorxiv.org/content/10.1101/2020.08.07.242214v1). Note that `feems` is in review so the method could be subject to change.  

Note: MS Windows users will struggle to install feems directly in a 
Windows environment because at least one of the dependencies does not
have a Windows port.  A virtual Linux machine should be preferable if 
you are on a Windows machine. 
 
 
 
To get started, setup a `conda` environment:

```
conda create -n=FRAME python=3.11.9
conda activate 
```
Note: For Mac M1 users they'll need to add `--platform=osx-64` to the conda create command to make sure that all the packages are found correctly in the channel-forge channel of conda. 

The dependencies are listed in dependencies.txt - though we recommend installing packages using `conda` and pip in the following sequence to avoid conflicts in packages:

```
conda install numpy==1.26.4 scipy==1.11.4 scikit-learn==1.5.1
conda install setuptools==71.0.4 pytest==8.3.2
pip install discreteMarkovChain
conda install pandas-plink==2.2.9 matplotlib==3.9.1 click==8.1.7
conda install fiona==1.9.6
conda install shapely==2.0.5
conda install pyproj==3.6.1
conda install cartopy=0.23.0
conda install networkx=3.3
conda install msprime==1.3.2

```
