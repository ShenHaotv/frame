

# frame

**F**ine **R**esolution **A**symmetric **M**igration **E**stimation (`frame`) is a python package 
implementing a statistical method for inferring and visualizing asymmetric gene-flow in 
spatial population genetic data.

The `frame` method and software was developed by Hao Shen and advised by John Novembre. The structure of project was adapted from feems https://github.com/NovembreLab/feems.git. We also used code from Benjamin M. Peter to help construct the spatial graphs. 
 
To get started, setup a `conda` environment:

```
conda create -n=frame-e python=3.11.9
conda activate frame-e
conda config --add channels conda-forge
```
Note: For Mac M1 users they'll need to add `--platform=osx-arm64` to the conda create command to make sure that all the packages are found correctly in the channel-forge channel of conda. 

The dependencies are listed in dependencies.txt - though we recommend installing packages using `conda` and pip in the following sequence to avoid conflicts in packages:

```
conda install numpy==1.26.4 scipy==1.11.4 scikit-learn==1.5.1
conda install setuptools==71.0.4 pytest==8.3.2
pip install discreteMarkovChain
conda install pandas-plink=2.2.9 matplotlib==3.9.1 click==8.1.7
conda install fiona==1.9.6
conda install shapely==2.0.5
conda install pyproj==3.6.1
conda install cartopy=0.23.0
conda install networkx=3.3
conda install msprime==1.3.2

```
Once the conda environment has been setup with these dependencies we can install `frame`:

```

pip install git+https://github.com/shenhaotv/frame

```


# Running frame

To help get your analysis started, we provide an example workflow in the [Example.ipynb](https://github.com/ShenHaotv/frame/blob/main/docsrc/Example.ipynb) notebook. The notebook analyzes empirical data from North American gray wolves populations published in [Schweizer et al. 2015](https://onlinelibrary.wiley.com/doi/full/10.1111/mec.13364?casa_token=idW0quVPOU0AAAAA:o_ll85b8rDbnW3GtgVeeBUB4oDepm9hQW3Y445HI84LC5itXsiH9dGO-QYGPMsuz0b_7eNkRp8Mf6tlW). 
