

# frame

**F**ine **R**esolution **A**symmetric **M**igration **E**stimation (`frame`) is a python package 
implementing a statistical method for inferring and visualizing asymmetric gene-flow in 
spatial population genetic data.

The `frame` method and software was developed by Hao Shen and advised by John Novembre. The structure of project was adapted from feems https://github.com/NovembreLab/feems.git. We also used code from Benjamin M. Peter to help construct the spatial graphs. 
 
We recommend users install [Anaconda][anaconda] as it prioritizes the MKL-optimized versions of numerical libraries such as ```numpy```, ```scipy```, and others. This is crucial for ensuring the numerical performance of our method.
To get started,set up a new conda environment, make sure that defaults is the only channel to avoid conflicts:

```
conda create -n=frame-e python=3.11.9
conda activate frame-e
```
Note: For Mac M1 users they'll need to add `--platform=osx-arm64` to the conda create command to make sure that all the packages are found correctly in the channel-forge channel of conda. 

The dependencies are listed in dependencies.txt - though we recommend installing packages using `conda` and pip in the following sequence to avoid conflicts in packages:

```
conda install numpy==1.26.4 scipy==1.11.4 scikit-learn==1.5.1
conda install setuptools pytest
conda install matplotlib click fiona
conda install shapely==2.0.5 pyproj==3.6.1
conda install cartopy networkx
```
Once the conda environment has been setup with these dependencies we can install `frame`:

```

pip install git+https://github.com/shenhaotv/frame

```


# Running frame

To help get your analysis started, we provide an example workflow in the [Example.ipynb](https://github.com/ShenHaotv/frame/blob/main/docsrc/Example.ipynb) notebook. The notebook analyzes empirical data from North American gray wolves populations published in [Schweizer et al. 2015](https://onlinelibrary.wiley.com/doi/full/10.1111/mec.13364?casa_token=idW0quVPOU0AAAAA:o_ll85b8rDbnW3GtgVeeBUB4oDepm9hQW3Y445HI84LC5itXsiH9dGO-QYGPMsuz0b_7eNkRp8Mf6tlW). 

[anaconda]: https://www.anaconda.com/products/distribution
