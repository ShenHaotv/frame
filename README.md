

# frame
(This method is still currently under review, so there may be some changes along the way. Please make sure to always pull the latest version).


**F**ine **R**esolution **A**symmetric **M**igration **E**stimation (`frame`) is a python package 
implementing a statistical method for inferring and visualizing asymmetric gene-flow in 
spatial population genetic data.

The `frame` method and software was developed by Hao Shen and advised by John Novembre. The structure of project was adapted from feems https://github.com/NovembreLab/feems.git. We also used code from Benjamin M. Peter to help construct the spatial graphs. 
 
We recommend that users install [Anaconda][anaconda] or [Miniconda][miniconda] to prioritize the MKL-optimized versions of numerical libraries such as ```numpy```, ```scipy```, and others. This is crucial for ensuring the numerical performance of our method. 
To get started, set up a new conda environment, **make sure that ```defaults``` is the only channel**:

```
conda create -n frame-e python=3.11.9
conda activate frame-e
```

The dependencies are listed in dependencies.txt. We recommend installing packages using `conda` and 'pip' in the following sequence to avoid conflicts in packages:

```
conda install numpy=1.26.4 scipy=1.11.4 scikit-learn=1.5.1 "blas=*=mkl" mkl-service threadpoolctl
pip install "mkl==2025.0.1" "intel-openmp==2025.0.3"
conda install pytest==8.3.4 pyproj==3.6.1 matplotlib==3.10
conda install click==8.1.8 fiona==1.10.1 cartopy==0.24.1
conda install networkx==3.4.2 setuptools==75.8.0 shapely==2.0.6
pip install pandas-plink==2.3.1 msprime==1.3.3 discreteMarkovChain
```

**Note: For Mac users, please use pip install "mkl==2023.2.2" "intel-openmp==2023.2.0" unless higher version for Mac is released** 

Once the conda environment has been setup with these dependencies we can install `frame`:

```
pip install --no-deps git+https://github.com/shenhaotv/frame
```

# Running frame

To help get your analysis started, we provide an example workflow in the [Example.ipynb](https://github.com/ShenHaotv/frame/blob/main/docsrc/Example.ipynb) notebook, along with guidance on advanced usage in the [Extension.ipynb](https://github.com/ShenHaotv/frame/blob/main/docsrc/Extension.ipynb) notebook. These notebooks analyze empirical data from North American gray wolves populations published in [Schweizer et al. 2015](https://onlinelibrary.wiley.com/doi/full/10.1111/mec.13364?casa_token=idW0quVPOU0AAAAA:o_ll85b8rDbnW3GtgVeeBUB4oDepm9hQW3Y445HI84LC5itXsiH9dGO-QYGPMsuz0b_7eNkRp8Mf6tlW). 

[anaconda]: https://www.anaconda.com/products/distribution
[miniconda]: https://docs.conda.io
