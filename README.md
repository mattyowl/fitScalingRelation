# fitScalingRelation

This is a code for fitting galaxy cluster scaling relations using orthogonal or bisector regression and MCMC. It takes
into account errors on both variables and intrinsic scatter. The algorithm is described in [Hilton et al. 
(2012; MNRAS, 424, 2086)](http://adsabs.harvard.edu/abs/2012MNRAS.424.2086H), where it was used to fit the 
luminosity--temperature scaling relation for the XMM Cluster Survey (XCS) first data release.

As of May 2016, a version of the [Kelly (2007; ApJ, 665, 1489)](http://adsabs.harvard.edu/abs/2007ApJ...665.1489K)
regression algorithm has been added. This uses the [python implementation by Joshua Meyers](https://github.com/jmeyers314/linmix),
which is being adapted in this package to additionally fit for redshift evolution. This is not thoroughly
road tested yet, and certainly contains bugs at the moment. Don't use it (yet). However, if you don't require
a fit for redshift evolution, set CFit = 0 and C0 = 0 in the ,par file and the code will work as expected.
In this case, it is just acting as a wrapper for the linmix python code.

Although the code is very much geared up for fitting galaxy cluster scaling relations of all kinds, it can
be used for any kind of regression problem with errors on both variables and intrinsic scatter.

## Software needed

The code is written in Python (ported to Python3 in March 2022). 

It needs the following packages to be installed:
    
* numpy
* scipy
* atpy (this could probably be replaced with astropy, but not yet)
* astLib
* matplotlib
* cython
* IPython (used for debugging)

## Installation

To build and install, do the usual:
    
```sudo python setup.py install```

## Running the code

To run either the bisector or orthogonal fitting algorthms, you can run the code using

```fitScalingRelation parFileName```

where `parFileName` is the name of a file containing all necessary parameters (e.g., location of the dataset,
priors, etc.).

You can find example `.par` files for running the code on the original XCS DR1 catalogue (warts and all) in
the `examples/` directory.

To run the Kelly (2007) algorithm (with the addition of redshift evolution switched on by default at the 
moment), use

```fitLinMixScalingRelation parFileName```

where the parameters file can be identical to the one used for ```fitScalingRelation``` - parameters
which don't apply for the Kelly algorithm are ignored.

## Caveats

If you're using this code to fit cluster scaling relations, taking into account selection effects and the 
mass function is important. This algorithm doesn't do that currently.

## Acknowledgement

If you use the orthogonal/bisector fitting code, please cite [Hilton et al. (2012; MNRAS, 424, 2086)](http://adsabs.harvard.edu/abs/2012MNRAS.424.2086H). 
If you use fitLinMixScalingRelation, cite [Kelly (2007; ApJ, 665, 1489)](http://adsabs.harvard.edu/abs/2007ApJ...665.1489K).

## Comments, bug reports, help, suggestions etc..

Please contact Matt Hilton <matt.hilton@mykolab.com>.
