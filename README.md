# fitScalingRelation

This is a code for fitting galaxy cluster scaling relations using orthogonal or bisector regression and MCMC. It takes
into account errors on both variables and intrinsic scatter. The algorithm is described in [Hilton et al. 
(2012; MNRAS, 424, 2086)](http://adsabs.harvard.edu/abs/2012MNRAS.424.2086H), where it was used to fit the 
luminosity--temperature scaling relation for the XMM Cluster Survey (XCS) first data release. Note that it is
simple to add Y|X and X|Y regression models as additional options, but these have not been implemented at present.

Although the code is very much geared up for fitting galaxy cluster scaling relations of all kinds, it can
be used for any kind of regression problem with errors on both variables and intrinsic scatter.

## Software needed

The code is written in python (2.7.x). It needs the following packages to be installed:
    
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

You can run the code using

```fitScalingRelation parFileName```

where `parFileName` is the name of a file containing all necessary parameters (e.g., location of the dataset,
priors, etc.).

You can find example `.par` files for running the code on the original XCS DR1 catalogue (warts and all) in
the `examples/` directory.

## Caveats

If you're using this code to fit cluster scaling relations, taking into account selection effects and the 
mass function is important. This algorithm doesn't do that currently.

## Acknowledgement

If you use this code, please cite [Hilton et al. (2012; MNRAS, 424, 2086)](http://adsabs.harvard.edu/abs/2012MNRAS.424.2086H). 

## Comments, bug reports, help, suggestions etc..

Please contact Matt Hilton <matt.hilton@mykolab.com>.
