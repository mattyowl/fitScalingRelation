# Cython routines for LT fitting
#
# These are the routines for doing MCMC rather than a brute force grid approach

from astLib import *
cimport numpy as np
import numpy as np
cimport cython
import math
import time
import sys

DTYPE=np.float64
ctypedef np.float64_t DTYPE_t

#-------------------------------------------------------------------------------------------------------------
# Constants

cdef DTYPE_t NSAMP=4
cdef DTYPE_t NSIG=4

#-------------------------------------------------------------------------------------------------------------
@cython.boundscheck(False) # turn off bounds-checking for entire function
def fastOrthogonalLikelihood(np.ndarray[DTYPE_t, ndim=1] pars,
                             np.ndarray[DTYPE_t, ndim=1] log10L, np.ndarray[DTYPE_t, ndim=1] log10LErrPlus, np.ndarray[DTYPE_t, ndim=1] log10LErrMinus,
                             np.ndarray[DTYPE_t, ndim=1] log10T, np.ndarray[DTYPE_t, ndim=1] log10TErrPlus, np.ndarray[DTYPE_t, ndim=1] log10TErrMinus,
                             np.ndarray[DTYPE_t, ndim=1] log10RedshiftEvo, np.ndarray[DTYPE_t, ndim=1] detP):
    """This is now defined orthogonal to the best fit line (see notes for trig).
    NOTE: Now zpx is assumed applied before, i.e. feed in log10T-zpx, not log10T
    
    """
    
    cdef DTYPE_t A, B, C, S
    A=pars[0]
    B=pars[1]
    C=pars[2]
    S=pars[3]
    
    cdef DTYPE_t prob, ydiff, yfit, xdiff, xfit, xerr, yerr, theta, orthDistance, phi, orthError, orthSigTotalSq, sqrt_orthSigTotalSq 
    cdef int i, k
    cdef DTYPE_t probs, xi, yi, x0, y0, xd, yd

    cdef DTYPE_t root2=1.4142135623730951
    cdef DTYPE_t root2pi=2.5066282746310002
    
    cdef DTYPE_t err
 
    probs=0.0
    probsArray=np.zeros(log10L.shape[0]) # for debugging
    for k in range(log10L.shape[0]):

        # This is the same as our trig - maybe it might be quicker, but we need theta anyway
        #y0=log10L[k]
        #x0=log10T[k]
        #xi=(x0+B*y0-B*A)/(B*B+1.0)
        #yi=B*xi+A
        #xd=x0-xi
        #yd=y0-yi
        #orthDistance=math.sqrt(xd*xd+yd*yd)

        # Our trig - we still need xdiff, ydiff for the asymmetric errors dance
        yfit=A+B*log10T[k]+C*log10RedshiftEvo[k]
        ydiff=log10L[k]-yfit
        xfit=(log10L[k]-A-C*log10RedshiftEvo[k])/B
        xdiff=log10T[k]-xfit
        theta=math.atan2(ydiff, xdiff)
        orthDistance=math.sin(theta)*xdiff # Or: orthDistance=math.cos(theta)*ydiff

        if xdiff < 0:
            xerr=log10TErrPlus[k]
        else:
            xerr=log10TErrMinus[k]
        if ydiff < 0:
            yerr=log10LErrPlus[k]
        else:
            yerr=log10LErrMinus[k]

        # As in the paper - ellipse projection
        #phi=np.pi-theta  # was 180 - pi/2.0 flips the offset to the other axis
        #orthError=(xerr*yerr) / math.sqrt((xerr*math.sin(phi))**2 + (yerr*math.cos(phi))**2)
        #print phi, theta, xerr, yerr, B, orthError

        # ^^^ Not sure why (xerr*yerr) above (lost in mists of time)... below added 02/10/15
        phi=np.pi/2.0 - theta
        orthError=math.sqrt(xerr*xerr*math.cos(phi)**2 + yerr*yerr*math.sin(phi)**2)

        orthSigTotalSq=orthError*orthError + S*S
        sqrt_orthSigTotalSq=math.sqrt(orthSigTotalSq)

        prob = (1./(root2pi*sqrt_orthSigTotalSq)) * math.exp(-(orthDistance*orthDistance)/(2.0*orthSigTotalSq))

        # apply selection function probability
        prob=prob*detP[k]
    
        # for debugging
        probsArray[k]=prob

        #print k, prob, orthError, S, xerr, yerr, log10T[k], log10L[k]

        # Running product of all probabilities
        # NOTE: Gone log likelihood here because we get overflows for samples of 1000s of sim clusters
        # This means when we decide if we want to keep samples, subtract prob ratios and use alpha > 0
        # (see the python side code in MCMCFit)
        #probs=probs*prob
        #print prob
        if prob > 0:
            probs=probs+math.log10(prob)
    
    #print probs

    return probs, probsArray

#-------------------------------------------------------------------------------------------------------------
@cython.boundscheck(False) # turn off bounds-checking for entire function
def fastBisectorLikelihood(np.ndarray[DTYPE_t, ndim=1] pars, 
                           np.ndarray[DTYPE_t, ndim=1] log10L, np.ndarray[DTYPE_t, ndim=1] log10LErrPlus, np.ndarray[DTYPE_t, ndim=1] log10LErrMinus,
                           np.ndarray[DTYPE_t, ndim=1] log10T, np.ndarray[DTYPE_t, ndim=1] log10TErrPlus, np.ndarray[DTYPE_t, ndim=1] log10TErrMinus,
                           np.ndarray[DTYPE_t, ndim=1] log10RedshiftEvo, np.ndarray[DTYPE_t, ndim=1] detP):
    """This is now defined using a bisector method.
    NOTE: Now zpx is assumed applied before, i.e. feed in log10T-zpx, not log10T

    
    """
    
    cdef DTYPE_t A, B, C, Sx, Sy
    A=pars[0]
    B=pars[1]
    C=pars[2]
    Sx=pars[3]
    Sy=pars[4]
           
    cdef DTYPE_t prob, prob_x, prob_y, ydiff, yfit, xdiff, xfit, xErrTotalSq, sqrt_xErrTotalSq, yErrTotalSq, sqrt_yErrTotalSq
    cdef int i, k
    cdef DTYPE_t probs

    cdef DTYPE_t root2=1.4142135623730951
    cdef DTYPE_t root2pi=2.5066282746310002
    
    cdef DTYPE_t err
 
    probs=0.0
    probsArray=np.zeros(log10L.shape[0]) # for debugging
    for k in range(log10L.shape[0]):

        yfit=A+B*log10T[k]+C*log10RedshiftEvo[k]
        ydiff=log10L[k]-yfit

        xfit=(log10L[k]-A-C*log10RedshiftEvo[k])/B
        xdiff=log10T[k]-xfit

        if xdiff < 0:
            xerr=log10TErrPlus[k]
        else:
            xerr=log10TErrMinus[k]
        if ydiff < 0:
            yerr=log10LErrPlus[k]
        else:
            yerr=log10LErrMinus[k]

        xErrTotalSq=xerr*xerr + Sx*Sx
        yErrTotalSq=yerr*yerr + Sy*Sy
  
        sqrt_xErrTotalSq=math.sqrt(xErrTotalSq)
        sqrt_yErrTotalSq=math.sqrt(yErrTotalSq)

        prob_x = (1./(root2pi*sqrt_xErrTotalSq)) * math.exp(-(xdiff*xdiff)/(2.0*xErrTotalSq))
        prob_y = (1./(root2pi*sqrt_yErrTotalSq)) * math.exp(-(ydiff*ydiff)/(2.0*yErrTotalSq))

        # apply selection function probability
        prob=(prob_x * prob_y) * detP[k]
    
        # for debugging
        probsArray[k]=prob

        #print k, prob, orthError, S, xerr, yerr, log10T[k], log10L[k]

        # Running product of all probabilities
        # NOTE: Gone log likelihood here because we get overflows for samples of 1000s of sim clusters
        # This means when we decide if we want to keep samples, subtract prob ratios and use alpha > 0
        # (see the python side code in MCMCFit
        if prob > 0:
            probs=probs+math.log10(prob)
    
    #print probs

    return probs, probsArray

#-------------------------------------------------------------------------------------------------------------
@cython.boundscheck(False) # turn off bounds-checking for entire function
def fast2DProbProjection(np.ndarray[DTYPE_t, ndim=1] par1Values, np.ndarray[DTYPE_t, ndim=1] par2Values,
                         int par1Axis, int par2Axis, np.ndarray[DTYPE_t, ndim=2] pars):
    """Makes 2d projected probability distribution for two parameters. 

    NOTE: Although this is called the same, it's different to fast2DProbProjection in the pre-MCMC code.


    par1Values  = values which define x axis of 2d grid, must be regularly spaced
    par2Values  = values which define y axis of 2d grid, must be regularly spaced
    par1Axis    = axis in pars that corresponds to parameter 1
    par2Axis    = axis in pars that corresponds to parameter 2
    pars        = 2d array containing all the parameter values fed into the likelihood calculation
    
    Returns:    [probability image, par1Values, par2Values]
    """
    
    
    cdef float norm
    cdef int i1, i2, k
    cdef np.ndarray[DTYPE_t, ndim=2] PDist

    PDist=np.zeros([par1Values.shape[0], par2Values.shape[0]], dtype = DTYPE)

    for i1 in range(par1Values.shape[0]-1):
        for i2 in range(par2Values.shape[0]-1):
            for k in range(pars.shape[0]):
                if pars[k, par1Axis] > par1Values[i1] and pars[k, par1Axis] <= par1Values[i1+1]:
                    if pars[k, par2Axis] > par2Values[i2] and pars[k, par2Axis] <= par2Values[i2+1]:
                        PDist[i1, i2]=PDist[i1, i2]+1
    
    norm=float(pars.shape[0])
    PDist=PDist/norm

    return PDist

#-------------------------------------------------------------------------------------------------------------
@cython.boundscheck(False) # turn off bounds-checking for entire function
def fast1DProbProjection(np.ndarray[DTYPE_t, ndim=1] par1Values, int par1Axis, 
                         np.ndarray[DTYPE_t, ndim=2] pars):
    """Makes 1d projected probability distribution for one parameter.

    NOTE: Different from code in non-MCMC code.

    par1Values  = values which define axis on which we project, must be regularly spaced
    par1Axis    = axis in pars that corresponds to parameter 1
    pars        = 2d array containing all the parameter values fed into the likelihood calculation
    
    """
    
    
    cdef float norm
    cdef int i1, k
    cdef np.ndarray[DTYPE_t, ndim=1] PDist

    PDist=np.zeros(par1Values.shape[0], dtype = DTYPE)

    for i1 in range(par1Values.shape[0]-1):
        for k in range(pars.shape[0]):
            if pars[k, par1Axis] > par1Values[i1] and pars[k, par1Axis] <= par1Values[i1+1]:
                PDist[i1]=PDist[i1]+1
    
    norm=float(pars.shape[0])
    PDist=PDist/norm

    return PDist
