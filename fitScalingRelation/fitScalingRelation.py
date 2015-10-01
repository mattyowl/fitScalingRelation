"""

    The MCMC fitting code used in Hilton et al. (2012), in a more general purpose form

    Copyright 2015 Matt Hilton (matt.hilton@mykolab.com)
    
    This file is part of fitScalingRelation.

    fitScalingRelation is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    fitScalingRelation is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with fitScalingRelation.  If not, see <http://www.gnu.org/licenses/>.

"""

import os
import sys
import math
import string
from astLib import *
import pylab as plt
import numpy as np
import atpy
import popen2
from scipy import stats
from scipy import special
from scipy import interpolate
from scipy import ndimage
import pyximport; pyximport.install()
import cythonScalingRelation as csr
import time
import pickle
import matplotlib
import IPython
np.random.seed()
plt.matplotlib.interactive(False)

#-------------------------------------------------------------------------------------------------------------
# Adopt Ed's cosmology
#astCalc.OMEGA_M0=0.27
#astCalc.OMEGA_L=0.73
    
#-------------------------------------------------------------------------------------------------------------
def ask_for( key ):
    s = raw_input( "ParametersDict: enter value for '%s': " % key )
    try:
        val = eval(s)
    except NameError:
        # allow people to enter unquoted strings
        val = s
    return val

class ParametersDict( dict ):

    def __getitem__( self, key ):
        if key not in self:
            print "ParametersDict: parameter '%s' not found" % key
            val = ask_for( key )
            print "ParametersDict: setting '%s' = %s" % (key,repr(val))
            dict.__setitem__( self, key, val )
        return dict.__getitem__( self, key )

    def read_from_file( self, filename ):
        f = open( filename )
        old = ''
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            s = line.split('#')
            line = s[0]
            #if line[-1] == '\\':
                #s = line.split('\\')
                #if len(s) > 1:
                    #old = string.join([old, s[0]])
                    #continue
                #else:
                    #line = string.join([old, s[0]])
                    #old = ''
                ##IPython.embed()
                ##sys.exit()
            s = line.split('=')
            if len(s) != 2:
                print "Error parsing line:"
                print line
                IPython.embed()
                sys.exit()
                continue
            try:
                key = s[0].strip()
                val = eval(s[1].strip()) # XXX:make safer
            except:
                raise Exception, "can't parse line: %s" % (line)
            self[key] = val
        f.close()

    def write_to_file( self, filename, mode = 'w' ):
        f = open( filename, mode )
        keys = self.keys()
        keys.sort()
        for key in keys:
            f.write( "%s = %s\n" % (key,repr(self[key])) )
        f.close()

    def cmp( self, otherDict ):
        
        diff = []
        ks = self.keys()
        for k in ks:
            try:
                if otherDict[k] == self.params[k]:
                    continue
                diff += [k]
                break
            except KeyError:
                diff += [k]
        return otherDict

#-------------------------------------------------------------------------------------------------------------
def selectStartParsFromPriors(settingsDict):
    """Choose random starting values for the MCMC from the priors we're placing on the parameters.
    
    """
    
    variables=settingsDict['variables']
    
    pars=np.zeros(len(variables))
    for i in range(len(variables)):
        v=variables[i]
        if settingsDict['%sFit' % (v)] == 'fixed':
            pars[i]=settingsDict['%s0' % (v)] 
        else:
            pars[i]=np.random.uniform(settingsDict['prior_%s_MIN' % (v)], settingsDict['prior_%s_MAX' % (v)])

    # This makes sure that if we're testing by swapping axes, we can use the same prior ranges
    if 'swapAxes' in settingsDict.keys() and settingsDict['swapAxes'] == True:
        b=1.0/pars[1]
        a=-pars[0]/pars[1]
        pars[0]=a
        pars[1]=b
    
    return pars

#-------------------------------------------------------------------------------------------------------------
def getPPrior(pPars, settingsDict):
    """Gets prior probability.
    
    """
    
    variables=settingsDict['variables']

    # This makes sure that if we're testing by swapping axes, we can use the same prior ranges
    if 'swapAxes' in settingsDict.keys() and settingsDict['swapAxes'] == True:
        b=1.0/pPars[1]
        a=-pPars[0]/pPars[1]
        pPars[0]=a
        pPars[1]=b

    priors=np.zeros(len(variables))
    for i in range(len(variables)):
        v=variables[i]
        if pPars[i] > settingsDict['prior_%s_MIN' % (v)] and pPars[i] < settingsDict['prior_%s_MAX' % (v)]:
            priors[i]=1.0
        else:
            priors[i]=0.0
        # Fixed parameters must surely be within the priors...
        if settingsDict['%sFit' % (v)] == 'fixed':
            priors[i]=1.0
        
    pPrior=np.product(priors)
        
    return pPrior

#-------------------------------------------------------------------------------------------------------------
def byteSwapArr(arr):
    """FITS is big-endian, but cython likes native-endian arrays (little-endian for x86)... so, byteswap
    if we need.
    
    """
    
    if arr.dtype.byteorder == '>':
        arr=arr.byteswap().newbyteorder('=')
    
    return arr
        
#-------------------------------------------------------------------------------------------------------------
def sampleGetter(settingsDict, sampleDef, outDir):
    """Loads in catalogue in .fits table format, and add columns xToFit, yToFit, xErrToFit, yErrToFit,
    which are fed into the MCMCFit routine. Applies any asked for scalings and cuts according to the 
    contents of settingsDict and sampleDef.
    
    """
    
    # Stuff we need from settings...
    xColumnName=settingsDict['xColumnName']
    xPlusErrColumnName=settingsDict['xPlusErrColumnName']
    xMinusErrColumnName=settingsDict['xMinusErrColumnName']
    yColumnName=settingsDict['yColumnName']
    yPlusErrColumnName=settingsDict['yPlusErrColumnName']
    yMinusErrColumnName=settingsDict['yMinusErrColumnName']
    
    xPivot=settingsDict['xPivot']
    yPivot=settingsDict['yPivot']
    
    xTakeLog10=settingsDict['xTakeLog10']
    yTakeLog10=settingsDict['yTakeLog10']
    
    redshiftColumnName=settingsDict['redshiftColumnName']
    xScaleFactor=settingsDict['xScaleFactor']
    yScaleFactor=settingsDict['yScaleFactor']
    yScaleFactorPower=settingsDict['yScaleFactorPower']

    newTab=atpy.Table(settingsDict['inFileName'])
    
    # Make a new table here with cuts applied
    # NOTE: we really need a better way of labelling constraints
    for key in sampleDef:
        if key not in ['label', 'plotLabel']:
            if key[-4:] == '_MIN':
                col=key[:-4]
                newTab=newTab.where(newTab[col] > sampleDef[key])
            elif key[-4:] == '_MAX':
                col=key[:-4]
                newTab=newTab.where(newTab[col] < sampleDef[key])
            else:
                if type(sampleDef[key]) != list:
                    newTab=newTab.where(newTab[key] == sampleDef[key])
                else:
                    print "Need to add more sampleDef key handling code"
                    IPython.embed()
                    sys.exit()
    if len(newTab) == 0:
        print "Hmm... all objects cut? empty newTab"
        IPython.embed()
        sys.exit()
        
    # Value added useful columns
    Ez=[]
    for row in newTab:
        Ez.append(astCalc.Ez(row[redshiftColumnName]))
    newTab.add_column('E(z)', Ez)
    
    # Add columns we will fit to, scaling and applying log10 as necessary
    # We apply pivots here also (undo them, if necessary, elsewhere)
    stab=newTab

    # We should probably make this default
    if xPivot == "median":
        xPivot=np.median(newTab[xColumnName])
        settingsDict['xPivot']=xPivot
    if yPivot == "median":
        yPivot=np.median(newTab[yColumnName])
        settingsDict['yPivot']=yPivot
        
    if yScaleFactor == "E(z)":
        yScaling=np.power(stab["E(z)"], yScaleFactorPower) 
    elif yScaleFactor == None:
        yScaling=np.ones(len(stab))
    else:
        raise Exception, "didn't understand yScaleFactor"

    if xTakeLog10 == True:
        xToFit=np.log10(stab[xColumnName]/xPivot)
        xErrToFitPlus=np.log10((stab[xColumnName]+stab[xPlusErrColumnName])/xPivot)-xToFit
        xErrToFitMinus=xToFit-np.log10((stab[xColumnName]-stab[xMinusErrColumnName])/xPivot)
    else:
        xToFit=stab[xColumnName]
        xErrToFitPlus=stab[xPlusErrColumnName]
        xErrToFitMinus=stab[xMinusErrColumnName]

    if yTakeLog10 == True:
        yToFit=np.log10(yScaling*stab[yColumnName]/yPivot)
        yErrToFitPlus=np.log10(yScaling*(stab[yColumnName]+stab[yPlusErrColumnName])/yPivot)-yToFit
        yErrToFitMinus=yToFit-np.log10(yScaling*(stab[yColumnName]-stab[yMinusErrColumnName])/yPivot)
    else:
        yToFit=stab[yColumnName]
        yErrToFitPlus=stab[yPlusErrColumnName]
        yErrToFitMinus=stab[yMinusErrColumnName]

    # Swap
    if xToFit.dtype.byteorder == '>':
        xToFit=xToFit.byteswap().newbyteorder('=')
        
    stab.add_column('xToFit', xToFit)
    stab.add_column('xErrToFitPlus', xErrToFitPlus)
    stab.add_column('xErrToFitMinus', xErrToFitMinus)
    stab.add_column('yToFit', yToFit)
    stab.add_column('yErrToFitPlus', yErrToFitPlus)
    stab.add_column('yErrToFitMinus', yErrToFitMinus)
    
    # If we ever get around to fiddling with detection probabilities again, change this...
    if 'detPColumnName' in settingsDict.keys():
        if settingsDict['detPColumnName'] != 'detP':
            stab.add_column('detP', stab[settingsDict['detPColumnName']])
        #stab['detP']=np.ones(len(stab))
        #stab['detP']=stab['detP'].byteswap().newbyteorder()
        #IPython.embed()
        #sys.exit()
    else:
        stab.add_column('detP', [1.0]*len(stab))
    if 'ignoreSelectionFunction' in settingsDict.keys() and settingsDict['ignoreSelectionFunction'] == True:
        stab['detP']=np.ones(len(stab))
        
    if settingsDict['symmetriseErrors'] == True:
        xAvErr=(stab['xErrToFitPlus']+stab['xErrToFitMinus'])/2.0
        yAvErr=(stab['yErrToFitPlus']+stab['yErrToFitMinus'])/2.0
        stab['xErrToFitPlus']=xAvErr
        stab['xErrToFitMinus']=xAvErr
        stab['yErrToFitPlus']=yAvErr
        stab['yErrToFitMinus']=yAvErr
            
    # Histograms of redshift and x property distribution, one above the other
    # Fiddle with this later...
    #print "plots"
    #IPython.embed()
    #sys.exit()
    
    #fontDict={'size': 16}
    #cols=1
    #pylab.figure(figsize=(6, 8*cols))
    #pylab.subplots_adjust(0.1, 0.06, 0.97, 0.97, 0.03, 0.12)
    #pylab.subplot(2, 1, 1)
    #pylab.hist(stab['redshift'], bins = numpy.linspace(0.0, 1.5, 16), histtype = 'stepfilled', color = 
               #'#A0A0A0', ec = '#A0A0A0')
    #pylab.xlabel("$z$", fontdict = fontDict)
    #pylab.ylabel("N", fontdict = fontDict)
    #pylab.ylim(0, 60)   
    #pylab.subplot(2, 1, 2)
    #pylab.hist(stab['temp'], bins = numpy.linspace(0, 12, 13), histtype = 'stepfilled', color = 
               #'#A0A0A0', ec = '#A0A0A0')
    #pylab.xlabel("$T$ (keV)", fontdict = fontDict)
    #pylab.ylabel("N", fontdict = fontDict)
    ##pylab.yticks(ylocs, [""]*len(ylabels))   
    #pylab.ylim(0, 60)   
    #pylab.savefig(outDir+os.path.sep+"zT_histograms.pdf")
    #pylab.close()

    return stab
    
#-------------------------------------------------------------------------------------------------------------
def MCMCFit(settingsDict, tab):
    """My attempt at fitting using MCMC and maximum likelihood.
    
    settingsDict = dictionary containing MCMC parameters and settings
    
    You can choose whether to use the likelihood for 'bisector' or 'orthogonal' fitting using the 'method' key.
    
    """
    
    # Can now swap axes for testing purposes
    if 'swapAxes' in settingsDict.keys():
        swapAxes=settingsDict['swapAxes']
    else:
        swapAxes=False
    print "... swapAxes = ", swapAxes

    # Choice of method
    method=settingsDict['method']
    if method == 'orthogonal':
        likelihood=csr.fastOrthogonalLikelihood
        variables=['A', 'B', 'C', 'S']
        numFreePars=4
    elif method == 'bisector':
        likelihood=csr.fastBisectorLikelihood
        variables=['A', 'B', 'C', 'Sx', 'Sy']
        numFreePars=5
    
    settingsDict['variables']=variables         # A handy place to store this for cutting down code elsewhere
    
    scales=[]
    for v in variables:
        scales.append(settingsDict['%sScale' % (v)])
        
    # Start by writing this in python, but calling the likelihood function in cython
    # MCMC parameters
    numSamples=settingsDict['numSamples']       # Total number of random steps over likelihood surface
    burnSamples=settingsDict['burnSamples']     # Throw away initial bunch of this many samples
    thinning=settingsDict['thinning']           # Keep only every ith sample - good in some ways, bad in others
        
    # Choice of evolution models
    if settingsDict['evoModel'] == '1+z':
        log10RedshiftEvo=np.log10(tab[settingsDict['redshiftColumnName']]+1)
    elif settingsDict['evoModel'] == 'E(z)':
        log10RedshiftEvo=np.log10(tab['E(z)'])
    else:
        raise Exception, "didn't understand evoModel '%s'" % (evoModel)
    
    # To start with, we're going to use the same proposal distribution for everything
    # But later on we could dig out the correlated random numbers code to generate random parameter values that
    # satisfy the covariance we see between parameters, which would speed things up.
    cPars=selectStartParsFromPriors(settingsDict)

    #print "... starting values [A, B, C, S] = [%.2f, %.2f, %.2f, %.2f]" % (cA, cB, cC, cS)

    # Byte swapping festival to keep cython happy
    yToFit=byteSwapArr(tab['yToFit'])
    yErrToFitPlus=byteSwapArr(tab['yErrToFitPlus'])
    yErrToFitMinus=byteSwapArr(tab['yErrToFitMinus'])
    xToFit=byteSwapArr(tab['xToFit'])
    xErrToFitPlus=byteSwapArr(tab['xErrToFitPlus'])
    xErrToFitMinus=byteSwapArr(tab['xErrToFitMinus'])    
    detP=byteSwapArr(tab['detP'])
    
    if swapAxes == False:
        cProb, probArray=likelihood(cPars, yToFit, yErrToFitPlus, yErrToFitMinus, xToFit, xErrToFitPlus,
                                    xErrToFitMinus, log10RedshiftEvo, detP)  
    else:
        cProb, probArray=likelihood(cPars, xToFit, xErrToFitPlus, xErrToFitMinus, yToFit, yErrToFitPlus,
                                    yErrToFitMinus, log10RedshiftEvo, detP) 
                                                   
    if cProb == 0:
        raise Exception, "initial position in MCMC chain has zero probability - change initial values/fiddle with priors in .par file?"
        
    allPars=[]  # == 'the Markov chain'
    likelihoods=[]
    
    # Metropolis-Hastings (actually just Metropolis since our candidate distribution is symmetric)
    for k in range(numSamples):
        
        # Progress update
        tenPercent=numSamples/10
        for j in range(0,11):
            if k == j*tenPercent:
                print "... "+str(j*10)+"% complete ..."
                
        pPars=makeProposal(cPars, scales, settingsDict)
        if swapAxes == False:
            pProb, probArray=likelihood(pPars, yToFit, yErrToFitPlus, yErrToFitMinus, xToFit, xErrToFitPlus,
                                        xErrToFitMinus, log10RedshiftEvo, detP)   
        else:
            pProb, probArray=likelihood(pPars, xToFit, xErrToFitPlus, xErrToFitMinus, yToFit, yErrToFitPlus,
                                        yErrToFitMinus, log10RedshiftEvo, detP)                                                         
                                                        
        if np.isinf(pProb) == True:
            print "Hmm - infinite probability?"
            IPython.embed()
            sys.exit()
        
        # Changed below because we're now dealing with log10 probabilities instead of the actual numbers 
        alpha=pProb-cProb
        acceptProposal=False
        if alpha > 0:
            acceptProposal=True
        else:
            U=math.log10(np.random.uniform(0, 1))
            if U <= alpha:
                acceptProposal=True
        
        # Our prior is uniform, so we're really just using it to force the answer into a range
        # i.e. if it's not 1.0, then something has strayed out of the box.
        pPrior=getPPrior(pPars, settingsDict)

        if acceptProposal == True and pPrior == 1.0:
            cPars=pPars
            cProb=pProb
            # Only keep samples after burning in and also thin as we go along
            if k > burnSamples and k % thinning == 0:                     
                # If we want to plot the trace (i.e. to check mixing) then we want to store these always in some fashion
                # As it is, we're only keeping the ones that are drawn from the probability distributions
                allPars.append(cPars)
                likelihoods.append(pProb)
    
    allPars=np.array(allPars)
    likelihoods=np.array(likelihoods)
        
    # If we swap axes, it's just easier to transform back into a form we know
    if 'swapAxes' in settingsDict.keys() and settingsDict['swapAxes'] == True:
        a=-allPars[:, 0]/allPars[:, 1]
        b=1.0/allPars[:, 1]
        allPars[:, 0]=a
        allPars[:, 1]=b

    # Gewerke test to check if the chain has converged
    # If z < 2 then we're converged
    index10Percent=int(len(allPars)*0.1)
    index50Percent=int(len(allPars)*0.5)
    mean10Percent=allPars[:index10Percent].mean(axis = 0)
    mean50Percent=allPars[::-1][:index50Percent].mean(axis = 0)
    var10Percent=allPars[:index10Percent].var(axis = 0)
    var50Percent=allPars[::-1][:index50Percent].var(axis = 0)
    zStatistic=(mean10Percent-mean50Percent)/np.sqrt(var10Percent+var50Percent)
    zStatistic=np.nan_to_num(zStatistic)
    
    # Zap entries in here that are fixed (avoids round off or div 0 making them look large when we don't care)
    for i in range(len(variables)):
        v=variables[i]
        if settingsDict['%sFit' % (v)] == 'fixed':
            zStatistic[i]=0.0
            numFreePars=numFreePars-1
        
    # Max likelihood values are simply the mean of the values in the probability distribution
    # 1-sigma errors are similarly easy (could also use calc1SigmaError routine, but this is quicker)
    resultsDict={}
    for i in range(len(variables)):
        v=variables[i]
        resultsDict['%s' % (v)]=allPars[:, i].mean()
        resultsDict['%sErr' % (v)]=calc68Percentile(allPars[:, i])
    
    # Scott's translation of orthogonal scatter S into scatter in y-variable at fixed x-variable
    if method == 'orthogonal':
        s=allPars[:, 3]/np.cos(np.arctan(allPars[:, 1]))
        resultsDict['s']=s.mean()
        resultsDict['sErr']=calc68Percentile(s)
    
    # We have numFreePars above
    lnL=np.log(np.power(10, likelihoods))
    resultsDict['AIC']=2*numFreePars-2*lnL.max()
    resultsDict['AICc']=resultsDict['AIC']+(2*numFreePars*(numFreePars+1))/(float(len(tab))-numFreePars-1)
    
    resultsDict['pars']=allPars
    resultsDict['zStatistic']=zStatistic
    
    return resultsDict
            
#-------------------------------------------------------------------------------------------------------------
def makeProposal(pars, scales, settingsDict):
    """Generates random set of parameters in format [A, B, C, S] for feeding into likelihood function. 
    
    Proposal distributions are assumed Gaussian with scales [AScale, BScale, CScale, SScale].
    
    """
  
    # This makes sure that if we're testing by swapping axes, we can use the same prior scales
    # To the same space as our scales
    if 'swapAxes' in settingsDict.keys() and settingsDict['swapAxes'] == True:
        b=1.0/pars[1]
        a=-pars[0]/pars[1]
        pars[0]=a
        pars[1]=b
        
    prop=np.random.normal(pars, scales)
            
    # And back...
    if 'swapAxes' in settingsDict.keys() and settingsDict['swapAxes'] == True:
        b=1.0/prop[1]
        a=-prop[0]/prop[1]
        prop[0]=a
        prop[1]=b
            
    # Force scatters +ve
    prop[3:]=abs(prop[3:])
    
    if settingsDict['AFit'] == 'fixed':
        prop[0]=settingsDict['A0']
    if settingsDict['BFit'] == 'fixed':
        prop[1]=settingsDict['B0']        
    if settingsDict['CFit'] == 'fixed':
        prop[2]=settingsDict['C0']
    if settingsDict['method'] == 'orthogonal':
        if settingsDict['SFit'] == 'fixed':
            prop[3]=settingsDict['S0']
    elif settingsDict['method'] == 'bisector':
        if settingsDict['SxFit'] == 'fixed':
            prop[3]=settingsDict['Sx0']
        if settingsDict['SyFit'] == 'fixed':
            prop[4]=settingsDict['Sy0']     
        
    return prop

#-------------------------------------------------------------------------------------------------------------
def make1DProbDensityPlots(fitResults, settingsDict, outDir):
    """Makes 1D plots of probability density distributions
    
    """
    
    sigmaScale=5.0
    bins=30
    variables=settingsDict['variables']
    axes=range(len(variables))
    
    # Individual plots
    #for v, a in zip(variables, axes):
        #if settingsDict['%sFit' % (v)] == 'free':
            #x=np.linspace(fitResults['%s' % (v)]-sigmaScale*fitResults['%sErr' % (v)], 
                             #fitResults['%s' % (v)]+sigmaScale*fitResults['%sErr' % (v)], bins)
            #P1D=LTCythonMCMC.fast1DProbProjection(x, a, fitResults['pars'])
            #make1DPlot(x, P1D, '%s' % (v), '%s = %.3f $\pm$ %.3f' % (v, fitResults['%s' % (v)], fitResults['%sErr' % (v)]), 
                       #outDir+os.path.sep+"1DProb_%s.pdf" % (v))
   
    # Make an uber plot with multiple panels
    cols=0
    for v, a in zip(variables, axes):
        if settingsDict['%sFit' % (v)] == 'free':
            cols=cols+1
    plt.figure(figsize=(4.5*cols, 3.94))
    plt.subplots_adjust(0.02, 0.12, 0.98, 0.92, 0.1, 0.1)
    count=0
    for v, a in zip(variables, axes):
        if settingsDict['%sFit' % (v)] == 'free':
            count=count+1
            x=np.linspace(fitResults['%s' % (v)]-sigmaScale*fitResults['%sErr' % (v)], 
                             fitResults['%s' % (v)]+sigmaScale*fitResults['%sErr' % (v)], bins)
            P1D=csr.fast1DProbProjection(x, a, fitResults['pars'])
            P1D=P1D/P1D.max()
            plt.subplot(1, cols, count)
            ax=plt.gca()
            y=P1D
            fitLabel='%s = %.3f $\pm$ %.3f' % (v, fitResults['%s' % (v)], fitResults['%sErr' % (v)])
            xLabel='%s' % (v)
            plt.plot(x, y, 'k-', label = fitLabel)
            plt.xlabel(xLabel, fontdict = {'size': 14})
            plt.ylabel("")
            plt.yticks([], [])
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
            plt.ylim(0, 1.2)
            leg=plt.legend(prop = {'size': 12})
            leg.draw_frame(False)
            plt.draw()
    
    plt.savefig(outDir+os.path.sep+"1DProb_allPars.pdf")
    plt.close()

#-------------------------------------------------------------------------------------------------------------
def make1DPlot(x, y, xLabel, fitLabel, outFileName):
    """Actually makes the 1D probability plots
    
    """
    
    plt.plot(x, y, label = fitLabel)
    plt.xlabel(xLabel)
    plt.ylabel("")
    plt.legend()
    plt.savefig(outFileName)
    plt.close()
    
#-------------------------------------------------------------------------------------------------------------
def makeContourPlots(fitResults, outDir, sampleLabel):
    """This takes fit results and turns it into contour plots.
    
    """
    
    mlA, mlAErr=fitResults['A'], fitResults['AErr']
    mlB, mlBErr=fitResults['B'], fitResults['BErr']
    mlC, mlCErr=fitResults['C'], fitResults['CErr']
    mlS, mlSErr=fitResults['S'], fitResults['SErr']
    
    pars=fitResults['pars']
        
    # Make 2d contour plots of valid combinations, determined by if they have a non null 1 sigma error
    As=np.linspace(mlA-5.0*mlAErr-math.fmod(mlA-5.0*mlAErr, 0.1), mlA+7.0*mlAErr-math.fmod(mlA+7.0*mlAErr, 0.1), 81)
    Bs=np.linspace(mlB-5.0*mlBErr-math.fmod(mlB-5.0*mlBErr, 0.1), mlB+7.0*mlBErr-math.fmod(mlB+7.0*mlBErr, 0.1), 81)
    Cs=np.linspace(mlC-5.0*mlCErr-math.fmod(mlC-5.0*mlCErr, 0.1), mlC+7.0*mlCErr-math.fmod(mlC+7.0*mlCErr, 0.1), 81)
    Ss=np.linspace(mlS-5.0*mlSErr-math.fmod(mlS-5.0*mlSErr, 0.05), mlS+7.0*mlSErr-math.fmod(mlS+7.0*mlSErr, 0.05), 81)
    if mlAErr > 0 and mlBErr > 0:     
        outFileName=outDir+os.path.sep+"contours_AvB_"+sampleLabel+".pdf"
        PDist2D=csr.fast2DProbProjection(As, Bs, 0, 1, pars)
        astImages.saveFITS(outFileName.replace(".pdf", ".fits"), PDist2D, None)
        probContourPlot(As, Bs, "A", "B", 0.1, 0.1, mlA, mlB, mlAErr, mlBErr, PDist2D, outFileName)
    if mlAErr > 0 and mlCErr > 0:
        outFileName=outDir+os.path.sep+"contours_AvC_"+sampleLabel+".pdf"
        PDist2D=csr.fast2DProbProjection(As, Cs, 0, 2, pars)
        probContourPlot(As, Cs, "A", "C", 0.1, 0.5, mlA, mlC, mlAErr, mlCErr, PDist2D, outFileName)
        astImages.saveFITS(outFileName.replace(".pdf", ".fits"), PDist2D, None)
    if mlAErr > 0 and mlSErr > 0:
        outFileName=outDir+os.path.sep+"contours_AvS_"+sampleLabel+".pdf"
        PDist2D=csr.fast2DProbProjection(As, Ss, 0, 3, pars)
        probContourPlot(As, Ss, "A", "S", 0.1, 0.05, mlA, mlS, mlAErr, mlSErr, PDist2D, outFileName)
        astImages.saveFITS(outFileName.replace(".pdf", ".fits"), PDist2D, None)
    if mlBErr > 0 and mlCErr > 0:
        outFileName=outDir+os.path.sep+"contours_BvC_"+sampleLabel+".pdf"
        PDist2D=csr.fast2DProbProjection(Bs, Cs, 1, 2, pars)
        probContourPlot(Bs, Cs, "B", "C", 0.1, 0.5, mlB, mlC, mlBErr, mlCErr, PDist2D, outFileName)
        astImages.saveFITS(outFileName.replace(".pdf", ".fits"), PDist2D, None)
        
#-------------------------------------------------------------------------------------------------------------
def probContourPlot(par1Values, par2Values, par1Label, par2Label, par1TickStep, par2TickStep, mlPar1, mlPar2, 
                    mlPar1Err, mlPar2Err, PDist2D, outFileName):
    """Make a 2d contour plot of probability surface of given parameters.
    
    par1Values      = values for parameter 1 (plotted on Y axis)
    par2Values      = values for parameter 2 (plotted on X axis)
    par1Label       = text label for Y axis
    par2Label       = text label for X axis
    par1TickStep    = tick step along Y axis
    par2TickStep    = tick step along X axis
    mlPar1          = maximum likelihood value for parameter 1
    mlPar2          = maximum likelihood value for parameter 2
    mlPar1Err       = 1d 1-sigma error in parameter 1
    mlPar2Err       = 1d 1-sigma error in parameter 2
    PDist2D         = 2d likelihood surface, made using fast2DProbProjection
    
    """
    
    tck1=interpolate.splrep(par1Values, np.arange(par1Values.shape[0]))
    par1TickLabels=np.arange(par1Values.min(), par1Values.max(), par1TickStep)
    par1TickIndices=interpolate.splev(par1TickLabels, tck1)
    plt.yticks(par1TickIndices, par1TickLabels)
    
    tck2=interpolate.splrep(par2Values, np.arange(par2Values.shape[0]))
    par2TickLabels=np.arange(par2Values.min(), par2Values.max(), par2TickStep)
    par2TickIndices=interpolate.splev(par2TickLabels, tck2)
    plt.xticks(par2TickIndices, par2TickLabels)

    # We have to smooth to get decent looking contours
    # Gaussian smoothing preserves the normalisation
    # NOTE: smoothing only needed if very fine grid
    PDist2D=ndimage.gaussian_filter(PDist2D, 1)

    # Work out where to put contours
    sigma1Level=calc2DProbThreshold(PDist2D, 0.683)
    sigma2Level=calc2DProbThreshold(PDist2D, 0.95)
    
    plt.contour(PDist2D, [sigma1Level, sigma2Level], colors = 'b')
            
    # Save plot - trim down area first (?) and add axes labels
    plt.plot(interpolate.splev(mlPar2, tck2), interpolate.splev(mlPar1, tck1), 'r*', 
               label = "%s = %.2f $\pm$ %.2f, %s = %.2f $\pm$ %.2f" % (par1Label, mlPar1, mlPar1Err, par2Label, mlPar2, mlPar2Err))
    plt.legend(numpoints = 1)
    plt.xlabel(par2Label)
    plt.ylabel(par1Label)
    
    if outFileName != None:
        plt.savefig(outFileName)
    plt.close()
    
#-------------------------------------------------------------------------------------------------------------
def calc1SigmaError(par1d, prob1d, mlParValue):
    """Calculates 1d 1-sigma error on a parameter (marginalised, is the word I'm looking for I think) relative
    to the maximum likelihood value.
    
    NOTE: Now we're using MCMC, the regular calc68Percentile routine below works just fine, and is quicker
    than this.
    
    """
    
    norm=np.trapz(prob1d, par1d)
    prob1d=prob1d/norm
    tckPDist=interpolate.splrep(par1d, prob1d)
    target=0.683    # 1 sigma
    dRange=np.linspace(0.0, par1d.max()-mlParValue, 1000)  # we need to wok out how to choose sensible values
    bestDiff=1e6
    dBest=1e6
    for d in dRange:
        integrationRange=np.linspace(mlParValue-d, mlParValue+d, 1000)
        diff=abs(target-np.trapz(interpolate.splev(integrationRange, tckPDist), integrationRange))
        if diff < bestDiff:
            bestDiff=diff
            dBest=d
    
    return dBest

#-------------------------------------------------------------------------------------------------------------
def calc2DProbThreshold(PDist2D, probThresh):
    """Calculates threshold probability per pixel in PDist2D needed to draw confidence contours at e.g.
    1-sigma, 2-sigma level 
    
    """
    
    p=PDist2D.flatten()
    p.sort()
    p=p[::-1]
    pCumSum=p.cumsum()
    diff=abs(pCumSum-probThresh)
    pIndex=diff.tolist().index(diff.min())    
    pLevel=p[pIndex]
    
    return pLevel
    
#------------------------------------------------------------------------------------------------------------
def calc68Percentile(arr):
    """Calculates the 68-percentile (i.e. equivalent to 1-sigma error) from an array.
    
    """
    
    res=np.abs(arr-np.median(arr))
    res=np.sort(res)
    index=int(round(0.683*arr.shape[0]))
    err=res[index]
    
    return err
        
#-------------------------------------------------------------------------------------------------------------
def makeScalingRelationPlot(sampleTab, fitResults, outDir, sampleDict, settingsDict):
    """Make a scaling relation plot.
    
    sampleDict  = the dictionary defining the sample (e.g. min z, max z etc.)
    
    """
    
    # Stuff we need from settings...
    xColumnName=settingsDict['xColumnName']
    xPlusErrColumnName=settingsDict['xPlusErrColumnName']
    xMinusErrColumnName=settingsDict['xMinusErrColumnName']
    yColumnName=settingsDict['yColumnName']
    yPlusErrColumnName=settingsDict['yPlusErrColumnName']
    yMinusErrColumnName=settingsDict['yMinusErrColumnName']
    
    xPivot=settingsDict['xPivot']
    
    xTakeLog10=settingsDict['xTakeLog10']
    yTakeLog10=settingsDict['yTakeLog10']
    
    redshiftColumnName=settingsDict['redshiftColumnName']
    xScaleFactor=settingsDict['xScaleFactor']
    yScaleFactor=settingsDict['yScaleFactor']
    yScaleFactorPower=settingsDict['yScaleFactorPower']
    
    # The plot
    plt.figure(figsize=(10, 10))
    plt.axes([0.1, 0.1, 0.85, 0.85])
    
    if yScaleFactor != None:
        yPlot=np.power(sampleTab['E(z)'], yScaleFactorPower)*sampleTab[yColumnName]
        yPlotErrs=np.array([np.power(sampleTab['E(z)'], yScaleFactorPower)*sampleTab[yMinusErrColumnName], 
                           np.power(sampleTab['E(z)'], yScaleFactorPower)*sampleTab[yPlusErrColumnName]])
    else:
        yPlot=sampleTab[yColumnName]
        yPlotErrs=np.array([sampleTab[yMinusErrColumnName], 
                            sampleTab[yPlusErrColumnName]])
    
    plt.errorbar(sampleTab[xColumnName], yPlot,
                   yerr = yPlotErrs,
                   xerr = np.array([sampleTab[xMinusErrColumnName],
                                    sampleTab[xPlusErrColumnName]]),
                   fmt = 'kD', mec = 'k', label = sampleDict['label']+" (N=%d)" % (len(sampleTab)))           
    if xTakeLog10 == True and yTakeLog10 == True:
        plt.loglog()
    elif xTakeLog10 == True and yTakeLog10 == False:
        plt.semilogx()
    elif xTakeLog10 == False and yTakeLog10 == True:
        plt.semilogy()

    #cmdata=np.outer(np.linspace(0, 1, 10), np.linspace(0, 1, 10)) #  to easily make a colorbar 0-1
    #cmim=plt.imshow(cmdata, cmap = "gray")
    #ax=plt.axes([0.1, 0.17, 0.85, 0.78])
    if np.sum(np.equal(sampleTab['detP'], 1.0)) == len(sampleTab):
        shadeByDetP=False
    else:
        shadeByDetP=True
    if shadeByDetP == True:
        for row, pY in zip(sampleTab, yPlot):
            plt.plot(row[xColumnName], [pY], 'D', color = (row['detP'], row['detP'], row['detP'])) 

        
    plotRange=np.linspace(settingsDict['xPlotMin'], settingsDict['xPlotMax'], 100)
    if xTakeLog10 == True and yTakeLog10 == True:
        yFit=settingsDict['yPivot']*np.power(10, fitResults['A'])*np.power((plotRange/xPivot), fitResults['B'])
    elif xTakeLog10 == False and yTakeLog10 == False:
        yFit=settingsDict['yPivot']*(fitResults['A']+fitResults['B']*(plotRange/xPivot))
    else:
        raise Exception, "add semilogx, semilogy fit line code"
    
    fitLabel='%s (%s) = 10$^{%.2f \pm %.2f}$ (%s/%.1f %s)$^{%.2f \pm %.2f}$' % (settingsDict['yPlotLabel'], settingsDict['yPlotLabelUnits'], fitResults['A'], fitResults['AErr'], settingsDict['xPlotLabel'], xPivot, settingsDict['xPlotLabelUnits'], fitResults['B'], fitResults['BErr'])

    yLabel="%s (%s)" % (settingsDict['yPlotLabel'], settingsDict['yPlotLabelUnits'])

    if settingsDict['yScaleFactor'] == "E(z)":
        fitLabel="$E^{%d}(z)$ " % (settingsDict['yScaleFactorPower'])+fitLabel
        yLabel="$E^{%d}(z)$ " % (settingsDict['yScaleFactorPower'])+yLabel

    plt.plot(plotRange, yFit, 'b--', label = fitLabel) 

    ## Below is just diagnostic
    #if sampleLabel == 'REXCESS':
        #prattLabel='$L_{\sf X}$ (erg s$^{-1}$) = 10$^{44.85 \pm 0.06}$ ($T/5.0$ keV)$^{3.35 \pm 0.32}$' 
        #prattLabel="$E^{-1}(z)$ "+prattLabel
        #prattLabel="P09: "+prattLabel
        #prattLX=np.power(10, 44.85)*np.power((plotRange/5.0), 3.35)
        #plt.plot(plotRange, prattLX, 'r:', label = prattLabel)
        #sample['plotLabel']=""
        
    plt.ylabel(yLabel, size = 16)
    plt.xlabel("%s (%s)" % (settingsDict['xPlotLabel'], settingsDict['xPlotLabelUnits']), size = 16)
    plt.xlim(settingsDict['xPlotMin'], settingsDict['xPlotMax'])
    plt.ylim(settingsDict['yPlotMin'], settingsDict['yPlotMax'])
    
    if settingsDict['showPlotLegend'] == True:
        leg=plt.legend(loc = 'upper left', prop = {'size': 16}, scatterpoints = 1, numpoints = 1)
        leg.draw_frame(False)
        plt.draw()
            
    ax=plt.gca()
    plt.text(0.95, 0.05, sampleDict['plotLabel'], ha = 'right', va = 'center', transform = ax.transAxes, 
            fontdict = {"size": 16, "linespacing" : 1.2, 'family': 'serif'})

    outFileName=outDir+os.path.sep+"scalingRelation_%s_%s.pdf" % (yColumnName, xColumnName)
    plt.savefig(outFileName)    
    plt.close()

#-------------------------------------------------------------------------------------------------------------
def makeScalingRelationPlot_ABC(sampleTab, fitResults, outDir, sampleDict, settingsDict, mode = 'normal'):
    """Make a scaling relation plot with y values scaling by normalisation and z evolution.
    
    sampleDict  = the dictionary defining the sample (e.g. min z, max z etc.)
    
    """
    
    # Stuff we need from settings...
    xColumnName=settingsDict['xColumnName']
    xPlusErrColumnName=settingsDict['xPlusErrColumnName']
    xMinusErrColumnName=settingsDict['xMinusErrColumnName']
    yColumnName=settingsDict['yColumnName']
    yPlusErrColumnName=settingsDict['yPlusErrColumnName']
    yMinusErrColumnName=settingsDict['yMinusErrColumnName']
    
    xPivot=settingsDict['xPivot']
    
    xTakeLog10=settingsDict['xTakeLog10']
    yTakeLog10=settingsDict['yTakeLog10']
    
    redshiftColumnName=settingsDict['redshiftColumnName']
    xScaleFactor=settingsDict['xScaleFactor']
    yScaleFactor=settingsDict['yScaleFactor']
    yScaleFactorPower=settingsDict['yScaleFactorPower']
    
    # The plot...
    if yScaleFactor != None:
        yPlot=np.power(sampleTab['E(z)'], yScaleFactorPower)*sampleTab[yColumnName]
        yPlotErrs=np.array([np.power(sampleTab['E(z)'], yScaleFactorPower)*sampleTab[yMinusErrColumnName], 
                           np.power(sampleTab['E(z)'], yScaleFactorPower)*sampleTab[yPlusErrColumnName]])
    else:
        yPlot=sampleTab[yColumnName]
        yPlotErrs=np.array([sampleTab[yMinusErrColumnName], 
                            sampleTab[yPlusErrColumnName]])

    fitLabel='%s = 10$^{%.2f \pm %.2f}$ (%s/%d)$^{%.2f \pm %.2f}$' % (settingsDict['yPlotLabel'], fitResults['A'], fitResults['AErr'], settingsDict['xPlotLabel'], xPivot, fitResults['B'], fitResults['BErr'])

    yLabel="%s (%s)" % (settingsDict['yPlotLabel'], settingsDict['yPlotLabelUnits'])

    if settingsDict['evoModel'] == '1+z':
        yPlot=np.power(sampleTab[redshiftColumnName]+1, -fitResults['C'])*yPlot
        yPlotErrs=np.power(sampleTab[redshiftColumnName]+1, -fitResults['C'])*yPlotErrs
        fitLabel=fitLabel+' (1+$z$)$^{%s}$' % (fitResults['plotLabel_C'])
        yLabel=yLabel.replace("(%s)" % (settingsDict['yPlotLabelUnits']), "(1+$z$)$^{%.1f}$ (%s)" % (-1*fitResults['C'], settingsDict['yPlotLabelUnits']))        
    elif settingsDict['evoModel'] == 'E(z)':
        yPlot=np.power(sampleTab['E(z)'], -fitResults['C'])*yPlot
        yPlotErrs=np.power(sampleTab['E(z)'], -fitResults['C'])*yPlotErrs
        fitLabel=fitLabel+' $E(z)^{%s}$' % (fitResults['plotLabel_C'])
        yLabel=yLabel.replace("(%s)" % (settingsDict['yPlotLabelUnits']), "$E(z)^{%.1f}$ (%s)" % (-1*fitResults['C'], settingsDict['yPlotLabelUnits']))

    if settingsDict['yScaleFactor'] == "E(z)":
        fitLabel="$E^{%d}(z)$ " % (settingsDict['yScaleFactorPower'])+fitLabel
        yLabel="$E^{%d}(z)$ " % (settingsDict['yScaleFactorPower'])+yLabel
        
    if mode == 'normal':
        plt.figure(figsize=(8, 8))
        ax=plt.axes([0.11, 0.1, 0.86, 0.85])

        plotRange=np.linspace(0.1*sampleTab[xColumnName].min(), 10*sampleTab[xColumnName].max(), 100)
        yFit=np.power(10, fitResults['A'])*np.power((plotRange/xPivot), fitResults['B'])
    
        plt.plot(plotRange, yFit, 'b--', label = fitLabel) 
    
        outFileName=outDir+os.path.sep+"scalingRelation_%s_%s_ABC.pdf" % (settingsDict['yColumnName'], settingsDict['xColumnName'])
        
        # Old
        #plt.errorbar(sampleTab['temp'], plotLXs,
                       #yerr = plotLXErrs,
                       #xerr = np.array([sampleTab['temp_min'],
                                        #sampleTab['temp_max']]),
                       #fmt = 'kD', mec = 'k', label = sampleLabel+" (N=%d)" % (len(sampleTab)))
                       
        # New (coding by redshift)
        zBins=[[0.0, 0.25], [0.25, 0.5], [0.5, 1.5]]
        labels=["0.0 < $z$ < 0.25", "0.25 < $z$ < 0.5", "0.5 < $z$ < 1.5"]
        #colours=['k', [0.5, 0, 1], [1, 0.5, 0]]
        colours=['k', 'c', 'r']
        symbols=['D', 'o', '^']
        for zBin, col, s, l in zip(zBins, colours, symbols, labels):
            mask=np.logical_and(np.greater(sampleTab[redshiftColumnName], zBin[0]), np.less_equal(sampleTab[redshiftColumnName], zBin[1]))
            plt.errorbar(sampleTab[xColumnName][mask], yPlot[mask],
                yerr = yPlotErrs[:, mask],
                xerr = np.array([sampleTab[xMinusErrColumnName][mask],
                                 sampleTab[xPlusErrColumnName][mask]]),
                fmt = s, ecolor = col, mfc = col, mec = col, label = l)
        
    elif mode == 'PDetCoded':
        plotRange=np.linspace(0.1, 22.0, 100)
        fitLXs=np.power(10, fitResults['A'])*np.power((plotRange/pivotT), fitResults['B'])
        #fitLabel='$L_{\sf X}$ (erg s$^{-1}$) = 10$^{%.2f \pm %.2f}$ ($T/%.1f$ keV)$^{%.2f \pm %.2f}$ (1+$z$)$^{%.2f \pm %.2f}$' % (fitResults['A'], fitResults['AErr'], pivotT, fitResults['B'], fitResults['BErr'], fitResults['C'], fitResults['CErr'])
        plt.plot(plotRange, fitLXs, 'b--', label = fitLabel) 
        outFileName=outDir+os.path.sep+"L-T_ABC_PDetCoded.pdf"
        plt.figure(figsize=(8, 8))
        plt.axes([0.5, 0.5, 0.1, 0.1])
        cmdata=np.outer(np.linspace(0, 1, 10), np.linspace(0, 1, 10)) #  to easily make a colorbar 0-1
        cmim=plt.imshow(cmdata, cmap = "gray")
        ax=plt.axes([0.1, 0.17, 0.85, 0.78])
        for row, pLX in zip(sampleTab, plotLXs):
            plt.plot(row['temp'], [pLX], 'D', color = (row['detP'], row['detP'], row['detP'])) 
        cmax=plt.axes([0.1, 0.075, 0.85, 0.1], frameon=False)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar(cmim, orientation = 'v', aspect = 40.0)
        plt.figtext(0.52, 0.03, "P$_{\sf det}$", va = 'center', ha = 'center')
        plt.axes(ax)
    else:
        raise Exception, "didn't understand mode"

    plt.loglog()

    plt.ylabel(yLabel, size = 16)
    plt.xlabel("%s (%s)" % (settingsDict['xPlotLabel'], settingsDict['xPlotLabelUnits']), size = 16)
    plt.xlim(settingsDict['xPlotMin'], settingsDict['xPlotMax'])
    plt.ylim(settingsDict['yPlotMin'], settingsDict['yPlotMax'])
        
    #leg=plt.legend(loc = 'upper left', prop = {'size': 16}, scatterpoints = 1, numpoints = 1)
    #leg.draw_frame(False)
    plt.draw()
            
    ax=plt.gca()
    plt.text(0.95, 0.05, sampleDict['plotLabel'], ha = 'right', va = 'center', transform = ax.transAxes, 
            fontdict = {"size": 16, "linespacing" : 1.2, 'family': 'serif'})

    plt.savefig(outFileName)    
    plt.close()
        
#-------------------------------------------------------------------------------------------------------------
def makeScalingRelationPlots_sideBySide(sampleDefs, outDir, settingsDict):
    """Makes side by side subpanel plots of all the scaling relations in sampleDefs
    
    """

    # Stuff we need from settings...
    xColumnName=settingsDict['xColumnName']
    xPlusErrColumnName=settingsDict['xPlusErrColumnName']
    xMinusErrColumnName=settingsDict['xMinusErrColumnName']
    yColumnName=settingsDict['yColumnName']
    yPlusErrColumnName=settingsDict['yPlusErrColumnName']
    yMinusErrColumnName=settingsDict['yMinusErrColumnName']
    
    xPivot=settingsDict['xPivot']
    
    xTakeLog10=settingsDict['xTakeLog10']
    yTakeLog10=settingsDict['yTakeLog10']
    
    redshiftColumnName=settingsDict['redshiftColumnName']
    xScaleFactor=settingsDict['xScaleFactor']
    yScaleFactor=settingsDict['yScaleFactor']
    yScaleFactorPower=settingsDict['yScaleFactorPower']
    
    # Make an uber plot with multiple panels
    # NOTE: add adjustable layout later...
    cols=len(sampleDefs)
    plt.figure(figsize=(6*cols, 6))
    plt.subplots_adjust(0.05, 0.1, 0.99, 0.99, 0.02, 0.02)
    count=0
    for s in sampleDefs:
        
        sampleTab=s['stab']
        fitResults=s['fitResults']
        
        count=count+1
        plt.subplot(1, cols, count)

        if yScaleFactor != None:
            yPlot=np.power(sampleTab['E(z)'], yScaleFactorPower)*sampleTab[yColumnName]
            yPlotErrs=np.array([np.power(sampleTab['E(z)'], yScaleFactorPower)*sampleTab[yMinusErrColumnName], 
                            np.power(sampleTab['E(z)'], yScaleFactorPower)*sampleTab[yPlusErrColumnName]])
        else:
            yPlot=sampleTab[yColumnName]
            yPlotErrs=np.array([sampleTab[yMinusErrColumnName], 
                                sampleTab[yPlusErrColumnName]])
        
        plt.errorbar(sampleTab[xColumnName], yPlot,
                    yerr = yPlotErrs,
                    xerr = np.array([sampleTab[xMinusErrColumnName],
                                        sampleTab[xPlusErrColumnName]]),
                    fmt = 'kD', mec = 'k', label = s['label']+" (N=%d)" % (len(sampleTab)))           
        plt.loglog()
            
        plotRange=np.linspace(0.1*sampleTab[xColumnName].min(), 10*sampleTab[xColumnName].max(), 100)
        yFit=settingsDict['yPivot']*np.power(10, fitResults['A'])*np.power((plotRange/xPivot), fitResults['B'])
        
        fitLabel='%s (%s) = 10$^{%.2f \pm %.2f}$ (%s/%.1f %s)$^{%.2f \pm %.2f}$' % (settingsDict['yPlotLabel'], settingsDict['yPlotLabelUnits'], fitResults['A'], fitResults['AErr'], settingsDict['xPlotLabel'], xPivot, settingsDict['xPlotLabelUnits'], fitResults['B'], fitResults['BErr'])

        yLabel="%s (%s)" % (settingsDict['yPlotLabel'], settingsDict['yPlotLabelUnits'])

        if settingsDict['yScaleFactor'] == "E(z)":
            fitLabel="$E^{%d}(z)$ " % (settingsDict['yScaleFactorPower'])+fitLabel
            yLabel="$E^{%d}(z)$ " % (settingsDict['yScaleFactorPower'])+yLabel

        plt.plot(plotRange, yFit, 'b--', label = fitLabel) 
            
        plt.ylabel(yLabel, size = 16)
        plt.xlabel("%s (%s)" % (settingsDict['xPlotLabel'], settingsDict['xPlotLabelUnits']), size = 16)
                        
        ax=plt.gca()
        plt.text(0.95, 0.05, s['plotLabel'], ha = 'right', va = 'center', transform = ax.transAxes, 
                fontdict = {"size": 16, "linespacing" : 1.2, 'family': 'serif'})
        
        if count > 1:
            ylocs, ylabels=plt.yticks()
            plt.ylabel("")
            plt.yticks(ylocs, [""]*len(ylabels))   

        plt.xlim(settingsDict['xPlotMin'], settingsDict['xPlotMax'])
        plt.ylim(settingsDict['yPlotMin'], settingsDict['yPlotMax'])
        
    outFileName=outDir+os.path.sep+"scalingRelation_multiPlot_%s_%s.pdf" % (yColumnName, xColumnName)
    
    plt.savefig(outFileName)        
    plt.close()
    
#-------------------------------------------------------------------------------------------------------------
def makeRoundedPlotLabelStrings(fitResults, variables, numSigFig = 1):
    """Add plot labels to fitResults, to given number of sig fig, taking care of rounding
    
    NOTE: disabled the rounding for now
    """
    
    # Not rounding, just dp not sf
    dps=[2, 2, 1, 3, 3]
    for p, dp in zip(variables, dps):
        if fitResults['%sErr' % (p)] != 0:
            fmt="%."+str(dp)+"f"
            valStr=fmt % (fitResults['%s' % (p)])
            errStr=fmt % (fitResults['%sErr' % (p)])
            fitResults['plotLabel_%s' % (p)]="%s \pm %s" % (valStr, errStr)            
        
#-------------------------------------------------------------------------------------------------------------
def makeNormEvoPlot(stab, fitResults, outDir, settingsDict):
    """Makes plot of evolution of the normalisation.
    
    """
   
    zs=np.linspace(0, 2.0, 100)
    Ez=[]
    for z in zs:
        Ez.append(astCalc.Ez(z))
    Ez=np.array(Ez)
    
    plt.figure(figsize=(8,6))
    plt.axes([0.13, 0.1, 0.85, 0.86])
    
    xColumnName=settingsDict['xColumnName']
    yColumnName=settingsDict['yColumnName']
    redshiftColumnName=settingsDict['redshiftColumnName']
    
    yLabel="%s / %s$_{Fit (z=0)}$" % (settingsDict['yPlotLabel'], settingsDict['yPlotLabel'])
    
    # If we have applied E(z)^{some power}, we want to plot that expected scaling,
    # as well as a null line for no evolution
    if settingsDict['yScaleFactor'] == 'E(z)':
        dataNormalisation=((np.power(stab['E(z)'], settingsDict['yScaleFactorPower'])*stab[yColumnName])/np.power(stab[xColumnName]/settingsDict['xPivot'], fitResults['B']))/np.power(10, fitResults['A'])
        nullLine=np.power(Ez, settingsDict['yScaleFactorPower']) # because E(z)^{some power} is flat in this form, null line is not
        yScalingLine=np.ones(len(Ez))  # because we've scaled it out it's flat
        yLabel="($E^{-1}(z)$ %s) / %s$_{Fit (z=0)}$" % (settingsDict['yPlotLabel'], settingsDict['yPlotLabel'])
    else:
        dataNormalisation=(stab[yColumnName]/np.power(stab[xColumnName]/settingsDict['xPivot'], fitResults['B']))/np.power(10, fitResults['A'])
        nullLine=np.zeros(len(Ez))
        yScalingLine=None
        yLabel="%s / %s$_{Fit (z=0)}$" % (settingsDict['yPlotLabel'], settingsDict['yPlotLabel'])

    dataLabel='%s$_{Fit (z=0)}$ = (%s/%d)$^{%.2f}$ / 10$^{%.2f}$' % (settingsDict['yPlotLabel'], settingsDict['xPlotLabel'], settingsDict['xPivot'], fitResults['B'], fitResults['A'])
    
    if settingsDict['yScaleFactor'] == 'E(z)':
        # Look for fractions
        if settingsDict['yScaleFactorPower'] == -1:
            yScalingLineLabel='$E(z)$'
        elif abs(settingsDict['yScaleFactorPower']) == 2/3.0:
            yScalingLineLabel='$E(z)$'
            powerFactor=settingsDict['yScaleFactorPower']
            # Need to swap power, remember we scaled these out...
            if powerFactor > 0:
                yScalingLineLabel=yScalingLineLabel+"$^{-2/3}$"
            else:
                yScalingLineLabel=yScalingLineLabel+"$^{2/3}$"
        else:
            print "yScalingLineLabel fraction handling?"
            IPython.embed()
            sys.exit()
    
    plt.plot(stab[redshiftColumnName], dataNormalisation, 'kD', label = dataLabel)
    if yScalingLine != None:
        plt.plot(zs, yScalingLine, 'b--', label = yScalingLineLabel, lw = 2)
    plt.plot(zs, nullLine, 'g-.', label = 'no evolution', lw = 2)
    
    if settingsDict['evoModel'] == '1+z':
        plt.plot(zs, np.power(1+zs, fitResults['C']), 'r', lw = 2, label = '(1+z)$^{%.2f \pm %.2f}$' % (fitResults['C'], fitResults['CErr']))
        shadedX=np.linspace(0, 2.0, 100)
        shadedYPlus=np.power(shadedX+1, fitResults['C']+fitResults['CErr'])
        shadedYMinus=np.power(shadedX+1, fitResults['C']-fitResults['CErr'])
    elif settingsDict['evoModel'] == 'E(z)':
        plt.plot(zs, np.power(Ez, fitResults['C']), 'r', lw = 2, label = '$E(z)^{%.2f \pm %.2f}$' % (fitResults['C'], fitResults['CErr']))
        shadedX=np.linspace(0, 2.0, len(Ez))
        shadedYPlus=np.power(Ez, fitResults['C']+fitResults['CErr'])
        shadedYMinus=np.power(Ez, fitResults['C']-fitResults['CErr'])
    
    if fitResults['C'] < 0:
        loc="upper right"
    else:
        loc="lower left"
    leg=plt.legend(loc = loc, prop = {'size': 14}, numpoints = 1)
    leg.draw_frame(False)
    plt.draw()
    plt.xlabel("$z$", fontdict = {'size': 20})
    plt.ylabel(yLabel, fontdict = {'size': 20})       
        
    xs=shadedX.tolist()+shadedX[::-1].tolist()
    ys=shadedYPlus.tolist()+shadedYMinus[::-1].tolist()
    plt.fill(xs, ys, 'b', alpha=0.2, edgecolor='none', label = "None", lw = 0.1)
    
    plt.semilogy()
    #plt.loglog()
    plt.xlim(0, 1.6)
    plt.ylim(1e-2, 1e2)
    
    plt.savefig(outDir+os.path.sep+"normEvo_%s_%s.pdf" % (yColumnName, xColumnName))
    plt.close()

#-------------------------------------------------------------------------------------------------------------
def makePaperContourPlots(fitResults, parDict, outDir):
    """Special case of plots, for 4 parameter fits, for the paper.
    
    """
    
    mlA, mlAErr=fitResults['A'], fitResults['AErr']
    mlB, mlBErr=fitResults['B'], fitResults['BErr']
    mlC, mlCErr=fitResults['C'], fitResults['CErr']
    mlS, mlSErr=fitResults['S'], fitResults['SErr']
    
    pars=fitResults['pars']
    
    # We only want to go on if we have a full set...
    if mlAErr == 0 or mlBErr == 0 or mlCErr == 0 or mlSErr == 0:
        return None
    
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(0.08, 0.07, 0.97, 0.97, 0.0, 0.0)
    
    # Make 2d contour plots of valid combinations, determined by if they have a non null 1 sigma error
    # NOTE: here steps have to be smaller than AStep, BStep, CStep, SStep below
    # NOTE: any strange numbers in here are fiddling to get non-overlapping plot labels
    As=np.linspace(mlA-5.0*mlAErr-math.fmod(mlA-5.0*mlAErr, 0.1), mlA+5.0*mlAErr-math.fmod(mlA+5.0*mlAErr, 0.1), 81)
    Bs=np.linspace(mlB-5.0*mlBErr-math.fmod(mlB-5.0*mlBErr, 0.1), mlB+5.0*mlBErr-math.fmod(mlB+5.0*mlBErr, 0.1), 81)
    Cs=np.linspace(mlC-5.0*mlCErr-math.fmod(mlC-5.0*mlCErr, 0.1), mlC+5.0*mlCErr-math.fmod(mlC+5.0*mlCErr, 0.1), 81)
    Ss=np.linspace(mlS-5.0*mlSErr-math.fmod(mlS-5.0*mlSErr, 0.01), mlS+6.0*mlSErr-math.fmod(mlS+6.0*mlSErr, 0.01), 81)
    
    # Steps for tick label plotting adjustment
    AStep=0.2
    BStep=0.4
    CStep=1.0
    SStep=0.02
    
    # Bottom row   
    # AB
    plt.subplot(4, 4, 15)
    PDist2D=csr.fast2DProbProjection(As, Bs, 0, 1, pars)
    probContourPlot_subPlot(As, Bs, "A", "B", AStep, BStep, mlA, mlB, mlAErr, mlBErr, PDist2D, noYLabels = True)
    # AC
    plt.subplot(4, 4, 14)
    PDist2D=csr.fast2DProbProjection(As, Cs, 0, 2, pars)
    probContourPlot_subPlot(As, Cs, "A", "C", AStep, CStep, mlA, mlC, mlAErr, mlCErr, PDist2D, noYLabels = True)
    # AS
    plt.subplot(4, 4, 13)
    PDist2D=csr.fast2DProbProjection(As, Ss, 0, 3, pars)
    probContourPlot_subPlot(As, Ss, "A", "S", AStep, SStep, mlA, mlS, mlAErr, mlSErr, PDist2D)  
    
    # Middle row
    # BC
    plt.subplot(4, 4, 10)
    PDist2D=csr.fast2DProbProjection(Bs, Cs, 1, 2, pars)
    probContourPlot_subPlot(Bs, Cs, "B", "C", BStep, CStep, mlB, mlC, mlBErr, mlCErr, PDist2D, noXLabels = True, noYLabels = True)  
    # BS
    plt.subplot(4, 4, 9)
    PDist2D=csr.fast2DProbProjection(Bs, Ss, 1, 3, pars)
    probContourPlot_subPlot(Bs, Ss, "B", "S", BStep, SStep, mlB, mlS, mlBErr, mlSErr, PDist2D, noXLabels = True)  
    
    # Top row
    # CS
    plt.subplot(4, 4, 5)
    PDist2D=csr.fast2DProbProjection(Cs, Ss, 2, 3, pars)
    probContourPlot_subPlot(Cs, Ss, "C", "S", CStep, SStep, mlC, mlS, mlCErr, mlSErr, PDist2D, noXLabels = True)      
    
    # 1D plots
    # S
    plt.subplot(4, 4, 1)
    PDist1D=csr.fast1DProbProjection(Ss, 3, pars)
    probPlot1D_subPlot(Ss, "S", SStep, mlS, mlSErr, PDist1D, fitResults['plotLabel_S'], noYLabels = True, noXLabels = True)
    # C
    plt.subplot(4, 4, 6)
    PDist1D=csr.fast1DProbProjection(Cs, 2, pars)
    probPlot1D_subPlot(Cs, "C", CStep, mlC, mlCErr, PDist1D, fitResults['plotLabel_C'], noYLabels = True, noXLabels = True)
    # B
    plt.subplot(4, 4, 11)
    PDist1D=csr.fast1DProbProjection(Bs, 1, pars)
    probPlot1D_subPlot(Bs, "B", BStep, mlB, mlBErr, PDist1D, fitResults['plotLabel_B'], noYLabels = True, noXLabels = True)
    # A
    plt.subplot(4, 4, 16)
    PDist1D=csr.fast1DProbProjection(As, 0, pars)
    probPlot1D_subPlot(As, "A", AStep, mlA, mlAErr, PDist1D, fitResults['plotLabel_A'], noYLabels = True, noXLabels = False)

    plt.savefig(outDir+os.path.sep+"2DProb_allPars.pdf")
    plt.close()

#-------------------------------------------------------------------------------------------------------------
def probPlot1D_subPlot(par1Values, par1Label, par1TickStep, mlPar1, mlPar1Err, PDist1D, resultLabel, 
                       noXLabels = False, noYLabels = False):
    """Make a 1d contour plot of marginalised probability for a parameter. 
    
    par1Values      = values for parameter 1 (plotted on Y axis)
    par1Label       = text label for Y axis
    par1TickStep    = tick step along Y axis
    mlPar1          = maximum likelihood value for parameter 1
    mlPar1Err       = 1d 1-sigma error in parameter 1
    PDist1D         = 1d prob distribution for parameter 1
    
    """
    
    par1TickLabels=np.arange(par1Values.min(), par1Values.max(), par1TickStep)
    plt.xticks(par1TickLabels, par1TickLabels)
                
    PDist1D=PDist1D/PDist1D.max()
    ax=plt.gca()
    fitLabel='%s = %s' % (par1Label, resultLabel.replace("\pm", "$\pm$"))

    plt.plot(par1Values, PDist1D, 'k-', label = fitLabel)
    plt.ylabel("")
    plt.yticks([], [])
    #ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    plt.ylim(0, 1.2)
    leg=plt.legend(loc = (0.0, 0.86), prop = {'size': 12})
    leg.draw_frame(False)
    plt.draw()            
    plt.xlabel(par1Label)
    
    if noYLabels == True:
        ylocs, ylabels=plt.yticks()
        plt.ylabel("")
        plt.yticks(ylocs, [""]*len(ylabels))
    if noXLabels == True:
        xlocs, xlabels=plt.xticks()
        plt.xlabel("")
        plt.xticks(xlocs, [""]*len(xlabels))
        
#-------------------------------------------------------------------------------------------------------------
def probContourPlot_subPlot(par1Values, par2Values, par1Label, par2Label, par1TickStep, par2TickStep, mlPar1, mlPar2, 
                    mlPar1Err, mlPar2Err, PDist2D, noXLabels = False, noYLabels = False):
    """Make a 2d contour plot of probability surface of given parameters. Somewhat needless duplication of
    code, for makePaperContourPlots
    
    par1Values      = values for parameter 1 (plotted on Y axis)
    par2Values      = values for parameter 2 (plotted on X axis)
    par1Label       = text label for Y axis
    par2Label       = text label for X axis
    par1TickStep    = tick step along Y axis
    par2TickStep    = tick step along X axis
    mlPar1          = maximum likelihood value for parameter 1
    mlPar2          = maximum likelihood value for parameter 2
    mlPar1Err       = 1d 1-sigma error in parameter 1
    mlPar2Err       = 1d 1-sigma error in parameter 2
    PDist2D         = 2d likelihood surface, made using fast2DProbProjection
    
    """
    
    tck1=interpolate.splrep(par1Values, np.arange(par1Values.shape[0]))
    par1TickLabels=np.arange(par1Values.min(), par1Values.max(), par1TickStep)
    par1TickIndices=interpolate.splev(par1TickLabels, tck1)
    plt.yticks(par1TickIndices, par1TickLabels)
    
    tck2=interpolate.splrep(par2Values, np.arange(par2Values.shape[0]))
    par2TickLabels=np.arange(par2Values.min(), par2Values.max(), par2TickStep)
    par2TickIndices=interpolate.splev(par2TickLabels, tck2)
    plt.xticks(par2TickIndices, par2TickLabels)
            
    # We have to smooth to get decent looking contours
    # Gaussian smoothing preserves the normalisation
    # NOTE: smoothing only needed if very fine grid
    PDist2D=ndimage.gaussian_filter(PDist2D, 1)

    # Work out where to put contours
    sigma1Level=calc2DProbThreshold(PDist2D, 0.683)
    sigma2Level=calc2DProbThreshold(PDist2D, 0.95)
    
    plt.contour(PDist2D, [sigma1Level, sigma2Level], colors = 'k')
            
    # Save plot - trim down area first (?) and add axes labels
    plt.plot(interpolate.splev(mlPar2, tck2), interpolate.splev(mlPar1, tck1), 'k*', 
               label = "%s = %.2f $\pm$ %.2f, %s = %.2f $\pm$ %.2f" % (par1Label, mlPar1, mlPar1Err, par2Label, mlPar2, mlPar2Err))
    #plt.legend(numpoints = 1)
    plt.xlabel(par2Label)
    plt.ylabel(par1Label)
    
    if noYLabels == True:
        ylocs, ylabels=plt.yticks()
        plt.ylabel("")
        plt.yticks(ylocs, [""]*len(ylabels))
    if noXLabels == True:
        xlocs, xlabels=plt.xticks()
        plt.xlabel("")
        plt.xticks(xlocs, [""]*len(xlabels))

    
    
