# 2018/10/15~
# Fernando Gama, fgama@seas.upenn.edu.
# Luana Ruiz, rubruiz@seas.upenn.edu.
"""
miscTools Miscellaneous Tools module

num2filename: change a numerical value into a string usable as a filename
saveSeed: save the random state of generators
loadSeed: load the number of random state of generators
writeVarValues: write the specified values in the specified txt file
"""

import os
import pickle
import numpy as np
import torch

def num2filename(x,d):
    """
    Takes a number and returns a string with the value of the number, but in a
    format that is writable into a filename.

    s = num2filename(x,d) Gets rid of decimal points which are usually
        inconvenient to have in a filename.
        If the number x is an integer, then s = str(int(x)).
        If the number x is a decimal number, then it replaces the '.' by the
        character specified by d. Setting d = '' erases the decimal point,
        setting d = '.' simply returns a string with the exact same number.

    Example:
        >> num2filename(2,'d')
        >> '2'

        >> num2filename(3.1415,'d')
        >> '3d1415'

        >> num2filename(3.1415,'')
        >> '31415'

        >> num2filename(3.1415,'.')
        >> '3.1415'
    """
    if x == int(x):
        return str(int(x))
    else:
        return str(x).replace('.',d)

def saveSeed(saveDir):
    """
    Saves the generator states of numpy and torch.
    
    Inputs:
        saveDir (path): where to save the seed, it will be saved under the 
            filenames 'randomTorchSeedUsed.pkl' and 'randomNumpySeedUsed.pkl'.
    
    Obs.: In the case of torch, it saves the 'torchState' of the RNG sate, and
    'torchSeed' of the initial seed used.
    """
    torchState = torch.get_rng_state()
    torchSeed = torch.initial_seed()
    pathToSeed = os.path.join(saveDir, 'randomTorchSeedUsed.pkl')
    with open(pathToSeed, 'wb') as seedFile:
        torch.save({'torchState': torchState,
                    'torchSeed': torchSeed},
                   pathToSeed)
    #   Numpy seeds
    numpyState = np.random.RandomState().get_state()
    
    pathToSeed = os.path.join(saveDir, 'randomNumpySeedUsed.pkl')
    with open(pathToSeed, 'wb') as seedFile:
        pickle.dump({'numpyState': numpyState}, seedFile)
        
def loadSeed(loadDir):
    """
    Loads the states and seed saved in a specified path
    
    Inputs:
        loadDir (path): where to look for thee seed to load; it is expected that
            the appropriate files within loadDir are named 
            'randomTorchSeedUsed.pkl' for the torch seed, and
            'randomNumpySeedUsed.pkl' for the numpy seed.
    
    Obs.: The file 'randomTorchSeedUsed.pkl' has to have two variables: 
        'torchState' with the RNG state, and 'torchSeed' with the initial seed
        The file 'randomNumpySeedUsed.pkl' has to have a variable 'numpyState'
        with the Numpy RNG state
    """
    #\\\ Torch
    pathToSeed = os.path.join(loadDir, 'randomTorchSeedUsed.pkl')
    with open(pathToSeed, 'rb') as seedFile:
        torchRandom = torch.load(seedFile)
        torchState = torchRandom['torchState']
        torchSeed = torchRandom['torchSeed']
    
    torch.set_rng_state(torchState)
    torch.manual_seed(torchSeed)
    
    #\\\ Numpy
    pathToSeed = os.path.join(loadDir, 'randomNumpySeedUsed.pkl')
    with open(pathToSeed, 'rb') as seedFile:
        numpyRandom = pickle.load(seedFile)
        numpyState = numpyRandom['numpyState']
    
    np.random.RandomState().set_state(numpyState)                

def writeVarValues(fileToWrite, varValues):
    """
    Write the value of several string variables specified by a dictionary into
    the designated .txt file.
    
    Input:
        fileToWrite (os.path): text file to save the specified variables
        varValues (dictionary): values to save in the text file. They are
            saved in the format "key = value".
    """
    with open(fileToWrite, 'a+') as file:
        for key in varValues.keys():
            file.write('%s = %s\n' % (key, varValues[key]))
        file.write('\n')
