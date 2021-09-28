# 2020/02/25~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
evaluation.py Evaluation Module

Methods for evaluating the models.

evaluate: evaluate a model
evaluateSingleNode: evaluate a model that has a single node forward
evaluateFlocking: evaluate a model using the flocking cost
evaluateLQR: evaluate a model using the LQR cost
"""

import os
import torch
import pickle
import numpy as np

def evaluate(model, data, **kwargs):
    """
    evaluate: evaluate a model using classification error
    
    Input:
        model (model class): class from Modules.model
        data (data class): a data class from the Utils.dataTools; it needs to
            have a getSamples method and an evaluate method.
        doPrint (optional, bool): if True prints results
    
    Output:
        evalVars (dict): 'errorBest' contains the error rate for the best
            model, and 'errorLast' contains the error rate for the last model
    """

    # Get the device we're working on
    device = model.device
    
    if 'doSaveVars' in kwargs.keys():
        doSaveVars = kwargs['doSaveVars']
    else:
        doSaveVars = True

    ########
    # DATA #
    ########

    xTest, yTest = data.getSamples('test')
    xTest = xTest.to(device)
    yTest = yTest.to(device)

    ##############
    # BEST MODEL #
    ##############

    model.load(label = 'Best')

    with torch.no_grad():
        # Process the samples
        yHatTest = model.archit(xTest)
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costBest = data.evaluate(yHatTest, yTest)

    ##############
    # LAST MODEL #
    ##############

    model.load(label = 'Last')

    with torch.no_grad():
        # Process the samples
        yHatTest = model.archit(xTest)
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costLast = data.evaluate(yHatTest, yTest)

    evalVars = {}
    evalVars['costBest'] = costBest.item()
    evalVars['costLast'] = costLast.item()
    
    if doSaveVars:
        saveDirVars = os.path.join(model.saveDir, 'evalVars')
        if not os.path.exists(saveDirVars):
            os.makedirs(saveDirVars)
        pathToFile = os.path.join(saveDirVars, model.name + 'evalVars.pkl')
        with open(pathToFile, 'wb') as evalVarsFile:
            pickle.dump(evalVars, evalVarsFile)

    return evalVars

def evaluateSingleNode(model, data, **kwargs):
    """
    evaluateSingleNode: evaluate a model that has a single node forward
    
    Input:
        model (model class): class from Modules.model, needs to have a 
            'singleNodeForward' method
        data (data class): a data class from the Utils.dataTools; it needs to
            have a getSamples method and an evaluate method and it also needs to
            have a 'getLabelID' method
        doPrint (optional, bool): if True prints results
    
    Output:
        evalVars (dict): 'errorBest' contains the error rate for the best
            model, and 'errorLast' contains the error rate for the last model
    """
    
    assert 'singleNodeForward' in dir(model.archit)
    assert 'getLabelID' in dir(data)

    # Get the device we're working on
    device = model.device
    
    if 'doSaveVars' in kwargs.keys():
        doSaveVars = kwargs['doSaveVars']
    else:
        doSaveVars = True

    ########
    # DATA #
    ########

    xTest, yTest = data.getSamples('test')
    xTest = xTest.to(device)
    yTest = yTest.to(device)
    targetIDs = data.getLabelID('test')

    ##############
    # BEST MODEL #
    ##############

    model.load(label = 'Best')

    with torch.no_grad():
        # Process the samples
        yHatTest = model.archit.singleNodeForward(xTest, targetIDs)
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costBest = data.evaluate(yHatTest, yTest)

    ##############
    # LAST MODEL #
    ##############

    model.load(label = 'Last')

    with torch.no_grad():
        # Process the samples
        yHatTest = model.archit.singleNodeForward(xTest, targetIDs)
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costLast = data.evaluate(yHatTest, yTest)

    evalVars = {}
    evalVars['costBest'] = costBest.item()
    evalVars['costLast'] = costLast.item()
    
    if doSaveVars:
        saveDirVars = os.path.join(model.saveDir, 'evalVars')
        if not os.path.exists(saveDirVars):
            os.makedirs(saveDirVars)
        pathToFile = os.path.join(saveDirVars, model.name + 'evalVars.pkl')
        with open(pathToFile, 'wb') as evalVarsFile:
            pickle.dump(evalVars, evalVarsFile)

    return evalVars

def evaluateFlocking(model, data, **kwargs):
    """
    evaluateClassif: evaluate a model using the flocking cost of velocity 
        variacne of the team
    
    Input:
        model (model class): class from Modules.model
        data (data class): the data class that generates the flocking data
        doPrint (optional; bool, default: True): if True prints results
        nVideos (optional; int, default: 3): number of videos to save
        graphNo (optional): identify the run with a number
        realizationNo (optional): identify the run with another number
    
    Output:
        evalVars (dict):
            'costBestFull': cost of the best model over the full trajectory
            'costBestEnd': cost of the best model at the end of the trajectory
            'costLastFull': cost of the last model over the full trajectory
            'costLastEnd': cost of the last model at the end of the trajectory
    """
    
    if 'doPrint' in kwargs.keys():
        doPrint = kwargs['doPrint']
    else:
        doPrint = True
        
    if 'nVideos' in kwargs.keys():
        nVideos = kwargs['nVideos']
    else:
        nVideos = 3
        
    if 'graphNo' in kwargs.keys():
        graphNo = kwargs['graphNo']
    else:
        graphNo = -1

    if 'realizationNo' in kwargs.keys():
        if 'graphNo' in kwargs.keys():
            realizationNo = kwargs['realizationNo']
        else:
            graphNo = kwargs['realizationNo']
            realizationNo = -1
    else:
        realizationNo = -1

    #\\\\\\\\\\\\\\\\\\\\
    #\\\ TRAJECTORIES \\\
    #\\\\\\\\\\\\\\\\\\\\

    ########
    # DATA #
    ########

    # Initial data
    initPosTest = data.getData('initPos', 'test')
    initVelTest = data.getData('initVel', 'test')

    ##############
    # BEST MODEL #
    ##############

    model.load(label = 'Best')

    if doPrint:
        print("\tComputing learned trajectory for best model...",
              end = ' ', flush = True)

    posTestBest, \
    velTestBest, \
    accelTestBest, \
    stateTestBest, \
    commGraphTestBest = \
        data.computeTrajectory(initPosTest, initVelTest, data.duration,
                               archit = model.archit)

    if doPrint:
        print("OK")

    ##############
    # LAST MODEL #
    ##############

    model.load(label = 'Last')

    if doPrint:
        print("\tComputing learned trajectory for last model...",
              end = ' ', flush = True)

    posTestLast, \
    velTestLast, \
    accelTestLast, \
    stateTestLast, \
    commGraphTestLast = \
        data.computeTrajectory(initPosTest, initVelTest, data.duration,
                               archit = model.archit)

    if doPrint:
        print("OK")

    ###########
    # PREVIEW #
    ###########

    learnedTrajectoriesDir = os.path.join(model.saveDir,
                                          'learnedTrajectories')
    
    if not os.path.exists(learnedTrajectoriesDir):
        os.mkdir(learnedTrajectoriesDir)
    
    if graphNo > -1:
        learnedTrajectoriesDir = os.path.join(learnedTrajectoriesDir,
                                              '%03d' % graphNo)
        if not os.path.exists(learnedTrajectoriesDir):
            os.mkdir(learnedTrajectoriesDir)
    if realizationNo > -1:
        learnedTrajectoriesDir = os.path.join(learnedTrajectoriesDir,
                                              '%03d' % realizationNo)
        if not os.path.exists(learnedTrajectoriesDir):
            os.mkdir(learnedTrajectoriesDir)

    learnedTrajectoriesDir = os.path.join(learnedTrajectoriesDir, model.name)

    if not os.path.exists(learnedTrajectoriesDir):
        os.mkdir(learnedTrajectoriesDir)

    if doPrint:
        print("\tPreview data...",
              end = ' ', flush = True)

    data.saveVideo(os.path.join(learnedTrajectoriesDir,'Best'),
                   posTestBest,
                   nVideos,
                   commGraph = commGraphTestBest,
                   vel = velTestBest,
                   videoSpeed = 0.5,
                   doPrint = False)

    data.saveVideo(os.path.join(learnedTrajectoriesDir,'Last'),
                   posTestLast,
                   nVideos,
                   commGraph = commGraphTestLast,
                   vel = velTestLast,
                   videoSpeed = 0.5,
                   doPrint = False)

    if doPrint:
        print("OK", flush = True)

    #\\\\\\\\\\\\\\\\\\
    #\\\ EVALUATION \\\
    #\\\\\\\\\\\\\\\\\\
        
    evalVars = {}
    evalVars['costBestFull'] = data.evaluate(vel = velTestBest)
    evalVars['costBestEnd'] = data.evaluate(vel = velTestBest[:,-1:,:,:])
    evalVars['costLastFull'] = data.evaluate(vel = velTestLast)
    evalVars['costLastEnd'] = data.evaluate(vel = velTestLast[:,-1:,:,:])

    return evalVars

def evaluateLQR(model, data, **kwargs):
    """
    evaluateLQR: evaluate a model using the LQR cost
    
    Input:
        model (model class): class from Modules.model
        data (data class): a data class from the Utils.dataTools; it needs to
            have a getSamples method and an evaluate method. It also needs to
            have defined Q and R for the LQR cost.
        doSaveVars (optional, bool, default: True): save the results of the
            evaluation in pickle format
    
    Output:
        evalVars (dict): 'costBest' contains the cost for the best model, and 
        'costLast' contains the cost for the last model. Also, 'costBestEnd'
        contains the last value of the control cost, x^T Q x to know whether
        a certain degree of control has been achieved. Likewise, for 
        'costLastEnd'.
    """

    # Get the device we're working on
    device = model.device
    
    # Check if we want to save vars
    if 'doSaveVars' in kwargs.keys():
        doSaveVars = kwargs['doSaveVars']
    else:
        doSaveVars = True
        
    # Check if we want to normalize the output
    if 'normalize' in kwargs.keys():
        normalize = kwargs['normalize']
        if normalize:
            if 'lowerBound' in kwargs.keys():
                lowerBound = kwargs['lowerBound']
        else: # If the lower bound was not specified, don't normalize
            normalize = False
    else:
        normalize = False
        
    if normalize:
        reduce = False
    else:
        reduce = True

    ########
    # DATA #
    ########

    # Get the data
    xTest, uTest = data.getSamples('test')
    # Move it to the corresponding device
    xTest = xTest.to(device)
    uTest = uTest.to(device)
    
    if 'torch' in repr(data.Q.dtype):
        Q = data.Q.to(device)
    else:
        Q = torch.tensor(data.Q).to(device)

    ##############
    # BEST MODEL #
    ##############

    model.load(label = 'Best')

    # Since this is the evaluation, we do not need a gradient of the model
    with torch.no_grad():
        # Compute the trajectory
        xHatTest, uHatTest = data.computeTrajectory(
                                                 x0 = xTest[:,0,:,:].squeeze(1),
                                                 controller = model.archit)

        # Compute the accuracy for the resulting trajectory
        costBest = data.evaluate(xHatTest, uHatTest, reduce = reduce)
        # Compute the end cost: x^T Q x
        xT = xHatTest[:,-1,:].unsqueeze(1) # nSamples x 1 x nNodes
        xTQ = torch.matmul(xT, Q) # nSamples x 1 x nNodes
        xTQxT = torch.matmul(xTQ, xT.permute(0,2,1)) # nSamples x 1 x 1
        costBestEnd = xTQxT.squeeze(2).squeeze(1) # nSamples

    ##############
    # LAST MODEL #
    ##############

    model.load(label = 'Last')

    with torch.no_grad():
        xHatTest, uHatTest = data.computeTrajectory(
                                                 x0 = xTest[:,0,:,:].squeeze(1),
                                                 controller = model.archit)

        costLast = data.evaluate(xHatTest, uHatTest, reduce = reduce)
        
        # Compute the end cost: x^T Q x
        xT = xHatTest[:,-1,:].unsqueeze(1) # nSamples x 1 x nNodes
        xTQ = torch.matmul(xT, Q) # nSamples x 1 x nNodes
        xTQxT = torch.matmul(xTQ, xT.permute(0,2,1)) # nSamples x 1 x 1
        costLastEnd = xTQxT.squeeze(2).squeeze(1) # nSamples
        
    if normalize:
        costBest = np.mean(costBest.cpu().numpy()/lowerBound)
        costLast = np.mean(costLast.cpu().numpy()/lowerBound)
    else:
        costBest = costBest.item()
        costLast = costLast.item()

    evalVars = {}
    evalVars['costBest'] = costBest
    evalVars['costLast'] = costLast
    evalVars['costBestEnd'] = costBestEnd.cpu().numpy()
    evalVars['costLastEnd'] = costLastEnd.cpu().numpy()
    
    if doSaveVars:
        saveDirVars = os.path.join(model.saveDir, 'evalVars')
        if not os.path.exists(saveDirVars):
            os.makedirs(saveDirVars)
        pathToFile = os.path.join(saveDirVars, model.name + 'evalVars.pkl')
        with open(pathToFile, 'wb') as evalVarsFile:
            pickle.dump(evalVars, evalVarsFile)

    return evalVars