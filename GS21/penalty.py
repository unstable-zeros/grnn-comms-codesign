# 2019/07/22~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
penalty.py Penalty functions

spectralNorm: compute the spectral norm of a matrix in a way that it keeps track
    of the gradient

getFilterTaps: Given a graph convolutional neural network architecture, get the
    matrices that form the filter taps

getFrequencyResponse: Compute the family of graph frequency responses for a
    given architecture.
getGraphFilters: Compute the family of graph filters for a given architecture.

computeFilterSizeMatrix: computes the matrix of the filter size
computeFilterSize: computes the size of a bank of filters

GNNsize: penalty function that computes the size of a given GNN
L2stabilityConstant: penalty function that computes the stability constant
"""
import numpy as np
import torch

# All penalty functions need to have two inputs: the model class, and the data
# class, irrespective whether they are used later or not. This will make
# the call of the penalty function generic during training, so it is always 
# called that way, irrespective of the actual penalty function.

def spectralNorm(A, method = 'fast', nRand = 1000, maxIt = 100, tol = 1e-2):
    """
    spectralNorm: compute the spectral norm of A in a way that it keeps track
        of the gradient
    
    Note: When we created this function, pytorch didn't compute the spectral 
    norm, and only computed eigenvalues from symmetric matrices, so we needed a
    'differentiable' way of computing the spectral norm.
    
    Input:
        A (torch.tensor, shape: * x N x N): set of square matrices to which
            compute the spectral norm
    
    Output:
        spectralNorm (torch.tensor, shape: *): the spectral norm of each
            of the matrices in the set
    
    Optional arguments:
        method ('bound, 'fast', 'exact', default: 'fast'): Choose the method
            to use when computing the spectral norm.
            'bound': Uses Hölder's inequality to compute
                    \|A\|_{2} \leq \sqrt(\|A\|_{1} \|A\|_{2})
                this overestimates \|A\|_{2} but is fast since the one and 
                infinity norm of the matrices can be easily computed
            'fast': Computes the spectral norm by using the power method for a
                number maxIt of iterations.
            'exact': Computes the actual spectral norm of the matrix by using 
                numpy and then uses the power method to approximate to that
                value; the iterations go on until the norm is within tol of
                the true value or when the number maxIt of iterations has been
                reached.
            nRand (int, default: 1000): Number of different initializations of
                the vector for the power iteration method.
            maxIt (int, default: 100): Maximum number of iterations for the 
                iteration method
            tol (float, default: 1e-2): tolerance for the exact method
            
    Observations:
        - If the initial random vector u lies on a subspace that doesn't contain
          v_max (the eigenvector associated to the maximum absolute eigenvalue)
          then the method won't work.
        - By selecting nRand > 1 we're generating many initial vectors to try
          to avoid the problem of lying in the wrong subspace; then the output
          is computed as the median from the spectral norms estimated with each
          of the random initializations, so that if one of those random 
          initializations does not lie in the subspace, it is simply ignored
        - By selecting nRand > 1, we're adding the gradient of torch.diagonal
          and torch.median to the mix. These gradients are avoided if nRand = 1.
        - Observations from a run of 100 different structures matrices, all with
          spectral norm 0.995, under the default conditions.
          - Failure: Computing the bound always overestimates the spectral norm,
            the 'fast' method and the 'exact' method failed in 10% of the runs,
            always on the same matrix A (this suggests that some matrices are
            not well suited for this method).
          - Failure: The 'bound' method outputs a norm of 2.9 (+- 0.4) quite a
            lot more than the spectral norm of 0.995, so it is not a very tight
            bound. The 'fast' method and the 'exact'' method, when they fail, 
            output a norm of 0.84 (+- 0.09) so they underestimate the true bound
            by a little.
          - Time: The 'bound' method takes 0.18 (+- 0.02) milliseconds to 
            compute, the 'fast' method takes 27 (+- 2) milliseconds to compute,
            and the 'exact' method takes 61 (+- 57) milliseconds to compute.
            Note that the 'exact' method has a lot of variability, this is
            because, when it fails, it takes a very long time (it completes
            all iterations). This can be seen that, the average time when it
            computes the exact bound, it takes 44 (+- 33) milliseconds, and when
            it fails, it takes 204 (+- 8) milliseconds.
        - Minimizing the bound is useful for minimizing the actual spectral norm?
    """
    
    # Check that the method selected is a valid method
    assert method == 'bound' or method == 'fast' or method == 'exact'
    
    # Assume A is of * x N x N shape
    N = A.shape[-1]
    assert A.shape[-2] == N
    device = A.device # Get device
    
    if method == 'bound':
        # For the bound method, we just use Hölder's inequality
        #   \| A \|_{2} \leq \sqrt{ \| A \|_{1} \| A \|_{\infty}}
        #
        #   \| A \|_{1} = \max_{1 \leq j \leq N} \sum_{i=1}^{N} |a_{ij}|
        Aone = torch.max(torch.sum(torch.abs(A), dim = -2), dim = -1)
        #   \| A \|_{\infty} = \max_{1 \leq i \leq N} \sum_{j=1}^{N} |a_{ij}|
        Ainfty = torch.max(torch.sum(torch.abs(A), dim = -1), dim = -1)
        
        Anorm = np.sqrt(Aone[0] * Ainfty[0])
    
    elif method == 'fast':
        # Create the random vector to initialize the power method
        u = torch.randn(A.shape[0:-2] + (nRand, N)).to(device)
        # Iterate a number maxIt of times multiplying by A and normalizing
        for it in range(maxIt):
            u = torch.matmul(u, A) # * x nRand x N
            uNorm = torch.norm(u, dim = -1).unsqueeze(-1) # * x nRand x 1
            u = u/uNorm # * x nRand x N
        # After all the iterations, get the eigenvalue
        #   This is given by \| A_{2} \| = |u^T A u|/(u^T u) = |u^T A u|
        # since u is being normalized every time.
        Au = torch.matmul(u, A) # * x nRand x N
        # Obtain the remaining dimensions * to be able to permute u
        sameDims = tuple(range(len(u.shape[0:-2])))
        # If we have more than one value of u
        if nRand > 1:
            # First, compute u^T A u, remember we need to transpose u
            uAu = torch.matmul(Au, u.permute(sameDims + (-1,-2))) 
            #   This is of shape * x nRand x nRand, but out of this, we only
            # need the diagonal elements:
            uAu = torch.diagonal(uAu, dim1 = -2, dim2 = -1) # * x nRand
            # Now compute the absolute value
            Anorm = torch.abs(uAu) # * x nRand
            # And compute the median: since there could be some values of u that
            # could have lied in a useless subspace, the norm for those values
            # will be very bad. We suggest not using the mean, since these can
            # be considered "outliers" and would really derail the value. By 
            # using the median, we're hoping that most of the values of u
            # didn't lie in the shitty subspace, so we would get the actual 
            # value. The problem: the gradient of torch.median could be shitty.
            # Also, the torch.diagonal could be sketchy.
            Anorm = torch.median(Anorm, dim = -1)[0] # * 
        else:
            # So if we selected only one random input, we run the risk that
            # it could be on the shitty subspace. But we avoid the 
            # torch.diagonal and torch.median gradients.
            uAu = torch.matmul(Au, u.permute(sameDims + (-1,-2))) # * x 1 x 1
            # Get rid of the unncessary dimensions
            uAu = uAu.squeeze(-1).squeeze(-1) # *
            Anorm = torch.abs(uAu) # Store current norm
            # If the norm hasn't changed much in one iteration or if we exceeded
            # a given number of iterations, sample a new vector u
    elif method == 'exact':
        # In this method, we are going to compute the actual spectral norm of
        # the matrix using numpy
        targetAnorm = np.linalg.norm(A.detach().cpu().numpy(), ord = 2,
                                  axis = (-2,-1))
        # We convert this value to torch.tensor to be able to compare it
        targetAnorm = torch.tensor(targetAnorm).to(device)
        
        # At each iteration, we're going to compare the norm obtained with the
        # target one, and if it's not similar, we iterate again, and so on.
        
        # This method is clearly slower: it needs to compute the norm, which is
        # expensive, and it adds a comparison at every single iteration.
        
        # Create the initial A2 which is going to be zero (so that it's 
        # different enough from the actual spectral norm so that it
        # goes into the while)
        Anorm = torch.zeros(targetAnorm.shape).to(device)
        # Create the random vector
        u = torch.randn(A.shape[0:-2] + (nRand, N)).to(device)
        # Count the number of iterations
        totalIts = 0.
        # And start iterating (while we are far from the target A2 and we
        # haven't reached the maximum number of iterations)
        while torch.mean(torch.abs(targetAnorm - Anorm)) > tol \
                and totalIts <= maxIt:
            # Compute Au
            u = torch.matmul(u, A) # * x nRand x N
            uNorm = torch.norm(u, dim = -1).unsqueeze(-1) # * x nRand x 1
            u = u/uNorm # * x nRand x N
            # And compute the eigenvalue (step not needed in the fast method)
            #   Compute Au
            Au = torch.matmul(u, A) # * x nRand x N
            #   Obtain the same dimensions * to be able to permute u
            sameDims = tuple(range(len(u.shape[0:-2])))
            #   Case when we have many random samples (this case is separate because
            #   it uses torch.median and torch.diagonal which could have shady 
            #   gradients)
            if nRand > 1:
                # For the explanation, see the fast method
                uAu = torch.matmul(Au, u.permute(sameDims + (-1,-2))) 
                uAu = torch.diagonal(uAu, dim1 = -2, dim2 = -1)
                Anorm = torch.abs(uAu)
                Anorm = torch.median(Anorm, dim = -1)[0]
            else:
                uAu = torch.matmul(Au, u.permute(sameDims + (-1,-2)))
                uAu = uAu.squeeze(-1).squeeze(-1)
                Anorm = torch.abs(uAu)
            # Add the iteration
            totalIts += 1

    return Anorm

def getFilterTaps(archit):
    """
    getFilterTaps: Given a graph convolutional neural network architecture, 
        get the matrices H_{\ell k} that form the filter taps
        
    Input:
        architecture (nn.Module): The nn.Module containing gml.GraphConv 
            layers, or similar. The architecture needs to have the following
            attributes: .L is an int consisting of the number of graph 
            convolutiona layers, .F is a list of (.L+1) elements containing
            the number of input features to each layer, and where the last
            element is the number of output features, .K is a list of .L 
            elements containing the number of filter taps in each layer, and
            .E is an int representing the number of features.
            Furthermore, if the architecture has a series of readout layers
            that act as local combinations (i.e. filters with K_{\ell}=1, that
            don't exchange information with neighbors), then these are counted
            as filters as well. To count this filters, the architecture needs
            to have an attribute .dimReadout that is a list containing the
            number of layers consisting of these last filters of K_{\ell} = 1
    
    Output:
        H (list): Each element in the list is a torch.tensor containing the
            filer taps from the architecture (torch.tensor with requires_grad).
            The list has L elements, where L is the number of layers, and every
            element is a torch.tensor of shape
                F_{\ell} x E x K_{\ell} x F_{\ell-1}
            where F_{\ell} is the number of output features, E is the number of
            edge features, K_{\ell} is the number of filter taps, and F_{\ell-1}
            is the number of input features. Note that length of the list will
            be .L+len(.dimReadout).
    """
    
    # Get the details of the architecture
    L = archit.L # Number of (graph filtering) layers
    F = archit.F # Number of filter taps per layer (list of length L+1, first
    #   element is the input features)
    K = archit.K # Number of filter taps per layer (list of length L)
    E = archit.E # Number of edge features (float)
    # The expected filter taps will be like
    filterTapsShape = []
    for l in range(L):
        filterTapsShape.append([F[l+1], E, K[l], F[l]])
    #   Now if the architecture has a 'dimReadout' then these are filters of
    #   just one coefficient and we should include them. If it says 
    #   'dimLayersMLP' then it's an MLP and all structure is gone
    readoutTapsShape = [] # Save the shapes of the expected tensors
    if 'dimReadout' in dir(archit):
        dimReadout = archit.dimReadout # dimReadout has to be a list
        # If it actually has elements
        if len(dimReadout) > 0:
            readoutTapsShape.append([dimReadout[0], F[-1]]) # First one
            for l in range(len(dimReadout)-1): # The rest
                readoutTapsShape.append([dimReadout[l+1], dimReadout[l]])
    
    # Collect the filter taps
    H = []
    readoutTaps = [] # This is just to collect the readout taps (if any), before
        # transforming them into the appropriate shape an adding them to the 
        # list in H; we want to keep separate track of them to be sure we got
        # everything we need
    for param in archit.parameters():
        #   So, here we have three possible options: it's either a graph filter
        #   from the filtering layers, or it's the bias, or it's a graph filter
        #   from the readout layer.
        #   A graph filter from the graph filtering layer has shape
        #       G (=F_{\ell}) x E (=1) x K (=K_{\ell}) x F (=F_{\ell-1})
        #   A bias has shape
        #       G (=F_{\ell}) x 1
        #   A graph filter from the readout layer has shape
        #       G (=F_{\ell}) x F (=F_{\ell-1})            
        if len(param.shape) == 4: # If it has shape four, we need to check that
            # the sizes are aligned
            if list(param.shape) in filterTapsShape:
                # If the parameters has one of the expected shapes
                H += [param] # Save the filter tap
        elif len(param.shape) == 2:
            # This has to be the readout layer
            if list(param.shape) in readoutTapsShape:
                readoutTaps += [param]
    # Now that we have collected all the parameters, let  us check that we
    # did it properly
    assert len(H) == len(filterTapsShape)
    assert len(readoutTaps) == len(readoutTapsShape)
    # Finally, let us reshape the readoutTaps so they can fit the format in H
    # and add them to H
    if len(readoutTaps) > 0:
        for thisReadoutTaps in readoutTaps:
            # It has shape out_features x in_features, but we want it to be
            # of shape out_features x E x 1 x in_features, so
            thisReadoutTaps = thisReadoutTaps.unsqueeze(2).unsqueeze(3)
            #   Shape: out_features x in_features x 1 x 1
            thisReadoutTaps = thisReadoutTaps.repeat(1, 1, E, 1)
            #   Shape: out_features x in_features x E x 1
            thisReadoutTaps = thisReadoutTaps.permute(0, 2, 3, 1)
            # Add it to H
            H += [thisReadoutTaps]
    
    return H

def getFrequencyResponse(*args,
                         useGSO = False,
                         lowEgvl = -1.,
                         highEgvl = 1.,
                         nEigs = 200):
    """
    getFrequencyResponse: Compute the family of graph frequency responses for a
        given architecture or some given filter taps.
        
    Input:
        (One argument)
        architecture (nn.Module): The nn.Module containing gml.GraphConv 
            layers, or similar. The architecture needs to have the following
            attributes: .S containing the graph shift operator, and .L, .F,
            .K, and .E as determined by the function getFilterTaps.
        -- OR --
        (One or two arguments)
        filterTaps (list): Each element in the list is a torch.tensor containing
            the filer taps from the architecture (torch.tensor with 
            requires_grad).
            The list has L elements, where L is the number of layers, and every
            element is a torch.tensor of shape
                F_{\ell} x E x K_{\ell} x F_{\ell-1}
            where F_{\ell} is the number of output features, E is the number of
            edge features, K_{\ell} is the number of filter taps, and F_{\ell-1}
            is the number of input features. Note that length of the list will
            be .L+len(.dimReadout).
        GSO (torch.tensor, required if useGSO = True): Graph shift operator,
            shape 1 x N x N. This input is not required if useGSO = False and 
            will be ignored if it is there.
    Options:
        useGSO (bool, default: False): Uses the GSO provided in architecture.S
            to compute the actual eigenvalues (computes an eigendecomposition,
            this is an expensive operation)
        lowEgvl (float, default: -1): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
        highEgvl (float, default: 1): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
        nEigs (int, default: 200): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
            
    Output:
        h (list): Each element in the list is a torch.tensor containing the
            frequency responses, i.e., the polynomials on the variable lambda,
            from the architecture (torch.tensor with requires_grad).
            The list has L elements, where L is the number of layers, and every
            element is a torch.tensor of shape
                E x F_{\ell} x F_{\ell-1} x nEigs
            where E is the number of edge features, F_{\ell} is the number of 
            output features, F_{\ell-1} is the number of input features, and
            nEigs is the number of values in the lambda l ine. Note that length
            of the list will be .L+len(.dimReadout).
            
    Observations:
        - S cannot be used as an optimization variable, i.e. S cannot have a 
          gradient.
        - We set [-1,1] which is the maximum range for normalized adjacency, if
          the normalized Laplacian is used (normalized by the largest eigenvalue)
          then use [0,1].
    """
    # Check that we have only one or two input arguments
    assert len(args) == 1  or len(args) == 2
    
    # If it's only one input argument
    if len(args) == 1:
        # This can be the architecture, or the filter taps right away
        if 'GraphFilter' in repr(args) or 'dimReadout' in repr(args):
            # Check if it is a graph convolutional architecture (i.e. has the 
            # GraphFilter layer)
            archit = args[0]
            # Now check we have a GSO
            if archit.S is None:
                # If we don't have a GSO, we cannot use it
                useGSO = False
            elif useGSO: # If we have an S and we are going to use it
                # Get S
                S = archit.S
                # Get the number of edge features
                E = S.shape[0]
                # Check that there's only one edge feature (right now the definition is
                # only for a single edge feature)
                assert E == 1
            # Now get the filter taps
            H = getFilterTaps(archit) # H[l]: F_out x E x K x F_in
        else:
            # If there's only one input argument and it's not the architecture,
            # then it has to be the filter taps, and there shouldn't be a 
            # requirement for a GSO specificed (because if useGSO = True, then
            # the second argument would be the GSO)
            assert useGSO == False
            # If it's not the architecture, then it can be the filter taps
            H = args[0]
            assert type(H) is list
    else:
        # If we have two input arguments
        assert type(args[0]) is list and len(args[1].shape) == 3
        # The first one has to be filter taps
        H = args[0]
        # The second one has to be the GSO, but we only care if we are gonig to
        # use it
        if useGSO:
            S = args[1]
            # Check it has only one edge features
            E = S.shape[0]
            assert E == 1
            # Get the number of nodes
            N = S.shape[1]
            # And check that it's square
            assert S.shape[2] == N
        
    # Now that we have collected the filter taps in H and the GSO in S we can
    # compute the eigenvalue or the sampling of the real line
    if useGSO:
        # Compute the eigendecomposition
        egvl = np.linalg.eigvals(S.detach().cpu().numpy())
        # Sort it from smallest to largest
        egvl = np.sort(egvl.squeeze())
        # Update the number of eigenvalues nEigs
        lowEgvl = np.min(egvl)
        highEgvl = np.max(egvl)
        nEigs = egvl.shape[0]
    else:
        # And if we don't have a GSO, so we need to create the sampling of the
        # eigenvalue real line
        egvl = np.linspace(lowEgvl, highEgvl, num = nEigs)
        
    # Now that we have the filter taps, we need to obtain the corresponding 
    # information
    L = len(H) # Number of layers
    # Get the device
    device = H[0].device
    # Get the parameters
    K = [] # Number of filter taps
    F = [H[0].shape[3]] # Number of features
    for l in range(L):
        # H[l] is of shape out x E x K x in
        K += [H[l].shape[2]]
        F += [H[l].shape[0]]
    # And the maximum number of filter taps
    maxK = max(K)
        
    # Now we need to compute the Vandermonde Matrix from nEigs to K x nEigs
    #      1            &      1            & \ldots  & 1
    # \lambda_{1}       & \lambda_{2}       & \ldots  & \lambda_{nEigs}
    #   \vdots          &    \vdots         & \ddots  &    \vdots
    # \lambda_{1}^{K-1} & \lambda_{2}^{K-1} & \ldots  & \lambda_{nEigs}^{K-1}
    # Save space for the Vandermonde matrix V
    V = np.zeros((maxK, nEigs))
    V[0,:] = np.ones((1, nEigs)) # k = 0
    # Each row is the multiplication of the previous row by egvl
    for k in range(1,maxK):
        V[k,:] = egvl * V[k-1,:]
    # Move it to a tensor
    V = torch.tensor(V, device = device)
    # And add the two dimensions corresponding to E and F_out so that the
    # multiplication will become (F_in x K) by (K x nEigs)
    V = V.reshape(1, 1, maxK, nEigs)
    
    # And finally we can compute the frequency response
    h = [] # Where to store the frequency response
    # Output a matrix of shape E x F_out x F_in x nEigs
    for l in range(L):
        h += [torch.matmul(H[l].permute(1, 0, 3, 2), V[:,:,0:K[l],:])] # K x nEigs
    
    return h # h[l] E x F_out x F_in x nEigs

def getGraphFilters(*args):
    """
    getGraphFilters: Compute the family of graph filters for a given
        architecture.
    
    Input:
        (One argument)
        architecture (nn.Module): The nn.Module containing gml.GraphConv 
            layers, or similar. The architecture needs to have the following
            attributes: .S containing the graph shift operator, and .L, .F,
            .K, and .E as determined by the function getFilterTaps.
        -- OR --
        (Two arguments)
        filterTaps (list): Each element in the list is a torch.tensor containing
            the filer taps from the architecture (torch.tensor with 
            requires_grad).
            The list has L elements, where L is the number of layers, and every
            element is a torch.tensor of shape
                F_{\ell} x E x K_{\ell} x F_{\ell-1}
            where F_{\ell} is the number of output features, E is the number of
            edge features, K_{\ell} is the number of filter taps, and F_{\ell-1}
            is the number of input features. Note that length of the list will
            be .L+len(.dimReadout).
        GSO (torch.tensor): Graph shift operator, shape 1 x N x N.
    
    Output:
        HS (list): Each element in the list is a torch.tensor containing the
            graph filters, i.e., the polynomials on S, from the architecture 
            (torch.tensor with requires_grad).
            The list has L elements, where L is the number of layers, and every
            element is a torch.tensor of shape
                E x F_{\ell} x F_{\ell-1} x N x N
            where E is the number of edge features, F_{\ell} is the number of 
            output features, F_{\ell-1} is the number of input features, and
            N is the number of nodes. Note that length of the list will be 
            .L+len(.dimReadout).
    """

    # Check that we have only one or two input arguments
    assert len(args) == 1  or len(args) == 2
    
    # If it's only one input argument
    if len(args) == 1:
        # It has to be an architecture with the layer GraphFilter
        assert 'GraphFilter' in repr(args) or 'dimReadout' in repr(args)
        # Save it
        archit = args[0]
        # Get S
        S = archit.S
        # Check that S exists
        assert S is not None
        # Get the number of edge features
        E = S.shape[0]
        # Check that there's only one edge feature (right now the definition is
        # only for a single edge feature)
        assert E == 1
        # Get the number of nodes
        N = S.shape[1]
        
        # Let us first get the filter taps
        H = getFilterTaps(archit)
    else:
        # If we have two input arguments
        assert type(args[0]) is list and len(args[1].shape) == 3
        # The first one has to be filter taps
        H = args[0]
        # The second one has to be the GSO
        S = args[1]
        # Check it has only one edge features
        E = S.shape[0]
        assert E == 1
        # Get the number of nodes
        N = S.shape[1]
        # And check that it's square
        assert S.shape[2] == N
    
    L = len(H) # Number of layers
    HS = [] # Where to store the polynomials
    # Get the device
    device = H[0].device
    # Get the parameters
    K = [] # Number of filter taps
    F = [H[0].shape[3]] # Number of features
    for l in range(L):
        # H[l] is of shape out x E x K x in
        K += [H[l].shape[2]]
        F += [H[l].shape[0]]

    # Let's start by building E x K x N x N
    # First, the identity
    SK = torch.eye(N).reshape(E, 1, N, N).to(device) # Cumulative S^k
    thisSk = torch.eye(N).reshape(E, N, N).to(device) # Latest S^k
    # Then, multiply by S, each time
    for k in range(1,max(K)):
        thisSk = torch.matmul(S, thisSk) # E x N x N
        SK = torch.cat((SK, thisSk.unsqueeze(1)), dim = 1) # E x k x N x N

    # So, now, we have SK of shape E x K x N x N
    SK = SK.permute(0, 2, 3, 1).reshape(E, 1, 1, N, N, max(K)) 
        # E x 1(out) x 1(in) x N x N x K
    # And multiply this by the filters
    for l in range(L):
        # Reshape the filters to be E x out x in x K
        HS += [H[l].permute(1, 0, 3, 2)]
        # If K[l] < max(K), add zeros so that we can do the multiplication
        # easily
        if K[l]< max(K):
            extraZeros = torch.zeros(E,F[l+1],F[l],max(K)-K[l]).to(device)
            HS[l] = torch.cat((HS[l], extraZeros), dim = 3) 
            #    E x out x in x max(K)
        # And now we can add the extra dimensions
        HS[l] = HS[l].reshape(E, F[l+1], F[l], 1, 1, max(K)) 
            # E x out x in x 1 x 1 x max(K)
        # Multiply
        HS[l] = HS[l] * SK # E x out x in x N x N x max(K)
        # Add up
        HS[l] = torch.sum(HS[l], dim = 5) # E x out x in x N x N
            
    return HS

def computeIntegralLipschitzConstantMatrix(*args,
                                           useGSO = False,
                                           lowEgvl = -1.,
                                           highEgvl = 1.,
                                           nEigs = 200):
    """
    computeIntegralLipschitzConstantMatrix: computes the matrix of the integral
        Lipschitz constants of the filters, i.e. the F_in x F_out matrix \Gamma
        such that [\Gamma]_fg = \gamma^{fg} with \gamma^{fg} > 0 such that
            |\lambda dh^{fg}(\lambda)/d\lambda| \leq \gamma^{fg}
        for a list of graph filters.
        
    Input:
        (One argument)
        architecture (nn.Module): The nn.Module containing gml.GraphConv 
            layers, or similar. The architecture needs to have the following
            attributes: .S containing the graph shift operator, and .L, .F,
            .K, and .E as determined by the function getFilterTaps.
        -- OR --
        (One or two arguments)
        filterTaps (list): Each element in the list is a torch.tensor containing
            the filer taps from the architecture (torch.tensor with 
            requires_grad).
            The list has L elements, where L is the number of layers, and every
            element is a torch.tensor of shape
                F_{\ell} x E x K_{\ell} x F_{\ell-1}
            where F_{\ell} is the number of output features, E is the number of
            edge features, K_{\ell} is the number of filter taps, and F_{\ell-1}
            is the number of input features. Note that length of the list will
            be .L+len(.dimReadout).
        GSO (torch.tensor, required if useGSO = True): Graph shift operator,
            shape 1 x N x N. This input is not required if useGSO = False and 
            will be ignored if it is there.
    Options:
        useGSO (bool, default: False): Uses the GSO provided in architecture.S
            to compute the actual eigenvalues (computes an eigendecomposition,
            this is an expensive operation)
        lowEgvl (float, default: -1): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
        highEgvl (float, default: 1): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
        nEigs (int, default: 200): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
            
    Output:
        Gamma (list): Each element in the list is a torch.tensor containing the
            matrix of shape F_in x F_out with the integral Lipschitz constant of
            each of the F_in x F_out graph filters.

    """
    # Check that we have only one or two input arguments
    assert len(args) == 1  or len(args) == 2
    
    # If it's only one input argument
    if len(args) == 1:
        # This can be the architecture, or the filter taps right away
        if 'GraphFilter' in repr(args) or 'dimReadout' in repr(args):
            # Check if it is a graph convolutional architecture (i.e. has the 
            # GraphFilter layer)
            archit = args[0]
            # Now check we have a GSO
            if archit.S is None:
                # If we don't have a GSO, we cannot use it
                useGSO = False
            elif useGSO: # If we have an S and we are going to use it
                # Get S
                S = archit.S
                # Get the number of edge features
                E = S.shape[0]
                # Check that there's only one edge feature (right now the definition is
                # only for a single edge feature)
                assert E == 1
            # Now get the filter taps
            H = getFilterTaps(archit) # H[l]: F_out x E x K x F_in
        elif type(args[0]) is list:
            # If there's only one input argument and it's not the architecture,
            # then it has to be the filter taps, and there shouldn't be a 
            # requirement for a GSO specificed (because if useGSO = True, then
            # the second argument would be the GSO)
            assert useGSO == False
            H = args[0]
        else: # Unsuited architecture
            H = []
    else:
        # If we have two input arguments
        assert type(args[0]) is list and len(args[1].shape) == 3
        # The first one has to be filter taps
        H = args[0]
        # The second one has to be the GSO, but we only care if we are going to
        # use it
        if useGSO:
            S = args[1]
            # Check it has only one edge feature
            E = S.shape[0]
            assert E == 1
            # Get the number of nodes
            N = S.shape[1]
            # And check that it's square
            assert S.shape[2] == N
            
    L = len(H) # Number of layers
    if L > 0:
        # Now that we have collected the filter taps in H and the GSO in S we can
        # compute the eigenvalue or the sampling of the real line
        if useGSO:
            # Compute the eigendecomposition
            egvl = np.linalg.eigvals(S.detach().cpu().numpy())
            # Sort it from smallest to largest
            egvl = np.sort(egvl.squeeze())
            # Update the number of eigenvalues nEigs
            lowEgvl = np.min(egvl)
            highEgvl = np.max(egvl)
            nEigs = egvl.shape[0]
        else:
            # And if we don't have a GSO, so we need to create the sampling of the
            # eigenvalue real line
            egvl = np.linspace(lowEgvl, highEgvl, num = nEigs)
            
        # Now that we have the filter taps, we need to obtain the corresponding 
        # information
        # Get the device
        device = H[0].device
        # Get the parameters
        K = [] # Number of filter taps
        F = [H[0].shape[3]] # Number of features
        for l in range(L):
            # H[l] is of shape out x E x K x in
            K += [H[l].shape[2]]
            F += [H[l].shape[0]]
        # And the maximum number of filter taps
        maxK = max(K)
        
        # Recall that a polynomial h(\lambda) = \sum_{k=0}^{K-1} h_{k} \lambda^{k}
        # has a derivative dh/d\lambda = \sum_{k=1}^{K-1} h_{k} k\lambda^{k-1}
        # which implies that 
        # \lambda dh/d\lambda = \sum_{k=1}^{K-1} h_{k} k\lambda^{k}
        
        # This means that the Vandermonde matrix needs to be one less order, and
        # be accompanied by a multiplication by k. The first row is not the row of
        # ones but the row of eigenvalues.
        # It also means that the filter taps corresponding to k=0 are not used.
            
        # Now we need to compute the Vandermonde Matrix from nEigs to K x nEigs
        # \lambda_{1}             & \ldots  & \lambda_{nEigs}
        #   \vdots                & \ddots  &    \vdots
        # (K-1) \lambda_{1}^{K-1} & \ldots  & (K-1) \lambda_{nEigs}^{K-1}
        # Save space for the Vandermonde matrix V
        V = np.zeros((maxK-1, nEigs)) # We have one less tap in the derivative
        V[0,:] = egvl # k = 0
        # Each row is the multiplication of the previous row by (k+1)/k*egvl
        for k in range(1,maxK-1): # We have one less filter tap in the derivative
            V[k,:] = (k+1)/k * egvl * V[k-1,:]
        # Move it to a tensor
        V = torch.tensor(V, device = device)
        # And add the one dimensions corresponding to F_out so that the
        # multiplication will become (F_in x K) by (K x nEigs)
        V = V.reshape(1, maxK-1, nEigs)
        
        # And finally we can compute the integral Lipschitz constants
        Gamma = [] # Where to store the frequency response
        for l in range(L):
            if K[l] > 1: # If there's only one filter tap, there's nothing to 
            # compute because the derivative is zero (there's no lambda)
                thisTap = H[l].permute(1, 0, 3, 2).squeeze(0) # F_out x F_in x K
                # Get rid of the first filter tap
                thisTap = torch.narrow(thisTap, 2, 1, K[l]-1) # F_out x F_in x (K-1)
                # Multiply with the adapted matrix V
                thisDerivative = torch.matmul(thisTap, V[:,0:K[l]-1,:]) 
                    # (K-1) x nEigs
                # The maximum value of the absolute value of the derivative is the
                # Lipschitz constant
                Gamma += [torch.max(torch.abs(thisDerivative), dim = 2)[0]]
                #   F_out x F_in
    else:
        Gamma = []
    
    return Gamma # Gamma[l]: F_out x F_in

def computeLipschitzConstantMatrix(*args,
                                   useGSO = False,
                                   lowEgvl = -1.,
                                   highEgvl = 1.,
                                   nEigs = 200):
    """
    computeLipschitzConstantMatrix: computes the matrix of the Lipschitz 
        constants of the filters, i.e. the F_in x F_out matrix \Gamma such that
        [\Gamma]_fg = \gamma^{fg} with \gamma^{fg} > 0 such that
            |dh^{fg}(\lambda)/d\lambda| \leq \gamma^{fg}
        for a list of graph filters.
        
    Input:
        (One argument)
        architecture (nn.Module): The nn.Module containing gml.GraphConv 
            layers, or similar. The architecture needs to have the following
            attributes: .S containing the graph shift operator, and .L, .F,
            .K, and .E as determined by the function getFilterTaps.
        -- OR --
        (One or two arguments)
        filterTaps (list): Each element in the list is a torch.tensor containing
            the filer taps from the architecture (torch.tensor with 
            requires_grad).
            The list has L elements, where L is the number of layers, and every
            element is a torch.tensor of shape
                F_{\ell} x E x K_{\ell} x F_{\ell-1}
            where F_{\ell} is the number of output features, E is the number of
            edge features, K_{\ell} is the number of filter taps, and F_{\ell-1}
            is the number of input features. Note that length of the list will
            be .L+len(.dimReadout).
        GSO (torch.tensor, required if useGSO = True): Graph shift operator,
            shape 1 x N x N. This input is not required if useGSO = False and 
            will be ignored if it is there.
    Options:
        useGSO (bool, default: False): Uses the GSO provided in architecture.S
            to compute the actual eigenvalues (computes an eigendecomposition,
            this is an expensive operation)
        lowEgvl (float, default: -1): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
        highEgvl (float, default: 1): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
        nEigs (int, default: 200): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
            
    Output:
        Gamma (list): Each element in the list is a torch.tensor containing the
            matrix of shape F_in x F_out with the Lipschitz constant of each of
            the F_in x F_out graph filters.

    """
    # Check that we have only one or two input arguments
    assert len(args) == 1  or len(args) == 2
    
    # If it's only one input argument
    if len(args) == 1:
        # This can be the architecture, or the filter taps right away
        if 'GraphFilter' in repr(args) or 'dimReadout' in repr(args):
            # Check if it is a graph convolutional architecture (i.e. has the 
            # GraphFilter layer)
            archit = args[0]
            # Now check we have a GSO
            if archit.S is None:
                # If we don't have a GSO, we cannot use it
                useGSO = False
            elif useGSO: # If we have an S and we are going to use it
                # Get S
                S = archit.S
                # Get the number of edge features
                E = S.shape[0]
                # Check that there's only one edge feature (right now the definition is
                # only for a single edge feature)
                assert E == 1
            # Now get the filter taps
            H = getFilterTaps(archit) # H[l]: F_out x E x K x F_in
        # If it's not an architecture, then it should be a list with the filter
        # taps
        elif type(args[0]) is list:
            # If there's only one input argument and it's not the architecture,
            # then it has to be the filter taps, and there shouldn't be a 
            # requirement for a GSO specificed (because if useGSO = True, then
            # the second argument would be the GSO)
            assert useGSO == False
            H = args[0]
        else: # But it can also happen that it actually is an unsuited architecture
            H = []
    else:
        # If we have two input arguments
        assert type(args[0]) is list and len(args[1].shape) == 3
        # The first one has to be filter taps
        H = args[0]
        # The second one has to be the GSO, but we only care if we are going to
        # use it
        if useGSO:
            S = args[1]
            # Check it has only one edge feature
            E = S.shape[0]
            assert E == 1
            # Get the number of nodes
            N = S.shape[1]
            # And check that it's square
            assert S.shape[2] == N
    
    L = len(H) # Number of layers
    
    # IF there are filter taps, do something, if not, don't do anything
    if L > 0:
        # Now that we have collected the filter taps in H and the GSO in S we can
        # compute the eigenvalue or the sampling of the real line
        if useGSO:
            # Compute the eigendecomposition
            egvl = np.linalg.eigvals(S.detach().cpu().numpy())
            # Sort it from smallest to largest
            egvl = np.sort(egvl.squeeze())
            # Update the number of eigenvalues nEigs
            lowEgvl = np.min(egvl)
            highEgvl = np.max(egvl)
            nEigs = egvl.shape[0]
        else:
            # And if we don't have a GSO, so we need to create the sampling of the
            # eigenvalue real line
            egvl = np.linspace(lowEgvl, highEgvl, num = nEigs)
            
        # Now that we have the filter taps, we need to obtain the corresponding 
        # information
        # Get the device
        device = H[0].device
        # Get the parameters
        K = [] # Number of filter taps
        F = [H[0].shape[3]] # Number of features
        for l in range(L):
            # H[l] is of shape out x E x K x in
            K += [H[l].shape[2]]
            F += [H[l].shape[0]]
        # And the maximum number of filter taps
        maxK = max(K)
        
        # Recall that a polynomial h(\lambda) = \sum_{k=0}^{K-1} h_{k} \lambda^{k}
        # has a derivative dh/d\lambda = \sum_{k=1}^{K-1} h_{k} k\lambda^{k-1}
        
        # This means that the Vandermonde matrix needs to be one less order, and
        # be accompanied by a multiplication by k.
        # It also means that the filter taps corresponding to k=0 are not used.
            
        # Now we need to compute the Vandermonde Matrix from nEigs to K x nEigs
        #      1                  & \ldots  & 1
        # 2 \lambda_{1}           & \ldots  & 2 \lambda_{nEigs}
        #   \vdots                & \ddots  &    \vdots
        # (K-1) \lambda_{1}^{K-2} & \ldots  & (K-1) \lambda_{nEigs}^{K-2}
        # Save space for the Vandermonde matrix V
        V = np.zeros((maxK-1, nEigs)) # We have one less tap in the derivative
        V[0,:] = np.ones((1, nEigs)) # k = 0
        # Each row is the multiplication of the previous row by (k+1)/k*egvl
        for k in range(1,maxK-1): # We have one less filter tap in the derivative
            V[k,:] = (k+1)/k * egvl * V[k-1,:]
        # Move it to a tensor
        V = torch.tensor(V, device = device)
        # And add the one dimensions corresponding to F_out so that the
        # multiplication will become (F_in x K) by (K x nEigs)
        V = V.reshape(1, maxK-1, nEigs)
        
        # And finally we can compute the Lipschitz constants
        Gamma = [] # Where to store the frequency response
        for l in range(L):
            if K[l] > 1: # If there's only one filter tap, there's nothing to 
            # compute because the derivative is zero (there's no lambda)
                thisTap = H[l].permute(1, 0, 3, 2).squeeze(0) # F_out x F_in x K
                # Get rid of the first filter tap
                thisTap = torch.narrow(thisTap, 2, 1, K[l]-1) # F_out x F_in x (K-1)
                # Multiply with the adapted matrix V
                thisDerivative = torch.matmul(thisTap, V[:,0:K[l]-1,:]) 
                    # (K-1) x nEigs
                # The maximum value of the absolute value of the derivative is the
                # Lipschitz constant
                Gamma += [torch.max(torch.abs(thisDerivative), dim = 2)[0]]
                #   F_out x F_in
    else:
        Gamma = []
    
    return Gamma # Gamma[l]: F_out x F_in

def computeFilterSizeMatrix(*args):
    """
    computeFilterSizeMatrix: computes the matrix of the filter size, i.e. the
        F_in x F_out matrix C such that [C]_fg = \| H^{fg}(S) \|_{2} with
        H^{fg}(S) = \sum_{k=0}^{K-1} [H_{k}]_{fg} S^{k}, for a list of graph
        filters.
        
    Input: (One argument)
        architecture (nn.Module): The nn.Module containing gml.GraphConv 
            layers, or similar. The architecture needs to have the following
            attributes: .S containing the graph shift operator, and .L, .F,
            .K, and .E as determined by the function getFilterTaps.
        -- OR --
        HS (list): Each element in the list is a torch.tensor containing the
            graph filters, i.e., the polynomials on S, from the architecture 
            (torch.tensor with requires_grad).
            The list has L elements, where L is the number of layers, and every
            element is a torch.tensor of shape
                E x F_{\ell} x F_{\ell-1} x N x N
            where E is the number of edge features, F_{\ell} is the number of 
            output features, F_{\ell-1} is the number of input features, and
            N is the number of nodes.
        -- OR --
        h (list): Each element in the list is a torch.tensor containing the
            frequency responses, i.e., the polynomials on the variable lambda,
            from the architecture (torch.tensor with requires_grad).
            The list has L elements, where L is the number of layers, and every
            element is a torch.tensor of shape
                E x F_{\ell} x F_{\ell-1} x nEigs
            where E is the number of edge features, F_{\ell} is the number of 
            output features, F_{\ell-1} is the number of input features, and
            nEigs is the number of values in the lambda line.
            
    Output:
        C (list): Each element in the list is a torch.tensor containing the
            matrix of shape F_in x F_out containing the spectral norm of each of
            the F_in x F_out graph filters, computed by either frquency or 
            polynomial.

    """
    # Check that it has only one input argument
    assert len(args) == 1
    
    # If it is the architecture
    if 'GraphFilter' in repr(args[0]) or 'dimReadout' in repr(args[0]):
        # Save the architecture
        archit = args[0]
        # Get the frequency response OR the graph filters (this is a choice)
        H = getFrequencyResponse(archit) # Frequency response
        # H = getGraphFilters(archit) # Graph filters
    elif type(args[0]) is list:
        # Get the length of the list
        H = args[0]
    else:
        # This probably is because it's an architecture that doesn't have
        # graph filters, so we just fit an empty list that will yield an empty
        # matrix list C
        H = []
    
    # Get the number of layers
    L = len(H)
    # Create the list of matrices C
    C = []
    # If there are more than one filter (it could be called without this)
    if L > 0:
        # Now we have to check whether this is a graph polynomial or a frequency
        # response
        if len(H[0].shape) == 5: # E x F_out x F_in x N x N
            # If it is a graph polynomial
            for l in range(L):
                # Check the dimensions are correct
                assert H[l].shape[3] == H[l].shape[4]
                # Check it has only one edge feature
                assert H[l].shape[0] == 1
                # Compute the spectral norm
                C += [spectralNorm(H[l].squeeze(0))] # Get rid of the extra
                    # dimension here so that we don't mutate the elements in
                    # the list (otherwise we won't be able to call this function
                    # in succession)
        # Then it has to be a frequency response
        else:
            assert len(H[0].shape) == 4 # E x F_out x F_in x nEigs
            # For each layer
            for l in range(L):
                # Check it has only one edge feature
                assert H[l].shape[0] == 1
                # Compute the maximum across all the eigenvalues
                C += [torch.max(torch.abs(H[l].squeeze(0)), dim = 2)[0]]
    
    return C

def computeIntegralLipschitzConstant(*args, **kwargs):
    """    
    computeIntegralLipschitzConstant: computes the integral Lipschitz constant
        of a bank of  filters, defined as \| \mathbf{\Gamma} \|_{\infty} for the
        F_in x F_out matrix \mathbf{\Gamma} of integral Lipschitz constants, 
        for a list of graph filters.
    
    Input: (One argument)
        architecture (nn.Module): The nn.Module containing gml.GraphConv 
            layers, or similar. The architecture needs to have the following
            attributes: .S containing the graph shift operator, and .L, .F,
            .K, and .E as determined by the function getFilterTaps.
        -- OR --
        (One or two arguments)
        filterTaps (list): Each element in the list is a torch.tensor containing
            the filer taps from the architecture (torch.tensor with 
            requires_grad).
            The list has L elements, where L is the number of layers, and every
            element is a torch.tensor of shape
                F_{\ell} x E x K_{\ell} x F_{\ell-1}
            where F_{\ell} is the number of output features, E is the number of
            edge features, K_{\ell} is the number of filter taps, and F_{\ell-1}
            is the number of input features. Note that length of the list will
            be .L+len(.dimReadout).
        GSO (torch.tensor, required if useGSO = True): Graph shift operator,
            shape 1 x N x N. This input is not required if useGSO = False and 
            will be ignored if it is there.
        -- OR --
        (One argument)
        matrixGamma (list): Each element in the list is a torch.tensor 
            containing the integral Lipschitz constant matrix. The list has L 
            elements,  where L is the number  of layers, and every element is a 
            torch.tensor of shape
                F_{\ell} x F_{\ell-1}
            where F_{\ell} is the number of output features and F_{\ell-1} is
            the number of input features.
    
    Options (not used if the matrixGamma is the input):
        useGSO (bool, default: False): Uses the GSO provided in architecture.S
            to compute the actual eigenvalues (computes an eigendecomposition,
            this is an expensive operation)
        lowEgvl (float, default: -1): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
        highEgvl (float, default: 1): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
        nEigs (int, default: 200): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
            
    Output:
        Gamma (list): Each element in the list is a torch.tensor containing the
            scalar representing the integral Lipschitz constant of each filter
            bank in the list.
    """
    
    # Check if it is the matrix
    if type(args[0]) is list and len(args[0]) > 0 and len(args[0][0].shape)==2:
        # Now we know we're in the matrix case
        mtGamma = args[0]
    else:
        # If not, we let the computeFilterSizeMatrix function to handle all the
        # checkings
        mtGamma = computeIntegralLipschitzConstantMatrix(*args, **kwargs)
    
    # mtC is a list of matrices of dimension F_in x F_out
    L = len(mtGamma) # Get the number of elements
    
    # To compute the size of each filter, we create a list
    Gamma = []
    for l in range(L):
        Gamma += [torch.max(torch.sum(torch.abs(mtGamma[l]), dim = 1))]
        # return a single scalar which is the infinity norm of this matrix
    
    return Gamma

def computeLipschitzConstant(*args, **kwargs):
    """    
    computeLipschitzConstant: computes the Lipschitz constant of a bank of 
        filters, defined as \| \mathbf{\Gamma} \|_{\infty} for the F_in x F_out
        matrix \mathbf{\Gamma} of Lipschitz constants, for a list of graph
        ilters.
    
    Input: (One argument)
        architecture (nn.Module): The nn.Module containing gml.GraphConv 
            layers, or similar. The architecture needs to have the following
            attributes: .S containing the graph shift operator, and .L, .F,
            .K, and .E as determined by the function getFilterTaps.
        -- OR --
        (One or two arguments)
        filterTaps (list): Each element in the list is a torch.tensor containing
            the filer taps from the architecture (torch.tensor with 
            requires_grad).
            The list has L elements, where L is the number of layers, and every
            element is a torch.tensor of shape
                F_{\ell} x E x K_{\ell} x F_{\ell-1}
            where F_{\ell} is the number of output features, E is the number of
            edge features, K_{\ell} is the number of filter taps, and F_{\ell-1}
            is the number of input features. Note that length of the list will
            be .L+len(.dimReadout).
        GSO (torch.tensor, required if useGSO = True): Graph shift operator,
            shape 1 x N x N. This input is not required if useGSO = False and 
            will be ignored if it is there.
        -- OR --
        (One argument)
        matrixGamma (list): Each element in the list is a torch.tensor 
            containing the Lipschitz constant matrix. The list has L elements, 
            where L is the number  of layers, and every element is a 
            torch.tensor of shape
                F_{\ell} x F_{\ell-1}
            where F_{\ell} is the number of output features and F_{\ell-1} is
            the number of input features.
    
    Options (not used if the matrixGamma is the input):
        useGSO (bool, default: False): Uses the GSO provided in architecture.S
            to compute the actual eigenvalues (computes an eigendecomposition,
            this is an expensive operation)
        lowEgvl (float, default: -1): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
        highEgvl (float, default: 1): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
        nEigs (int, default: 200): If we do not want to use the eigenvalues 
            of the GSO, we create a variable lambda which goes from lowEgvl
            to highEgvl and samples the line with nEigs points. If useGSO = True
            this is ignored.
            
    Output:
        Gamma (list): Each element in the list is a torch.tensor containing the
            scalar representing the Lipschitz constant of each filter bank in 
            the list.
    """
    
    # Check if it is the matrix
    if type(args[0]) is list and len(args[0]) > 0 and len(args[0][0].shape)==2:
        # Now we know we're in the matrix case
        mtGamma = args[0]
    else:
        # If not, we let the computeFilterSizeMatrix function to handle all the
        # checkings
        mtGamma = computeLipschitzConstantMatrix(*args, **kwargs)
    
    # mtC is a list of matrices of dimension F_in x F_out
    L = len(mtGamma) # Get the number of elements
    
    # To compute the size of each filter, we create a list
    Gamma = []
    for l in range(L):
        Gamma += [torch.max(torch.sum(torch.abs(mtGamma[l]), dim = 1))]
        # return a single scalar which is the infinity norm of this matrix
    
    return Gamma

def computeFilterSize(*args):
    """
    computeFilterSize: computes the size of a bank of filters, defined as
        \| C \|_{\infty}
    for the F_in x F_out matrix of spectral norms (see computeFilterSizeMatrix),
    for element in the list.
    
    Input: (One argument)
        architecture (nn.Module): The nn.Module containing gml.GraphConv 
            layers, or similar. The architecture needs to have the following
            attributes: .S containing the graph shift operator, and .L, .F,
            .K, and .E as determined by the function getFilterTaps.
        -- OR --
        HS (list): Each element in the list is a torch.tensor containing the
            graph filters, i.e., the polynomials on S, from the architecture 
            (torch.tensor with requires_grad).
            The list has L elements, where L is the number of layers, and every
            element is a torch.tensor of shape
                E x F_{\ell} x F_{\ell-1} x N x N
            where E is the number of edge features, F_{\ell} is the number of 
            output features, F_{\ell-1} is the number of input features, and
            N is the number of nodes.
        -- OR --
        h (list): Each element in the list is a torch.tensor containing the
            frequency responses, i.e., the polynomials on the variable lambda,
            from the architecture (torch.tensor with requires_grad).
            The list has L elements, where L is the number of layers, and every
            element is a torch.tensor of shape
                E x F_{\ell} x F_{\ell-1} x nEigs
            where E is the number of edge features, F_{\ell} is the number of 
            output features, F_{\ell-1} is the number of input features, and
            nEigs is the number of values in the lambda line.
        -- OR --
        C (list): Each element in the list is a torch.tensor containing the
            filter size matrix. The list has L elements, where L is the number 
            of layers, and every element is a torch.tensor of shape
                F_{\ell} x F_{\ell-1}
            where F_{\ell} is the number of output features and F_{\ell-1} is
            the number of input features.
            
    Output:
        C (list): Each element in the list is a torch.tensor containing the
            scalar representing the filter size of each graph filter in the 
            list.
    """
    # Check that it has only one input argument
    assert len(args) == 1
    
    # Check if it is the matrix
    if type(args[0]) is list and len(args[0]) > 0 and len(args[0][0].shape)==2:
        # Now we know we're in the matrix case
        mtC = args[0]
    else:
        # If not, we let the computeFilterSizeMatrix function to handle all the
        # checkings
        mtC = computeFilterSizeMatrix(*args)
    
    # mtC is a list of matrices of dimension F_in x F_out
    L = len(mtC) # Get the number of elements
    
    # To compute the size of each filter, we create a list
    C = []
    for l in range(L):
        C += [torch.max(torch.sum(torch.abs(mtC[l]), dim = 1))]
        # return a single scalar which is the infinity norm of this matrix
    
    return C

def GNNsize(model, data, device):
    """
    GNNsize: penatly function that computes the size of a given GNN.
    
    The size of a GNN is computed as
        C_{\phi} = \prod_{\ell = 1}^{L} C_{H_{\ell}}
    where H_{\ell} is the bank of filters at layer ell, and consequently, the
    size of the filter H is used
        C_{H} = \| \mathbf{C}_{H} \|_{\infty}
    with matrix \mathbf{C}_{H} of shape F_in x F_out, where the (f,g)th element
    is given by \| \mathbf{H}^{fg}(\mathbf{S}) \|_{2}, i.e, by the spectral norm
    of the (f,g)th polynomial filter in the bank, where f = 1,\ldots,F_in and
    g = 1,\ldots,F_out.
    
    Input:
        model (Modules.model.Model class) contains the architecture in .archit
        data (Utils.dataTools._data class) -not used, but input for consistency-
        
    Output:
        C (torch.tensor): size of the GNN, it may have gradients (if the filter 
            taps of the architecture require gradients).

    """
    # Compute the size of the GNN defined as
    # C_{\phi} = \sup_{\ell in \{1, \ldots, L\}} C_{H_{\ell}}
    # with
    # C_{H_{\ell}} = \| \mathbf{C}_{H_{\ell}} \|_{\infty}
    # where \mathbf{C}_{H_{\ell}} \in \mathbb{R}^{F_{\ell-1} \times F_{\ell}}
    # such that
    # [C_{H_{\ell}}]_{fg} = \| \mathbf{H}_{\ell}^{fg}(\mathbf{S}) \|_{2}
    # where 
    # \mathbf{H}_{\ell}^{fg}(\mathbf{S}) 
    #    = \sum_{k=0}^{K_{\ell}-1} [\mathbf{H}_{\ell k}]_{fg} \mathbf{S}^{k}
    # Recall that the graph filter at each layer is
    # \sum_{k=0}^{K_{\ell}} \mathbf{S}^{k} \mathbf{X}_{\ell-1} \mathbf{H}_{\ell k}
    
    # So, first we need the architecture
    # Then, we need S
    S = model.S
    # Check that S exists
    assert S is not None
    # Get the number of edge features
    E = S.shape[0]
    # Check that there's only one edge feature (right now the definition is
    # only for a single edge feature)
    assert E == 1
    
    # Now get the list of sizes for each layer
    C = computeFilterSize(model)
    # NOTE: By default, as it is now, this computes the values using the 
    # frequency response, with no input from the GSO whatsoever. If we want
    # to use the GSO change computeFilterSizeMatrix.
    
    # Now, each of the elements in this list has a different tensor (a scalar)
    # potentially with gradients. And I want to get the product of them to
    # get the size of the GNN, so
    
    # First, get the length of the list
    L = len(C)
    # Compute Cphi only if the list is not empty
    if L > 0:
        Cphi = torch.tensor(1., device = device)
        for l in range(L):
            Cphi = Cphi * C[l]
    else:
    # Because if it's empty, then the architecture is not suited for a size
        Cphi = torch.tensor(float('nan'), device = device)
        
    return Cphi

def GNNintegralLipschitz(model, data):
    """
    GNNintegralLipschitz: penalty function that computes the integral Lipschitz
        constant of a given GNN.
    
    The integral Lipschitz constant of a given GNN is computed as
        \Gamma_{\phi} = \sup_{\ell \in \{1,\ldots,L\}} \Gamma_{H_{\ell}}
    where H_{\ell} is the bank of filters at layer ell, and consequently, the
    integral Lipschitz constant of the filter bank H is
        \Gamma_{H} = \| \mathbf{\Gamma}_{H} \|_{\infty}
    with matrix \mathbf{\Gamma}_{H} of shape F_in x F_out, where the (f,g)th 
    entry is given by \gamma^{fg} > 0 such that
        | \lambda dh^{fg}(\lambda)/d\lambda | < \gamma^{fg}
    where h^{fg}(\lambda) is the polynomial in lambda using coefficients given 
    by the corresponding filter taps, and where f = 1,\ldots,F_in and 
    g = 1,\ldots,F_out.
    
    Input:
        model (Modules.model.Model class) contains the architecture in .archit
        data (Utils.dataTools._data class) -not used, but input for consistency-
        
    Output:
        Gamma (torch.tensor): integral Lipschitz constant of the GNN, it may 
            have gradients (if the filter taps of the architecture require
            gradients).

    """
    
    # So, first we need the architecture
    archit = model.archit
    # Then, we need S
    S = archit.S
    # Check that S exists
    assert S is not None
    # Get the number of edge features
    E = S.shape[0]
    # Check that there's only one edge feature (right now the definition is
    # only for a single edge feature)
    assert E == 1
    
    # Now get the list of sizes for each layer
    Gamma = computeIntegralLipschitzConstant(archit)
    # Remember that this function has options like using the GSO or the sampling
    # of the real line to compute the Lipschitz constant.
    
    # Finally, simply take the maximum of every element in the list
    L = len(Gamma)
    if L > 0:
        GammaPhi = torch.tensor(0., device = model.device)
        for l in range(L):
            if Gamma[l] > GammaPhi:
                GammaPhi = Gamma[l]
    else:
        GammaPhi = torch.tensor(float('nan'), device = model.device)
        
    return GammaPhi

def GNNlipschitz(model, data):
    """
    GNNlipschitz: penalty function that computes the Lipschitz constant of a
        given GNN.
    
    The Lipschitz constant of a given GNN is computed as
        \Gamma_{\phi} = \sup_{\ell \in \{1,\ldots,L\}} Gamma_{H_{\ell}}
    where H_{\ell} is the bank of filters at layer ell, and consequently, the
    Lipschitz constant of the filter bank H is
        \Gamma_{H} = \| \mathbf{\Gamma}_{H} \|_{\infty}
    with matrix \mathbf{\Gamma}_{H} of shape F_in x F_out, where the (f,g)th 
    entry is given by \gamma^{fg} > 0 such that
        | dh^{fg}(\lambda)/d\lambda | < \gamma^{fg}
    where h^{fg}(\lambda) is the polynomial in lambda using coefficients given 
    by the corresponding filter taps, and where f = 1,\ldots,F_in and 
    g = 1,\ldots,F_out.
    
    Input:
        model (Modules.model.Model class) contains the architecture in .archit
        data (Utils.dataTools._data class) -not used, but input for consistency-
        
    Output:
        Gamma (torch.tensor): Lipschitz constant of the GNN, it may have 
            gradients (if the filter taps of the architecture require gradients)

    """
    
    # So, first we need the architecture
    archit = model.archit
    # Then, we need S
    S = archit.S
    # Check that S exists
    assert S is not None
    # Get the number of edge features
    E = S.shape[0]
    # Check that there's only one edge feature (right now the definition is
    # only for a single edge feature)
    assert E == 1
    
    # Now get the list of sizes for each layer
    Gamma = computeLipschitzConstant(archit)
    # Remember that this function has options like using the GSO or the sampling
    # of the real line to compute the Lipschitz constant.
    
    # Finally, simply take the maximum of every element in the list
    L = len(Gamma)
    if L > 0:
        GammaPhi = torch.tensor(0., device = model.device)
        for l in range(len(Gamma)):
            if Gamma[l] > GammaPhi:
                GammaPhi = Gamma[l]
    else:
        GammaPhi = torch.tensor(float('nan'), device = model.device)
        
    return GammaPhi

def L2stabilityConstant(model, A, B, device):
    """
    L2stabilityConstat: penalty function that computes the stability constant
        \| \mathbf{A} \|_{2} + C_{\Phi}^{L} \| \mathbf{B} \|_{2}
    where \mathbf{A} and \mathbf{B} are the linear model matrices and C_{\Phi}
    is the size of the GNN.
    
    Input:
        model (Modules.Model class) contains the GNN architecture in the .archit
            attribute, it is assumed that the nonlinearities are Lipschitz
            continuous with C_{\sigma} = 1 and sigma(0) = 0.
        data (LQR data class) contains the matrices \mathbf{A} and \mathbf{B} in
            the attributes .A and .B, respectively.
            
    Output:
        stabilityConstant (torch.tensor): scalar with the value of the stability
            constant
    """
    
    # Check they are defined
    assert A is not None and B is not None
    # And have the proper dimensions
    N = A.shape[0]
    assert A.shape[1] == B.shape[0] == B.shape[1]
    # Get the architecture
    # Check the GSO has been defined
    S = model.S
    assert S is not None
    assert S.shape[1] == S.shape[2] == N
    # Check it has only one edge feature, since that's the only scenario
    # consdiered so far
    E = S.shape[0]
    assert E == 1 # Right now it doesn't work for multiple edge features
    
    # Compute the size of the matrices
    Anorm = np.linalg.norm(A.detach().cpu().numpy(), ord = 2)
    Anorm = torch.tensor(Anorm, device = device)
    Bnorm = np.linalg.norm(B.detach().cpu().numpy(), ord = 2)
    Bnorm = torch.tensor(Bnorm, device = device)
    CPhi = GNNsize(model, None, device)
    
    # Finally, compute the stability constant
    return Anorm + CPhi * Bnorm
