def minesota_prior(model,decay = .9,discount = .1):
    """ Adjusting the priors according to Litterman’s ‘Minnesota Prior’
    
    There are many approaches in the literature to construct priors. For example, 
    Litterman’s ‘Minnesota Prior’ specifies the location of the first lag as 1 and 
    the rest 0 as a random walk is often appropriate for economic time series. The 
    Minnesota prior also contains a way to downweight the importance of past lags 
    through the covariance hyperparameters. Here we follow the Litterman prior for 
    the first moments, and set the location of the first own lags to 1 and the rest 
    to 0. For the downweighting, we do something quick and dirty and penalize the 
    longer lags with a discount rate \delta. 
    
    Maybe this could be generalized later according to the approaches reported in 
    http://static1.1.sqspcdn.com/static/f/1335391/26409938/1437581777390/BMR.pdf?token=JVBKOoxMjqcRxFlnGDLWLkxQSd0%3D
    
    Parameters
    ----------
    discount : float
        delta in the example
    decay : float
        fixed in the example to 0.5
    """    

    feat = model.data_original.shape[1]
    lags = model.lags
    
    size = (1+lags*feat) * feat
    incr = (1+lags*feat)
    const = [x for x in range(0,size,incr)]
    varAR = [x for x in range(1,size,incr)]
    crossvarAR = []
    for j in range(2,feat+1):
        crossvarAR += [x for x in range(j,size+j,incr)]
    
    model.adjust_prior([x for x in const],pf.Normal(0,100)) # constant prior - fairly uninformative

    for i in range(lags): 
        if i == 0:
            model.adjust_prior([el+feat*i for el in varAR],pf.Normal(1,1.0*(discount**(i)))) # AR(1) lags
        else:
            model.adjust_prior([el+feat*i for el in varAR],pf.Normal(0,decay*(discount**(i)))) # AR(1) lags
        model.adjust_prior([el+feat*i for el in crossvarAR],pf.Normal(0,decay*(discount**(i)))) # AR(2+) lags 
            
def featureIndex(model,nr):
    """ Gather indices of a variable's latent lag parameters 

    Parameter
    ----------
    nr : int
        number of a variable for which the indices of the latent variables in the model should be determined 
    """    
    
    feat = model.data_original.shape[1]
    lags = model.lags
    
    assert nr > 0 and nr <= feat, "variable number %d does not exist " % nr
    
    return [x for x in range(nr,feat*lags,feat)]
    
