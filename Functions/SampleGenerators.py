from scipy.stats import norm
import numpy as np

# Function to make a data set which has percentage_A samples drawn from parameters_A
# and 1-percentage_A drawn from parameters_B. The TotalSize is defaulted to 200000 but can be changed.
def MakeSamples(parameters_A, parameters_B, percentage_A, TotalSize=200000):
    sizeA = int(percentage_A*TotalSize)
    sizeB = TotalSize-sizeA
    
    setA=[]
    for mu0, sigma0 in parameters_A:
        setA.append(norm.rvs(loc=mu0,scale=sigma0,size=sizeA))
    setA=np.array(setA)
    ones = np.ones([1,setA.shape[1]])
    setA=np.transpose(np.append(setA,ones,axis=0))

    
    setB=[]
    for mu0, sigma0 in parameters_B:
        setB.append(norm.rvs(loc=mu0,scale=sigma0,size=sizeB))
    setB=np.array(setB)
    zeros = np.zeros([1,setB.shape[1]])
    setB=np.transpose(np.append(setB,zeros,axis=0))
  
    npout = np.vstack([setA,setB])
    
    npout = np.concatenate( (npout,percentage_A * np.ones([npout.shape[0],1])), axis=1)
    
    return npout

# Function to make a data set which has percentage_A samples drawn from parameters_A
# and 1-percentage_A drawn from parameters_B.
# This function labels the data with percentage_A_Expected even though percentage_A is the real value used
# in splitting the data.
def MakeSamplesWithError(parameters_A, parameters_B, percentage_A, percentage_A_Expected, TotalSize=200000):
   
    sizeA = int(percentage_A*TotalSize)
    sizeB = TotalSize-sizeA
    
    setA=[]
    for mu0, sigma0 in parameters_A:
        setA.append(norm.rvs(loc=mu0,scale=sigma0,size=sizeA))
    setA=np.array(setA)
    ones = np.ones([1,setA.shape[1]])
    setA=np.transpose(np.append(setA,ones,axis=0))
    
    setB=[]
    for mu0, sigma0 in parameters_B:
        setB.append(norm.rvs(loc=mu0,scale=sigma0,size=sizeB))
    setB=np.array(setB)
    zeros = np.zeros([1,setB.shape[1]])
    setB=np.transpose(np.append(setB,zeros,axis=0))
    
    npout = np.vstack([setA,setB])    
    npout = np.concatenate( (npout,percentage_A_Expected * np.ones([npout.shape[0],1])), axis=1)
    
    return npout

# Function to make a data set which has percentage_A samples drawn from parameters_A
# and 1-percentage_A drawn from parameters_B.
# This function labels the data with percentage_A_Expected even though percentage_A is the real value used
# in splitting the data.
# Now parameters_A and parameters_B can have multiple gaussians in them
def MakeMultiGSamples(parameters_A, parameters_B, percentage_A, percentage_A_Expected, TotalSize=200000):
   
    sizeA = int(percentage_A*TotalSize)
    sizeB = TotalSize-sizeA
    print sizeA, sizeB
    
    setA=[]
    for feature in parameters_A:
        #print feature
        tmp_feat=np.array([])
        for (mu0, sigma0), percent in feature:
            #print mu0, sigma0, percent
            tmp_feat=np.append(tmp_feat,norm.rvs(loc=mu0,scale=sigma0,size=int(sizeA*percent)))
            #print "tmp_feat:",tmp_feat
            np.random.shuffle(tmp_feat)
        setA.append(tmp_feat)
        #print setA
                            
    setA=np.array(setA)
    ones = np.ones([1,setA.shape[1]])
    setA=np.transpose(np.append(setA,ones,axis=0))
    #print setA
    
    setB=[]
    for feature in parameters_B:
        tmp_feat=np.array([])
        for (mu0, sigma0), percent in feature:
            #print mu0, sigma0, percent
            tmp_feat=np.append(tmp_feat,norm.rvs(loc=mu0,scale=sigma0,size=int(sizeB*percent)))
            np.random.shuffle(tmp_feat)
        setB.append(tmp_feat)
    setB=np.array(setB)
    zeros = np.zeros([1,setB.shape[1]])
    setB=np.transpose(np.append(setB,zeros,axis=0))
    
    print "Set1 shape:",setA.shape
    print "Set2 shape:",setB.shape
    
    
    
    npout = np.vstack([setA,setB])    
    npout = np.concatenate( (npout,percentage_A_Expected * np.ones([npout.shape[0],1])), axis=1)
    
    return npout