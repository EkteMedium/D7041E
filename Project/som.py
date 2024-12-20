import numpy as np
from matplotlib import pyplot as plt
import math
import time
from tqdm import tqdm
import pandas as pd
import seaborn as sn
import matplotlib

def getEuclideanDistance(single_point,array):
    nrows, ncols, nfeatures=array.shape[0],array.shape[1], array.shape[2]
    points=array.reshape((nrows*ncols,nfeatures))
                
    dist = (points - single_point)**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist.astype(float))

    dist=dist.reshape((nrows,ncols))
    return dist


def plot_SOM(dispRes,som,ndim,text=None,subfig:matplotlib.figure.SubFigure=None):
        if subfig is None:
            fig, ax=plt.subplots(nrows=ndim, ncols=ndim, figsize=(15,15))
        else:
            fig = subfig
            ax=subfig.subplots(nrows=ndim, ncols=ndim)
            subfig.subplots_adjust(left=0.0625,bottom=0,right=0.9375,top=0.875,wspace=0,hspace=0)
        
        title = "\n"+str(ndim)+"x"+str(ndim)+" SOM grid."
        if text is not None:
             title = title +"\n"+text
        fig.suptitle(title, fontsize=20)
        
        for k in range(ndim):
            for l in range (ndim):
                A=som[k,l,:].reshape((dispRes[0],dispRes[1]))
                ax[k,l].imshow(A,cmap="plasma")
                ax[k,l].set_yticks([])
            ax[k,l].set_xticks([])

def SOM (trainingData, ndim=10, nepochs=10, eta0=0.1, etadecay=0.05, sgm0=20, sgmdecay=0.05, showMode=0, dispRes=None):
    nfeatures=trainingData.shape[1]
    ntrainingvectors=trainingData.shape[0]
    
    nrows = ndim
    ncols = ndim
    
    mu, sigma = 0, 0.1
    np.random.seed(int(time.time()))
    som = np.random.normal(mu, sigma, (nrows,ncols,nfeatures))

    if showMode >= 1:
        plot_SOM(dispRes=dispRes,som=som,ndim=ndim,text="0% of iterations")
    
    #Generate coordinate system
    x,y=np.meshgrid(range(ncols),range(nrows))
    
    
    for t in tqdm(range (1,nepochs+1)):
        if t==round((nepochs+1)/2) and showMode>=1:
            plot_SOM(dispRes=dispRes,som=som,ndim=ndim,text="50% of iterations")

        #Compute the learning rate for the current epoch
        eta = eta0 * math.exp(-t*etadecay)
        
        #Compute the variance of the Gaussian (Neighbourhood) function for the ucrrent epoch
        sgm = sgm0 * math.exp(-t*sgmdecay)
        
        #Consider the width of the Gaussian function as 3 sigma
        width = math.ceil(sgm*3)
        
        for ntraining in range(ntrainingvectors):

            trainingVector = trainingData[ntraining,:]
            
            # Compute the Euclidean distance between the training vector and
            # each neuron in the SOM map
            dist = getEuclideanDistance(trainingVector, som)
       
            # Find 2D coordinates of the Best Matching Unit (bmu)
            bmurow, bmucol =np.unravel_index(np.argmin(dist, axis=None), dist.shape)
              
            #Generate a Gaussian function centered on the location of the bmu
            g = np.exp(-((np.power(x - bmucol,2)) + (np.power(y - bmurow,2))) / (2*sgm*sgm))

            #Determine the boundary of the local neighbourhood
            fromrow = max(0,bmurow - width)
            torow   = min(bmurow + width,nrows)
            fromcol = max(0,bmucol - width)
            tocol   = min(bmucol + width,ncols)

            
            #Get the neighbouring neurons and determine the size of the neighbourhood
            neighbourNeurons = som[fromrow:torow,fromcol:tocol,:]
            sz = neighbourNeurons.shape
            
            #Transform the training vector and the Gaussian function into 
            # multi-dimensional to facilitate the computation of the neuron weights update
            T = np.matlib.repmat(trainingVector,sz[0]*sz[1],1).reshape((sz[0],sz[1],nfeatures));                  
            G = np.dstack([g[fromrow:torow,fromcol:tocol]]*nfeatures)

            # Update the weights of the neurons that are in the neighbourhood of the bmu
            neighbourNeurons = neighbourNeurons + eta * G * (T - neighbourNeurons)

            
            #Put the new weights of the BMU neighbouring neurons back to the
            #entire SOM map
            som[fromrow:torow,fromcol:tocol,:] = neighbourNeurons

    if showMode >= 1:
        plot_SOM(dispRes=dispRes,som=som,ndim=ndim,text="100% of iterations")
    return som
    

    #verification of correctness on the training set:
def SOM_Test ( som_, X_train, L_train, X_test, L_test, nclasses, ndim=60,grid_=None,ConfusionMatrix=None):
    nfeatures=X_train.shape[1]
    ntrainingvectors=X_train.shape[0]
    
    nrows = ndim
    ncols = ndim
    
    if grid_ is None:
        grid_=np.zeros((nrows,ncols))

    if ConfusionMatrix is None:
        ConfusionMatrix=np.zeros((nclasses,nclasses))

    som_cl=np.zeros((ndim,ndim,nclasses+1))
    
    for ntraining in range(ntrainingvectors):
        trainingVector = X_train[ntraining,:]
        class_of_sample= L_train[ntraining]    
        # Compute the Euclidean distance between the training vector and
        # each neuron in the SOM map
        dist = getEuclideanDistance(trainingVector, som_)
       
        # Find 2D coordinates of the Best Matching Unit (bmu)
        bmurow, bmucol =np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        
        
        som_cl[bmurow, bmucol,class_of_sample]=som_cl[bmurow, bmucol,class_of_sample]+1
    
    
    
    for i in range (nrows):
        for j in range (ncols):
            grid_[i,j]=np.argmax(som_cl[i,j,:])

    
    ntestingvectors=X_test.shape[0]
    for ntesting in range(ntestingvectors):
        testingVector = X_test[ntesting,:]
        class_of_sample= L_test[ntesting]    
        # Compute the Euclidean distance between the training vector and
        # each neuron in the SOM map
        dist = getEuclideanDistance(testingVector, som_)
       
        # Find 2D coordinates of the Best Matching Unit (bmu)
        bmurow, bmucol =np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        
        predicted=np.argmax(som_cl[bmurow, bmucol,:])
        ConfusionMatrix[class_of_sample, predicted]=ConfusionMatrix[class_of_sample, predicted]+1
        
    return grid_, ConfusionMatrix
    


def display_conf(conf,classes,text,ax:matplotlib.axes.Axes=None,normalize=False):

    # Calculate accuracy
    total=0
    correct=0
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            total += conf[i,j]
            if i==j:
                correct +=conf[i,j]

    accuracy = correct/total

    # Calculate Average recall and precision
    sum_recall = 0
    sum_precision = 0
    for i in range(conf.shape[0]):
        sum_recall += conf[i,i]/sum(conf[i,:])
        sum_precision += conf[i,i]/sum(conf[:,i])
    avg_recall = sum_recall/conf.shape[0]
    avg_precision = sum_precision/conf.shape[0]
    avg_f1 = 2 * ((avg_precision*avg_recall)/(avg_precision+avg_recall))

    if normalize:
        # Normalize plot
        c2 = np.astype(((conf/np.sum(conf,axis=1)[:,np.newaxis])*1000)+0.5,int)/10
        conf = pd.DataFrame(c2, index = classes,
                      columns =classes)
    if ax is None:
        plt.figure(figsize=(10,8))
        plt.title(text+"\n\nAccuracy: " + str(round(accuracy*10000)/100) + " %.\nMacro-Average F1-score: " + str(round(avg_f1*10000)/100))
        sn.heatmap(conf,annot=True, fmt='.3g', cbar=False)
        plt.xlabel("Prediction")
        plt.ylabel("Label")
    else:
        ax.set_title(text+"\nAccuracy: " + str(round(accuracy*10000)/100) + " %.\nMacro-Average F1-score: " + str(round(avg_f1*10000)/100),None)
        sn.heatmap(conf,annot=True, fmt='.3g', cbar=False, ax=ax)
        ax.set_xlabel("Prediciton")
        ax.set_ylabel("Label")

