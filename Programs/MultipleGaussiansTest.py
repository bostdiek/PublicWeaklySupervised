import sys
import numpy as np
import scipy as sp
import scipy.stats

import pandas as pd

from scipy.stats import norm

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

#Import our functions
sys.path.append('../Functions/')
from SampleGenerators import *
from WeakCostFunction import *

def MakeErrorTrials(err, samples=100):

#   Take from the toy model
    set0nums = [[[(26,8),0.5],[(5,4),0.5]],[[(0.09,0.04),0.5],[(-0.01,0.03),0.5]],[[(0.45,0.04),0.5],[(0.08,0.05),0.5]]]
    set1nums = [[[(18,7),0.5],[(38,9),0.5]],[[(-0.06,0.04),0.5],[(0.15,0.03),0.5]],[[(0.23,0.05),0.5],[(0.4,0.08),0.5]]]

#     Make the first data set
    Set_40_60=MakeMultiGSamples(set0nums, set1nums, 0.4,0.4, 200000)
    XTrain1 = Set_40_60[:,:3]
    YTrainPercentage1 = Set_40_60[:,4]
    YTrain1 = Set_40_60[:,3]

#     Make the test data set: Stays the same for all cases
    Set_40_60t = MakeMultiGSamples(set0nums, set1nums, 0.4,0.4, 100000)
    Set_70_30t = MakeMultiGSamples(set0nums, set1nums, 0.7,0.7, 100000)
    XTest = np.vstack([Set_40_60t[:,:3], Set_70_30t[:,:3]])
    YTest = np.append(Set_40_60t[:,3], Set_70_30t[:,3])

    outarray = [['ErrorBase', 'RunNumber', 'ErrorPercent', 'Label', 'ROC_AUC']]
    i = 1
    while i <= samples:
        print str(i)+'/'+str(samples),
        print '\r',

        Set_70_30 = MakeMultiGSamples(set0nums, set1nums, 0.7,float(err)/100, 200000)

        XTrain2 = Set_70_30[:,:3]
        YTrain2 = Set_70_30[:,3]
        YTrainPercentage2 = Set_70_30[:,4]

        x_train, x_valid, y_train, y_valid = train_test_split(
                    np.vstack([XTrain1,XTrain2]),
                    np.append(YTrainPercentage1,YTrainPercentage2),
                    test_size=0.2)

        #Scale the data
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(XTest)

        print "X_Train"
        print x_train[:32]

        print "Y_Train"
        print y_train[:32]


        #Re-initialize the model
        WeaklyModel= Sequential()
        WeaklyModel.add(Dense(30, activation="sigmoid", kernel_initializer="normal", input_dim=3))
        WeaklyModel.add(Dense(1, activation="sigmoid", kernel_initializer="normal"))
        WeaklyModel.compile(loss=WeakSupervision,optimizer=Adam(lr=0.00009, clipnorm=1.))

        #Train
        checkpointer = ModelCheckpoint('../Data/KerasModelWeights/MuliG_weights_'+str(err)+'_'+str(i)+'.h5',
                          monitor='val_loss',
                          save_best_only=True)
        es=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
        WeaklyModel.fit(x_train, y_train,
                        validation_data=(x_valid, y_valid),
                        batch_size=32,
                        epochs=100,
                        verbose=0,
                        callbacks=[checkpointer, es])
        WeaklyModel.load_weights('../Data/KerasModelWeights/MuliG_weights_'+str(err)+'_'+str(i)+'.h5')


        #Test
        preds=WeaklyModel.predict_proba(x_test, verbose=0)
        print "Y_Predictions, Y_True"
        print np.append(preds[:10], YTest[:10])
        print np.append(preds[-10:], YTest[-10:])
        tpr, fpr, thresh = metrics.roc_curve(YTest[:],preds[:])
        auc_score = metrics.auc(fpr,tpr)
        if auc_score != 0:
            if auc_score < 0.5: #got flipped
                fpr, tpr, thresh = metrics.roc_curve(YTest[:],preds[:])
                auc_score = metrics.auc(fpr,tpr)
            outarray.append([err, i, (float(err)/100. )/0.7, YTrainPercentage2[0], auc_score])
            print "AUC:{0}".format(auc_score)
            with open('../Data/TrueFalsePositiveRates/MuliG_weights_fpr_' + str(err) + '_'+str(i) + '.dat','w') as ffpr:
                for l in fpr:
                    ffpr.write(str(l))
                    ffpr.write('\n')
            with open('../Data/TrueFalsePositiveRates/MuliG_weights_tpr_'+ str(err) + '_' + str(i) + '.dat','w') as tfpr:
                for l in tpr:
                    tfpr.write(str(l))
                    tfpr.write('\n')
            df=pd.DataFrame(outarray[1:],columns=outarray[0])
            df.to_csv('../Data/TrueFalsePositiveRates/MuliG_Error_InFraction_' + str(err) + '.dat',index=False)

            i+=1
    print df.head(6)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) != 2:
        print "There must be two inputs. Should be run as"
        print "...> python MultipleGaussiansTest.py Label SampleSize"
        print "where label is 70 for a label of 0.7 "
        print "and SampleSize is an integer value\n"

    else:
        error = int(argv[0])
        samplesize = int(argv[1])

        MakeErrorTrials(error, samplesize)
