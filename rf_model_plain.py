'''
Created on 12/03/2015

@author: David Manzano
'''
import csv as csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from time import time


df_train = pd.read_csv('data/train.csv', header=0)
df_prod = pd.read_csv('data/test.csv', header=0)

print "train info"
print df_train.info()

print "prod info"
print df_prod.info()


# Cleaning data
print "Cleaning and normalizing data"
df_train = df_train.drop(['Id'], axis=1) 

## Collect the test data's Ids before dropping it
ids = df_prod['Id'].values
df_prod = df_prod.drop(['Id'], axis=1) 


train_data = df_train.values
train_data = train_data.astype('float')
prod_data = df_prod.values
prod_data = prod_data.astype('float')

#print "y"
#print train_data[0::,-1]

#print "x"
#print train_data[0::,0:54]

###############################################################################
# Split into a training set and a test set using a stratified k fold
x_train, x_test, y_train, y_test = cross_validation.train_test_split(train_data[0::,0:54],train_data[0::,54],test_size=0.25)

###############################################################################
# Normalize train and prod data
#print x_train
min_max_scaler = preprocessing.MinMaxScaler()
x_train_minmax=min_max_scaler.fit_transform(x_train)
x_prod=min_max_scaler.fit_transform(prod_data)
print "-----------------------"
print "X_train"
print x_train_minmax
print "xtrain shape"
print x_train_minmax.shape
print "X_prod"
print x_prod
print "xprod shape"
print x_prod.shape

###############################################################################
# Training the model
print("Fitting the model to the training set")
t0 = time()
n_estimators = [100, 150, 200, 250, 300]
results = [];
for estimators in n_estimators:
    model = RandomForestClassifier(n_estimators=estimators)
    model = model.fit(x_train_minmax, y_train)
    print("done in %0.3fs" % (time() - t0))


###############################################################################
# Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = model.predict(x_test)
    print("done in %0.3fs" % (time() - t0))
    target_names=['Cover_Type1', 'Cover_Type2', 'Cover_Type3', 'Cover_Type4', 'Cover_Type5', 'Cover_Type6', 'Cover_Type7']
    #n_classes = target_names.shape[0]
    print "Classification report"
    report = metrics.classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    print "Confusion matrix"
    cfm=metrics.confusion_matrix(y_test, y_pred)
    print(cfm)

    # Predict with production data
    print ('Processing test data...')
    y_pred = model.predict(x_prod)
    # Save to file
    predictions_file = open("output/modelRF"+str(estimators)+".csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Id","Cover_Type"])
    open_file_object.writerows(zip(ids, y_pred.astype(np.int)))
    predictions_file.close()
    result =[estimators, report, cfm]
    results.append(result)
print "##########################################################"
print results
print ('Done.')