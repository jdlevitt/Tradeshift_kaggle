########Tradeshift Classification Challenge##############
#
# Code from Dmitry Dryomov with my comments and some tweaks
#
# This was originally written in an Ipython notebook
#
##########################################################

# Importations
import pandas as pd
import numpy as np
from collections import Counter 
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, f1_score, log_loss, make_scorer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC

from datetime import datetime

print "*"*50
print ""
print ""
print "########Tradeshift Classification Challenge##############"
print ""


######## Data Importation and File Management ############
start = datetime.now()
print "Importing data and munging..."
train = pd.read_csv('train.csv', engine = 'c')


# Taking a sample of data for CV

sample_fraction = 0.1  # Set by hand for now
sample_size = sample_fraction * train.shape[0]   

ratio = 1/sample_fraction   #redundant, but makes things easier to read
train_sample = train[[hash(id) % ratio == 0 for id in train['id']]]

train_sample.to_csv('train_sample.csv', index = False)   # index False to prevent the extra ,
del train   #free some memory

# Now we merge labels with our file
labels = pd.read_csv('trainLabels.csv', engine = 'c')

train_with_labels = pd.merge(train_sample, labels, on = 'id')
#train_with_labels = pd.merge(train, labels, on = 'id')


# cleanup
del labels
del train_sample   # might be able to have this all pre-made before sending to ec2
#del train
# loading test data
test = pd.read_csv('test.csv', engine = 'c')






######### Data Wrangling and Feature Encoding ################

# Initializations
X_numerical = []
X_test_numerical = []

vec = DictVectorizer()

names_categorical = []

# Encode yes/no's as numerical

train_with_labels.replace('YES', 1, inplace = True)
train_with_labels.replace('NO', 0, inplace = True)
train_with_labels.replace('nan', np.NaN, inplace = True)

test.replace('YES', 1, inplace = True)
test.replace('NO', 0, inplace = True)
test.replace('nan', np.NaN, inplace = True)

# Encode rest of features

for name in train_with_labels.columns:
    if name.startswith('x'): # only selecting features, not labels
        # Find the dominant data type in a column
        column_type, _ = max(Counter(map(lambda x: str(type(x)), train_with_labels[name])).items(), key = lambda x:x[1])
        
        if column_type == str(str):
            train_with_labels[name] = map(str, train_with_labels[name])
            #test[name] = map(str, test[name])
            
            names_categorical.append(name)
            
            #extra
            #print name, len(np.unique(train_with_labels[name]))
        
        #elif column_type == float:
            #mn = train_with_labels[name].median()
            #X_numerical.append(train_with_labels[name].fillna(mn))
            #X_test_numerical.append(test[name].fillna(-999))
            
        else:
            X_numerical.append(train_with_labels[name].fillna(-999))
            X_test_numerical.append(test[name].fillna(-999))
            
            
# Creating arrays from the numerical columns and categorical
X_numerical = np.column_stack(X_numerical)
X_test_numerical = np.column_stack(X_test_numerical)

X_sparse = vec.fit_transform(train_with_labels[names_categorical].T.to_dict().values())
X_test_sparse = vec.transform(test[names_categorical].T.to_dict().values())

X_numerical = np.nan_to_num(X_numerical)
X_test_numerical = np.nan_to_num(X_test_numerical)


print "Munging complete... compressing."
print "Time so far:", str(datetime.now() - start)
print "-"*20
# Compression

joblib.dump( (X_numerical, X_sparse, X_test_numerical, X_test_sparse), 
    'X.dump', compress = 1, )



###################################################################################

###################################################################################

#### LOOP for HYERPARAMETER TUNING #########################

# List of numbers of features to use in RF for max_features, where default root of n between 11 and 12
# Can further bisect or something later... I prefer to handcode rather than automate this

feature_nums = [12, 15, 17] 
result_log = {} # dictionary of results from the grid search

for feats in feature_nums:
    print "Beginning tuning with %d features used in the forest..." % feats


######### Base Classifier Level #############

    log_loss_scorer = make_scorer(log_loss, needs_proba = True)

#some definitions

    y_columns = [name for name in train_with_labels.columns if name.startswith('y')]

    X_numerical_base, X_numerical_meta, X_sparse_base, X_sparse_meta, y_base, y_meta = train_test_split(X_numerical, X_sparse, train_with_labels[y_columns].values,
            test_size = 0.5) # Note these are random splits 50/50

    X_meta = []
    X_test_meta = []

# training base layer and building meta layer
    print "Training base layer, then building meta layer...."


    for i in range(y_base.shape[1]): # a prediction for each label
        print i
    
        y = y_base[:,i]
        if len(np.unique(y)) == 2:
        
        #Random forest on numerical features
            rf = RFC(n_estimators = 80, max_features = feats, n_jobs = -1)
            rf.fit(X_numerical_base, y)
            X_meta.append(rf.predict_proba(X_numerical_meta)[:,1])
            #X_test_meta.append(rf.predict_proba(X_test_numerical)[:,1])
        
        #SVC on categorical features
            svm = LinearSVC()
            svm.fit(X_sparse_base, y)
            X_meta.append(svm.decision_function(X_sparse_meta))
        #X_test_meta.append(svm.decision_function(X_test_sparse))
        print "  -- Total time so far is:", str(datetime.now() - start)    
        
        
    X_meta = np.column_stack(X_meta)
    #X_test_meta = np.column_stack(X_test_meta)


    print " Meta layer built!  Starting training..."
    print "-" * 40




######## Meta Level Training and Prediction #########

    p_test = []
    score_tot = []
    for i in range(y_base.shape[1]):
        y = y_meta[:,i]
    
        constant = Counter(y)
        constant = constant[0] < 4 or constant[1] < 4
    
        predicted = None
    
        if constant:
        # Best constant   //// basically when almost all the labels are the same
            constant_pred = np.mean(list(y_base[:,i]) + list(y_meta[:,i]))
        
            #predicted = np.ones(X_test_meta.shape[0]) * constant_pred
            #print "%d is constant like: %f" % (i, constant_pred)
        
        else: #fit a random forest to the meta level
            rf = RFC(n_estimators = 80, max_features = 17, n_jobs = -1) # estimators were at 20... maybe go way higher
            rf.fit(np.hstack([X_meta, X_numerical_meta]),y)
        
            #predicted = rf.predict_proba(np.hstack([X_test_meta, X_test_numerical])[:,1])
            #predicted = predicted[:,1] #redundant
        
            #rf = RFC(n_estimators = 30, n_jobs = -1) Do I need this?
            scores = cross_val_score(rf, np.hstack([X_meta, X_numerical_meta]),y, cv = 4,
                n_jobs = -1, scoring = log_loss_scorer)
        
        #print i, 'RF log-loss: %.4f +/- %.4f, mean = %.6f' %(np.mean(scores), np.std(scores),np.mean(predicted))
            print i, 'RF log-loss: %.4f +/- %.4f' %(np.mean(scores), np.std(scores))
            print ""
        
    #p_test.append(predicted)
        score_tot.append(np.mean(scores))
        print "  -- Total time so far is:", str(datetime.now() - start)
    
    #p_test = np.column_stack(p_test)


    print '-' * 40        
    print "For %d max features," % feats
    print "overall log loss on CV is: ", np.sum(score_tot)/33  
    print "Total time so far is:", str(datetime.now() - start)

    result_log[feats] = np.sum(score_tot)/33

############### print results ###################

print "\n"
print "#" * 50
print "\n"
print "The logloss results of our hyperparameter search are:"

for feats in feature_nums:

    print "max_features = %d \t \t logloss: %f" % (feats, result_log[feats])  
    
print "Hope this is useful! Bye!" 
    








