d_train=lightgbm.Dataset(xtrain, label=ytrain)
d_test=lightgbm.Dataset(xtest, label=ytest)
nb_classes=len(ytrain.value_counts()) 
params={'boosting_type':'gbdt', 'objective':'multiclass', 'metric':'multi_logloss', 'learning_rate':0.01,
       'num_classes':nb_classes, 'subsample_freq':3, 'min_child_samples':5, 'min_child_weight':0.1,
       'colsample_bytree':0.8, 'subsample':0.7, 'min_split_gain':0.05, 'max_bin':25, 'max_depth':-1,
       'num_leaves':25, 'early_stopping_round':100, 'random_state':rs} 
n_estimators=1000
watchlist=[d_train, d_test]
model=lightgbm.train(params=params, train_set=d_train, num_boost_round=n_estimators, valid_sets=watchlist, verbose_eval=10)
