
def logistic_reg(X,y):
  """
  """
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.1,random_state=1)
  lr = LogisticRegressionCV(random_state=0).fit(X_train, y_train)
  prediction=lr.predict(X_test)
  #score model
  score = f1_score(y_test, prediction, average='macro')
  #save model
  dump(lr, 'logistic_reg.joblib')
  return (prediction , y_train , score)


def xg_boost(X , y):
  """
  """
  # convert the dataset into a Dmatrix that gives it  performance and efficiency gains.
  data_dmatrix = xgb.DMatrix(data=X,label=y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
  xg_reg = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
  xg_reg.fit(X_train,y_train)
  preds = xg_reg.predict(X_test)
  params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
  cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="auc", as_pandas=True, seed=123)
  #score model
  score = f1_score(y_test, preds, average='macro')
  #save model
  dump(xg_reg, 'xg_boost.joblib')
  return (preds , cv_results , score)

def multi_layer_percep(X , y):
  """
  
  """
  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
  clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
  preds = clf.predict(X_test)
  #score model
  score = f1_score(y_test, preds, average='macro')
  #save model
  dump(clf, 'xg_boost.joblib')
  return  (preds ,y_train , score) 