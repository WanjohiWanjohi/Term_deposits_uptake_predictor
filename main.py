
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split , KFold
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import argparse
#custom imports
from data import *
from model import *
def main():
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',action='store_true', default= 'bank-additional.csv'  help='Name of the file you want to load')
    args = parser.parse_args()

    result = []
   if args.filename == 'bank-additional.csv':
     df= pd.read_csv(args.filename , sep=";")
     categorical = ['job' , 'marital' , 'education' , 'contact' , 'housing' , 'default' , 'loan' , 'poutcome']
      target = ['y' ]
      binned = ['pdays']
      dates = ['month' , 'day_of_week']
      converted_df = convert_categorical(df , categorical)
      month = convert_dates(df , dates[0]  , '%b')
      day = convert_dates(df , dates[1] , '%a')
      scaled = scale_columns(df , binned)
      scaled_emp_no = scale_columns(df , ['nr.employed'])
      scaled_age = scale_columns(df , ['age'])
      #   we start by encoding the content of the target column
      target_df = convert_categorical(df , target)

      #   use upsample function
      upsampled = upsample(target_df ,'y_no' , 'y_yes' )

      target_column = pd.DataFrame(upsampled , columns=["deposit"])
      converted_df.drop(categorical , axis=1 , inplace=True)
      converted_df

      # bring in the campaign columns and the age column
      converted_df['age'] = scaled_age
      converted_df['campaign'] = df['campaign']
      converted_df['previous'] = df['previous']

      #bring in the derived month column
      converted_df['month'] = month

      #bring in the scaled employee numbers and number of days that pass before a customer is contacted
      converted_df['pdays'] = scaled
      converted_df['employee_no'] = scaled_emp_no

      #calculate total number of contacts made with user
      sum_column = converted_df['previous'] + converted_df['campaign']
      converted_df['total_contacts'] = sum_column

      converted_df.drop('day' , axis=1 , inplace=True)
      converted_df.astype('float64').dtypes

      #transform features
      transformed_features = pca_reduction(converted_df , 10)

      #define target and training features
      X = transformed_features  
      y = get_target_sample(target_column , 'deposit' , 41188  )

      #logistic regression
     log_r= logistic_reg(X , y)

      #gradient boost
      xg = xg_boost(X, y)

      #multilayer perceptron
      percep = multi_layer_percep(X , y)

      return log_r , xg , percep




     

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print('Something went wrong {0}'.format(e))