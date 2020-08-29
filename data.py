
def convert_categorical(dataframe , columns):
   df = dataframe[columns]
   converted_df = pd.get_dummies(df)
   res = pd.concat([df, converted_df], axis=1)
   return res

def get_column_counts(df ):
  """
  Get count of unique items in the dataframe column
  """
  for c in df.columns:
    print(c)
    print(df[c].value_counts())

def scale_columns(df , column):
  """"
  Scale columns using MinMax scaler
  """
  col = df[column]
  scaler = MinMaxScaler() 
  num2 = scaler.fit_transform(col)
  num2 = pd.DataFrame(num2, columns = col.columns)
  return num2

def upsample(df , majority_col, minority_col):
  """"
  Perform upsampling by resampling while replacing
  """
  # we seperate the classes
  majority_class = df[majority_col]
  minority_class = df[minority_col]
  #Upsample minority class
  df_minority_upsampled = resample(minority_class, 
                                 replace=True,     # sample with replacement
                                 n_samples=36548,    # to match majority class
                                 random_state=123) # reproducible results
  # Combine majority class with upsampled minority class
  df_upsampled = pd.concat([majority_class, df_minority_upsampled])
  return df_upsampled

def convert_dates(df , column , format):
  """
  convert date and months to numerical format

  """
  months = []
  day_of_week = []
  #convert column items to strings
  
    #format months
  if format == "%b":
    for index, row in df[column].items():
      datetime_object = datetime.datetime.strptime(row , "%b")
      month_number = datetime_object.month
      months.append(month_number)
    return months
    ## format days
  elif format == "%a":
    for index, row in df[column].items():
      datetime_object = datetime.datetime.strptime(row , "%a")
      day_number = datetime_object.weekday()
      day_of_week.append(day_number)
    return day_of_week
def get_correlation_map(correlation_matrix):
  """
  return a heatmap of the correlation
  """
  plt.figure(figsize=(20,5))
  return sns.heatmap(correlation_matrix)


def get_target_sample(df , column , sample_number):
  """
  """
  sample = df[column].sample(n=int(sample_number), random_state=1)
  return sample

def pca_reduction(df , components):
  """
  """
  feat_cols = df.columns.tolist()
  pca_data = PCA(n_components=components).fit(df)
  var_exp = pca_data.explained_variance_ratio_
  #apply the dimensionality reduction
  transformed = pca_data.transform(df)
  
  return transformed