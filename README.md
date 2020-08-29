# Term_deposits_uptake_predictor

## Objective
The objective of this notebook is to answer whether or not a customer will subscribe to the `term_deposit` depending on a set of features.


## Data

The data used herein is found and described on the UCI Machine Learning Repository 
http://archive.ics.uci.edu/ml/datasets/Bank+Marketing#

It consists of the following columns:
#### bank client data:
- age (numeric)
- job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
- marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
- education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
- default: has credit in default? (categorical: 'no','yes','unknown')
- housing: has housing loan? (categorical: 'no','yes','unknown')
- loan: has personal loan? (categorical: 'no','yes','unknown')
#### related with the last contact of the current campaign:
- contact: contact communication type (categorical: 'cellular','telephone')
- month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
- day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
- duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
#### other attributes:
- campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
- pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
- previous: number of contacts performed before this campaign and for this client (numeric)
- poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
#### social and economic context attributes
- emp.var.rate: employment variation rate - quarterly indicator (numeric)
- cons.price.idx: consumer price index - monthly indicator (numeric)
- cons.conf.idx: consumer confidence index - monthly indicator (numeric)
- euribor3m: euribor 3 month rate - daily indicator (numeric)
- nr.employed: number of employees - quarterly indicator (numeric)

## Methods Applied
To prepare the data for the algorithms , a Principal component approach is applied on the data which has been encoded using onehot encoding

The methods applied include a Logistic Regression algorithm , a Gradient Booster algorithm , and a mulitilayer feed forward neural network using k-fold cross validation to continously correct the error on the previous training set. 
To identify the best model , a score of 

## Problems

The problems with this approach is the use of k-fold cross validation
## Limitations

## Conclusions

# How to run
1. Download as zip folder
2. Extract and navigate to folder 
	`cd Term_deposits_uptake_predictor
3. Run the module:
 ` python3 main.py
