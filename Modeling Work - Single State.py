#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:18:53 2020

@author: nathancrugge
"""

import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

## Read data
    
infile = open('US_Accidents_Dec19', 'rb')
df = pickle.load(infile)
infile.close()

print('Imported {:,} rows.'.format(len(df)))

### FILTER TO SELECTED STATE ###
sel_state = 'GA'
#df_state = df.loc[df['State']==sel_state]
df_state = df

print('Filtered to ', sel_state, '- contains {:,} rows.'.format(len(df_state)))

def encode_and_bind(original_df, feature_to_encode):
    dummies = pd.get_dummies(original_df[[feature_to_encode]])
    res = pd.concat([original_df, dummies], axis=1)
#    res = res.drop([feature_to_encode], axis=1)
    return res

df_state['Severity'] = df_state['Severity'].apply(str)
df_state = encode_and_bind(df_state, 'Severity')
df_state['Severity'] = df_state['Severity'].apply(int)

## Split into train/test

train, test = train_test_split(df_state, test_size = 0.3, random_state = 126)

# Excluded potential predictors:
# Pressure(in) - There is a mass point at ~30, which is also where all the severity is (was massively skewing the lasso results); probably some default value?
# Precipitation(in) - Used a y/n based on this field plus Weather_Condition instead
# Removed all of the _Night dummy variables, since I'm including the _Day versions
predictor_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)',  
                  'Start_Hour', 'Start_Month', 'Weekend', 'Precipitation', 'Wind_Direction_CALM', 
                  'Wind_Direction_E', 'Wind_Direction_ENE', 'Wind_Direction_ESE', 
                  'Wind_Direction_N', 'Wind_Direction_NE', 'Wind_Direction_NNE', 'Wind_Direction_NNW', 'Wind_Direction_NW', 'Wind_Direction_S', 
                  'Wind_Direction_SE', 'Wind_Direction_SSE', 'Wind_Direction_SSW', 'Wind_Direction_SW', 'Wind_Direction_VAR', 'Wind_Direction_W',
                  'Wind_Direction_WNW', 'Wind_Direction_WSW', 'Weather_Condition_Clear', 'Weather_Condition_Dust', 'Weather_Condition_Fog', 
                  'Weather_Condition_Funnel Cloud', 'Weather_Condition_Hail', 'Weather_Condition_Overcast', 'Weather_Condition_Rain', 
                  'Weather_Condition_Sleet', 'Weather_Condition_Snow', 'Weather_Condition_Thunderstorm', 'Weather_Condition_Tornado', 
                  'Sunrise_Sunset_Day', 'Civil_Twilight_Day', 'Nautical_Twilight_Day', 'Astronomical_Twilight_Day']


x_train = pd.DataFrame(train, columns = predictor_cols)
x_test = pd.DataFrame(test, columns = predictor_cols)

y_train = train['Severity']
y_test = test['Severity']

y_train1 = train['Severity_1']
y_test1 = test['Severity_1']

y_train2 = train['Severity_2']
y_test2 = test['Severity_2']

y_train3 = train['Severity_3']
y_test3 = test['Severity_3']


y_train4 = train['Severity_4']
y_test4 = test['Severity_4']

## Gradient Boosting Classifier as Baseline
# Consider logistic regression for 4 severity "classes", then take highest prob as label

#gbclass = GradientBoostingClassifier(random_state = 126, max_features = 'auto', 
#                                     subsample = 0.5, verbose = 1)
#gbclass.fit(x_train, y_train)
#
#gb_feature_importances = pd.DataFrame(gbclass.feature_importances_, 
#                                      index = x_train.columns,
#                                      columns = ['importance']).sort_values('importance', ascending = False)
#
#print("Gradient Boosting Classifier Feature Importance:", gb_feature_importances.head(15))
#
#y_preds = gbclass.predict(x_test)
#
#print('Confusion Matrix (rows are actual, columns are predicted):')
#print(confusion_matrix(y_test, y_preds))
#print('Classification Report:')
#print(classification_report(y_test, y_preds, digits=3))

## Gradient Boosting Classifier - All 4 Labels at Once
#Filtered to  GA - contains 83,620 rows.
#      Iter       Train Loss      OOB Improve   Remaining Time 
#         1       34324.8280        1524.0629           36.95s
#         2       33090.4703        1258.7412           35.61s
#         3       31976.3073        1052.5433           35.68s
#         4       31134.5199         887.1583           35.12s
#         5       30334.8344         750.7214           34.81s
#         6       29673.2420         637.4605           34.62s
#         7       29191.8190         558.7511           34.13s
#         8       28644.3103         474.4000           33.65s
#         9       28209.6896         414.9593           34.53s
#        10       27910.5744         359.2969           34.94s
#        20       25874.7781         105.7380           29.76s
#        30       25154.3908          41.2259           25.53s
#        40       24789.0091          18.2018           21.68s
#        50       24648.1212           8.3250           17.94s
#        60       24461.2104           5.6551           14.45s
#        70       24428.9453           1.6458           10.77s
#        80       24348.7754           0.4488            7.16s
#        90       24259.1340           1.0234            3.60s
#       100       24267.8721          -0.6404            0.00s
#Gradient Boosting Classifier Feature Importance:                            importance
#Start_Hour                   0.330385
#Weekend                      0.221152
#Wind_Speed(mph)              0.072954
#Humidity(%)                  0.067907
#Weather_Condition_Clear      0.051558
#Temperature(F)               0.044730
#Wind_Chill(F)                0.043502
#Start_Month                  0.040008
#Visibility(mi)               0.017021
#Nautical_Twilight_Day        0.014031
#Civil_Twilight_Day           0.010552
#Astronomical_Twilight_Day    0.009031
#Weather_Condition_Fog        0.008867
#Wind_Direction_N             0.008547
#Wind_Direction_NNE           0.007200
#Confusion Matrix:
#[[   0    7    8    0]
# [   1 5212 5154   47]
# [   0 3287 9238  144]
# [   0  537 1243  208]]
#Classification Report:
#              precision    recall  f1-score   support
#
#           1      0.000     0.000     0.000        15
#           2      0.576     0.500     0.536     10414
#           3      0.591     0.729     0.653     12669
#           4      0.521     0.105     0.174      1988
#
#   micro avg      0.584     0.584     0.584     25086
#   macro avg      0.422     0.334     0.341     25086
#weighted avg      0.579     0.584     0.566     25086


## Logistic Regression with Recursive Feature Elimination

logreg = LogisticRegression(random_state=126, solver='lbfgs', max_iter=1000)

# Severity selection (just update the number for both y measures)
# ALSO UPDATE GRAPH LABEL
ytrain = y_train4
ytest = y_test4

rfe = RFE(logreg, 15)
rfe = rfe.fit(x_train, ytrain)

rfe_idx = [i for i,j in enumerate(rfe.ranking_) if j == 1]
rfe_cols = [predictor_cols[i] for i in rfe_idx]
print("Selected Features: ", rfe_cols)

x_train_rfe = pd.DataFrame(train, columns = rfe_cols)
x_test_rfe = pd.DataFrame(test, columns = rfe_cols)

logreg.fit(x_train_rfe, ytrain)

y_preds = logreg.predict(x_train_rfe)
print("Training Accuracy: {:.2f}".format(logreg.score(x_train_rfe, ytrain)))
print("Test Accuracy: {:.2f}".format(logreg.score(x_test_rfe, ytest)))

y_preds = logreg.predict(x_test_rfe)

print('Confusion Matrix (rows are actual, columns are predicted):')
print(confusion_matrix(ytest, y_preds))

print('Classification Report:')
# Note: Precision = TP / (TP + FP), Recall = TP / (TP + FN)
print(classification_report(ytest, y_preds, digits=3))

# Plot ROC Curve
diag_probs = [0 for _ in range(len(ytest))]

lr_probs = logreg.predict_proba(x_test_rfe)
lr_probs = lr_probs[:, 1]

diag_auc = roc_auc_score(ytest, diag_probs)
lr_auc = roc_auc_score(ytest, lr_probs)

print('Logistic Regression ROC AUC = %.3f' % (lr_auc))

diag_fpr, diag_tpr, _ = roc_curve(ytest, diag_probs)
lr_fpr, lr_tpr, _ = roc_curve(ytest, lr_probs)

plt.plot(diag_fpr, diag_tpr, linestyle='--')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic - Sev 4›')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

### Logistic Regression with RFE - Severity 1
#Selected Features:  ['Weekend', 'Wind_Direction_E', 'Wind_Direction_ENE', 'Wind_Direction_ESE', 'Wind_Direction_NE', 'Wind_Direction_NNW', 'Wind_Direction_NW', 'Wind_Direction_S', 'Wind_Direction_SSW', 'Wind_Direction_W', 'Wind_Direction_WNW', 'Wind_Direction_WSW', 'Weather_Condition_Rain', 'Sunrise_Sunset_Day', 'Civil_Twilight_Day']
#Training Accuracy: 1.00
#Test Accuracy: 1.00
#Confusion Matrix (rows are actual, columns are predicted):
#[[25071     0]
# [   15     0]]
#Classification Report:
#              precision    recall  f1-score   support
#
#           0      0.999     1.000     1.000     25071
#           1      0.000     0.000     0.000        15
#
#   micro avg      0.999     0.999     0.999     25086
#   macro avg      0.500     0.500     0.500     25086
#weighted avg      0.999     0.999     0.999     25086

### Logistic Regression with RFE - Severity 2
#Selected Features:  ['Weekend', 'Wind_Direction_E', 'Wind_Direction_ESE', 'Wind_Direction_N', 'Wind_Direction_NNE', 'Wind_Direction_NNW', 'Wind_Direction_NW', 'Wind_Direction_SSE', 'Wind_Direction_W', 'Weather_Condition_Clear', 'Weather_Condition_Fog', 'Weather_Condition_Thunderstorm', 'Civil_Twilight_Day', 'Nautical_Twilight_Day', 'Astronomical_Twilight_Day']
#Training Accuracy: 0.61
#Test Accuracy: 0.60
#Confusion Matrix (rows are actual, columns are predicted):
#[[12276  2396]
# [ 7643  2771]]
#Classification Report:
#              precision    recall  f1-score   support
#
#           0      0.616     0.837     0.710     14672
#           1      0.536     0.266     0.356     10414
#
#   micro avg      0.600     0.600     0.600     25086
#   macro avg      0.576     0.551     0.533     25086
#weighted avg      0.583     0.600     0.563     25086

### Logistic Regression with RFE - Severity 3
#Selected Features:  ['Weekend', 'Precipitation', 'Wind_Direction_CALM', 'Wind_Direction_N', 'Wind_Direction_NE', 'Wind_Direction_NNE', 'Wind_Direction_NNW', 'Wind_Direction_NW', 'Weather_Condition_Fog', 'Weather_Condition_Overcast', 'Weather_Condition_Rain', 'Weather_Condition_Thunderstorm', 'Civil_Twilight_Day', 'Nautical_Twilight_Day', 'Astronomical_Twilight_Day']
#Training Accuracy: 0.58
#Test Accuracy: 0.58
#Confusion Matrix (rows are actual, columns are predicted):
#[[6771 5646]
# [4962 7707]]
#Classification Report:
#              precision    recall  f1-score   support
#
#           0      0.577     0.545     0.561     12417
#           1      0.577     0.608     0.592     12669
#
#   micro avg      0.577     0.577     0.577     25086
#   macro avg      0.577     0.577     0.577     25086
#weighted avg      0.577     0.577     0.577     25086


### Logistic Regression with RFE - Severity 4
#Selected Features:  ['Weekend', 'Wind_Direction_NNE', 'Wind_Direction_NW', 'Wind_Direction_SSW', 'Wind_Direction_VAR', 'Wind_Direction_WNW', 'Weather_Condition_Clear', 'Weather_Condition_Fog', 'Weather_Condition_Hail', 'Weather_Condition_Overcast', 'Weather_Condition_Rain', 'Weather_Condition_Snow', 'Weather_Condition_Thunderstorm', 'Nautical_Twilight_Day', 'Astronomical_Twilight_Day']
#Training Accuracy: 0.92
#Test Accuracy: 0.92
#Confusion Matrix (rows are actual, columns are predicted):
#[[23098     0]
# [ 1987     1]]
#Classification Report:
#              precision    recall  f1-score   support
#
#           0      0.921     1.000     0.959     23098
#           1      1.000     0.001     0.001      1988
#
#   micro avg      0.921     0.921     0.921     25086
#   macro avg      0.960     0.500     0.480     25086
#weighted avg      0.927     0.921     0.883     25086

