#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:18:54 2020

@author: nathancrugge
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

## Read data
    
infile = open('US_Accidents_Dec19_with_ClusterLabels', 'rb')
df = pickle.load(infile)
infile.close()

print('Imported {:,} rows.'.format(len(df)))

### FILTER TO SELECTED STATE ###
sel_state = 'GA'
df_state = df.loc[df['State']==sel_state]

print('Filtered to ', sel_state, '- contains {:,} rows.'.format(len(df_state)))

# Calc cluster average severities
df_state['Cluster_Avg_Sev'] = df_state['Severity'].groupby(df_state['Cluster']).transform('mean')

def encode_and_bind(original_df, feature_to_encode):
    dummies = pd.get_dummies(original_df[[feature_to_encode]])
    res = pd.concat([original_df, dummies], axis=1)
#    res = res.drop([feature_to_encode], axis=1)
    return res

df_state['Severity'] = df_state['Severity'].apply(str)
df_state = encode_and_bind(df_state, 'Severity')
df_state['Severity'] = df_state['Severity'].apply(int)

# Calc Adjusted y Metric
df_state['Severity_Adj'] = df_state['Severity'] - df_state['Cluster_Avg_Sev']

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

y_train = train['Severity_Adj']
y_test = test['Severity_Adj']

## Linear Regression with Recursive Feature Elimination

linreg = LinearRegression()

rfe = RFE(linreg, 15)
rfe = rfe.fit(x_train, y_train)

rfe_idx = [i for i,j in enumerate(rfe.ranking_) if j == 1]
rfe_cols = [predictor_cols[i] for i in rfe_idx]
print("Selected Features (Linear Regression Model w RFE): ", rfe_cols)

x_train_rfe = pd.DataFrame(train, columns = rfe_cols)
x_test_rfe = pd.DataFrame(test, columns = rfe_cols)

linreg.fit(x_train_rfe, y_train)

y_preds = linreg.predict(x_train_rfe)
print("Lin Reg Training Accuracy: {:.2f}".format(linreg.score(x_train_rfe, y_train)))
print("Lin Reg Test Accuracy: {:.2f}".format(linreg.score(x_test_rfe, y_test)))

# Train GBR
gbmod = GradientBoostingRegressor(subsample=0.5, random_state=126, max_features='auto',
                                  verbose=0)

gbmod.fit(x_train, y_train)

gb_feature_importances = pd.DataFrame(gbmod.feature_importances_, 
                                      index = x_train.columns,
                                      columns = ['importance']).sort_values('importance', ascending = False)

print("Gradient Boosting Regressor Feature Importance:", gb_feature_importances.head(15))

print("GBR Training Accuracy: {:.2f}".format(gbmod.score(x_train, y_train)))
print("GBR Test Accuracy: {:.2f}".format(gbmod.score(x_test, y_test)))
