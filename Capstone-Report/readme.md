# Factors associated with differences in Life Expectancy across the United States
	A Capstone project: 
        	Tonia Chu
	Under the mentorship: 
   	    	Srdjan Santic (Consulting Data Scientist at the Research Centre for Cheminformatics)
	For the course: 
		Data Science Career Track (Springboard)
  
## I. Introduction
Addressing socioeconomic disparities in health is a major policy goal. Yet what is the magnitude of socioeconomic gaps in life expectancy? How these gaps are changing over time? And what are their determinants? Answers to these questions are not clear. 

In this project, we use new data from 1.4 billion anonymous earnings and mortality records to construct more precise estimates of the relationship between income and life expectancy at the national level. We then construct new local area (county and metro area) estimates of life expectancy by income group and identify factors that are associated with higher levels of life expectancy for low-income individuals. 

The purpose of this project is to characterize life expectancy by income, over time, and across areas. We will use de-identified data from tax records covering the US population from 2001-2014 to characterize income-mortality gradients. We will also characterize correlates of the spatial variation and construct publicly available statistics. We will build a model to predict the life expectancy of individuals by their age, income, living area and other aspects.

The analysis try to provide information for government agencies and  health care companies to improve their services and environmental factors as well as help individuals to change their behaviors to get long life expectancy.
#### In this study, I want to solve following problems:  
1. What is the shape of the income–life expectancy gradient?
2. How are gaps in life expectancy changing over time?
3. How do the gaps vary across local areas?
4. What are the factors associated with differences in life expectancy?

## II. Dataset
The dataset include 14 csv files of data tables. After loading these tables into python notebook, we can have a whole picture of the dataset. It provide life expectancy of people with different gender, household income, in  different states, commuting zones, county during year 2001 to 2014. It also provide informations about fraction current smokers, fraction obese, percent uninsured, 30-day hospital mortality rate, percent of Medicare enrollees, percent religious, percent black, unemployment rate, labor force participation, population density and so on in commuting zones and county level. All these informations are important for the research.

The limitation of the dataset is that it doesn’t provide all the informations in each year during 2001 and 2014. So we can’t answer questions related with factors of life expectancy changing over time. 

## III. Data Wrangling
In data wrangling step, we first remove the unadjusted and Standard Error columns in the tables, then fill missing values in table 10 and table 12. 

There are 3 steps to fill missing values in table 10:   
* A column is removed if there are more than 10% missing value.
* A commuting zone is removed if all the values of a column are missing.
* Fill missing values with the mean value of that that commuting zone.

There are 3 steps to fill missing values in table 12:   
* A county is removed if all the values of a column are missing.
* A column is removed if there are more than 20% missing value.
* Fill missing values with the mean value of that that county.

After data wrangling, we save the tables in csv files table_1.csv ~ table_14.csv. 

The python code of above jobs is in file Capstone-1.ipynb

## IV. Exploratory Data Analysis

### 1. National Levels of Life Expectancy by Income

### 2. National Trends in Life Expectancy by Income in year 2001~2014

### 3. Local Area Variation in Life Expectancy gap by Income
#### Life Expectancy gap by State
#### Life Expectancy gap by Commuting Zone




## V. Data Modeling
In this step, we will find a model to predict average life expectancy of a county by factors associated with life expectancy. Before use machine learning algorithms to get the best model, we need to prepare dataset. We merged table 11 and 12 by County ID, got average values of life expectancy of each county, and caculated average values of Fraction Current Smokers, Fraction Obese, Fraction Exercised in Past 30 Days. We extracted 53 features related with life expectancy as *X*, average life expectancy of each county as *y*. 
```python
X.columns.values
```
```
array(['cty_pop2000', 'intersects_msa', 'cur_smoke', 'bmi_obese',
       'exercise_any', 'puninsured2010', 'reimb_penroll_adj10',
       'mort_30day_hosp_z', 'adjmortmeas_amiall30day',
       'adjmortmeas_chfall30day', 'adjmortmeas_pnall30day',
       'med_prev_qual_z', 'primcarevis_10', 'diab_hemotest_10',
       'diab_eyeexam_10', 'diab_lipids_10', 'mammogram_10',
       'amb_disch_per1000_10', 'cs00_seg_inc', 'cs00_seg_inc_pov25',
       'cs00_seg_inc_aff75', 'cs_race_theil_2000', 'gini99', 'poor_share',
       'inc_share_1perc', 'frac_middleclass', 'scap_ski90pcm', 'rel_tot',
       'cs_frac_black', 'cs_frac_hisp', 'unemp_rate', 'pop_d_2000_1980',
       'lf_d_2000_1980', 'cs_labforce', 'cs_elf_ind_man',
       'cs_born_foreign', 'mig_inflow', 'mig_outflow', 'pop_density',
       'frac_traveltime_lt15', 'hhinc00', 'median_house_value',
       'ccd_exp_tot', 'ccd_pup_tch_ratio', 'score_r', 'dropout_r',
       'cs_educ_ba', 'e_rank_b', 'cs_fam_wkidsinglemom', 'crime_total',
       'subcty_exp_pc', 'taxrate', 'tax_st_diff_top20'], dtype=object)
```

### 1. Machine Learning Models
We use three algorithms to get the machine learning models. They are Linear Regression, Support Vector Regression, and Random Forest Regressor. For each model, we adjust the parameters, caculate Coefficient of determination R^2 of the prediction and Mean squared error (MSE) to get the best model.

#### 1) Linear Regression
```python
from sklearn.linear_model import LinearRegression
X0 = X.copy()
# Creates a LinearRegression object
lm = LinearRegression()
lm.fit(X0, y)
```
**Evaluation result of the model:**

Model | Features | S^2 | MSE
--- | --- | --- | ---
LinearRegression()| 53 |0.8249|0.2469

True life expectancy compared to the predicted life expectancy is shown in the plot below:
![le-ple-lm](https://github.com/anxin16/Capstone-Project-2/blob/master/Figures/le-ple-lm.png)

**Predict test dataset with the model**
```python
from sklearn.model_selection import train_test_split
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X0, y, test_size=0.2, random_state = 5)
# Build a linear regression model using training data sets
lm = LinearRegression()
lm.fit(X_train, y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)
# Calculate the mean squared error
print("Fit a model X_train, and calculate MSE with y_train:", np.mean((y_train - pred_train) ** 2))
print("Fit a model X_train, and calculate MSE with X_test, y_test:", np.mean((y_test - pred_test) ** 2))
```
Fit a model X_train, and calculate MSE with y_train: 0.23989782683795785  
Fit a model X_train, and calculate MSE with X_test, y_test: 0.29637196828670975

*So this is a relatively good model. The only shortcoming of this model is that it uses all the features, so X has high demention. We'll try to reduce demention with feature selection later.*

#### 2) Support Vector Regression
```python
from sklearn.svm import SVR
X0 = X.copy()
# Support Vector Regression 
svr = SVR()
svr.fit(X0, y)
```
Here default kernel='rbf', we tried to use kernel=‘linear’ and kernel='poly', but after a long training time we didn't get any result, so we gave up.

We tried to adjust Penalty parameter C of the error term.
```python
# RBF model with C=2
svr1 = SVR(C=2)
svr1.fit(X0, y)
```
```python
# RBF model with C=5
svr2 = SVR(C=5)
svr2.fit(X0, y)
```
**Evaluation result of the model:**

Model | C | S^2 | MSE
--- | --- | --- | ---
SVR()|1.0|0.7835|0.3054
SVR(C=2)|2.0|0.9656|0.0486
SVR(C=5)|5.0|0.9932|0.0096

From the evaluation result, we can see that SVR with C=5 is the best model. It has the highest S^2 and the smallest MSE. But this is tested with training dataset. We need to use it on test dataset to make sure it's not overfitting.

True life expectancy compared to the predicted life expectancy of SVR(C=5) is shown in the plot below:
![le-ple-svr2](https://github.com/anxin16/Capstone-Project-2/blob/master/Figures/le-ple-svr2.png)

**Predict test dataset with the model**

```python
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X0, y, test_size=0.2, random_state = 5)
# Build a support vector regression model using training data sets
svr = SVR()
svr.fit(X_train, y_train)
pred_train = svr.predict(X_train)
pred_test = svr.predict(X_test)
# Calculate the mean squared error
print("Fit a model X_train, and calculate MSE with y_train:", np.mean((y_train - pred_train) ** 2))
print("Fit a model X_train, and calculate MSE with X_test, y_test:", np.mean((y_test - pred_test) ** 2))
```
Fit a model X_train, and calculate MSE with y_train: 0.3056692394741886  
Fit a model X_train, and calculate MSE with X_test, y_test: 1.4713715795892517

```python
# Build a support vector regression model using training data sets
svr3 = SVR(C=2)
svr3.fit(X_train, y_train)
pred_train3 = svr3.predict(X_train)
pred_test3 = svr3.predict(X_test)
# Calculate the mean squared error
print("Fit a model X_train, and calculate MSE with y_train:", np.mean((y_train - pred_train3) ** 2))
print("Fit a model X_train, and calculate MSE with X_test, y_test:", np.mean((y_test - pred_test3) ** 2))
```
Fit a model X_train, and calculate MSE with y_train: 0.049667362297045686  
Fit a model X_train, and calculate MSE with X_test, y_test: 1.472355197361246

```python
# Build a support vector regression model using training data sets
svr4 = SVR(C=5)
svr4.fit(X_train, y_train)
pred_train4 = svr4.predict(X_train)
pred_test4 = svr4.predict(X_test)
# Calculate the mean squared error
print("Fit a model X_train, and calculate MSE with y_train:", np.mean((y_train - pred_train4) ** 2))
print("Fit a model X_train, and calculate MSE with X_test, y_test:", np.mean((y_test - pred_test4) ** 2))
```
Fit a model X_train, and calculate MSE with y_train: 0.009595667888111285  
Fit a model X_train, and calculate MSE with X_test, y_test: 1.4732298835854551

*Comparing MSE of the predict results on test dataset, we found that SVR models had relatively same predicting MSE with different Penalty parameter C of the error term. SVR(C=5) has the lowest training MSE but the highest test MSE. So high C only lead to overfitting, it didn't improve the test result.*

#### 3) Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor
# Random Forest Regression 
rfr = RandomForestRegressor()
rfr.fit(X0, y)
```
Here we use model with default parameters. Next we'll tune the parameters of Random Forest model. One is max_features, the number of features to consider when looking for the best split. The other is n_estimators, the number of trees in the forest.
```python
# max_features = 20
rfr1 = RandomForestRegressor(max_features=20)
rfr1.fit(X0, y)
```
```python
# max_features = 10
rfr2 = RandomForestRegressor(max_features=10)
rfr2.fit(X0, y)
```
```python
# max_features = 5
rfr3 = RandomForestRegressor(max_features=5)
rfr3.fit(X0, y)
```
```python
# n_estimators = 20
rfr4 = RandomForestRegressor(n_estimators=20)
rfr4.fit(X0, y)
```
```python
# n_estimators = 100
rfr5 = RandomForestRegressor(n_estimators=100, oob_score=True)
rfr5.fit(X0, y)
```
```python
# n_estimators = 200
rfr6 = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=50)
rfr6.fit(X0, y)
```
**Evaluation result of the model:**

Model | max_features | n_estimators | S^2 | MSE
--- | --- | --- | --- | --- |
RandomForestRegressor()|53|10|0.9535|0.0656
RandomForestRegressor(max_features=20)|20|10|0.9506|0.0697
RandomForestRegressor(max_features=10)|10|10|0.9526|0.0668
RandomForestRegressor(max_features=5)|5|10|0.9497|0.0710
RandomForestRegressor(n_estimators=20)|53|20|0.9607|0.0555
RandomForestRegressor(n_estimators=100, oob_score=True)|53|100|0.9679|0.0452
RandomForestRegressor(n_estimators=200, oob_score=True, random_state=50)|53|200|0.9692|0.0434

From the evaluation result, we can see that RandomForestRegressor is very good model. Among them RandomForestRegressor with n_estimators=200 is the best model. It has the highest S^2 and the smallest MSE. But this is tested with training dataset. We need to use it on test dataset. 

True life expectancy compared to the predicted life expectancy of RandomForestRegressor(n_estimators=200, oob_score=True, random_state=50) is shown in the plot below:
![le-ple-rfr6](https://github.com/anxin16/Capstone-Project-2/blob/master/Figures/le-ple-rfr6.png)

**Predict test dataset with the model**

Model | MSE with training dataset | MSE with test dataset
--- | --- | ---
RandomForestRegressor()|0.0633|0.4030
RandomForestRegressor(max_features=20)|0.07655|0.3929
RandomForestRegressor(max_features=10)|0.0823|0.4107
RandomForestRegressor(max_features=5)|0.0736|0.4419
RandomForestRegressor(n_estimators=20)|0.0612|0.3746
RandomForestRegressor(n_estimators=100, oob_score=True)|0.0459|0.3526
RandomForestRegressor(n_estimators=200, oob_score=True, random_state=50)|0.0441|0.3623

*Comparing MSE of the predict results on test dataset, we found that Random Forest Regressor models had relatively same predicting MSE with different parameters. RandomForestRegressor(n_estimators=200, oob_score=True, random_state=50) is the best.*

### 2. Feature Selection Methods
Feature selection is the process of selecting a subset of relevant features for use in model construction. Feature selection techniques are used for four reasons:
* simplification of models to make them easier to interpret by users,
* shorter training times,
* to avoid the curse of dimensionality,
* enhanced generalization by reducing overfitting

In this project, X has 53 features all together. We use feature selection methods to reduce X's dimention while try to get as good models. 

#### 1) Principal Component Analysis
Principal component analysis (PCA) is a technique used to emphasize variation and bring out strong patterns in a dataset. It's often used to make data easy to explore and visualize. The number of principal components is less than or equal to the smaller of the number of original variables or the number of observations. 

In this project, we tried PCA in Linear Regression and Support Vector Regression, but we didn't see difference with SVR model, so we only discuss PCA in Linear Regression model here.
```python
from sklearn.decomposition import PCA
# Use Minka’s MLE to guess the dimension
# feature extraction
pca = PCA(n_components='mle', svd_solver='full')
X1 = pca.fit_transform(X) 
# summarize components
print('Number of Components:', pca.n_components_)
```
Number of Components: 52
```python
lm1 = LinearRegression()
lm1.fit(X1, y)
```
Here we use Minka’s MLE to guess the dimension and get 52 features out of all 53 features. This means that most of the features are relevant with life expectancy. Next we'll adjust the number of components in PCA and observe the quality of models.

**Evaluation result of the model:**

Model | Number of Components | S^2 | MSE
--- | --- | --- | --- |
PCA(n_components='mle', svd_solver='full')|52|0.8249|0.2470
PCA(n_components=20)|20|0.7643|0.3324
PCA(n_components=10)|10|0.6944|0.4311
PCA(n_components=5)|5|0.5536|0.6296

From the evaluation result, we can see that PCA method affect the quality of the models. PCA with different number of component can lead to different S^2 and MSE. The less the number of component is, the worse the model is. So if we banlance the number of component and model score, PCA with n_components=20 is relatively good.

**Predict test dataset with the model**

Model | MSE with training dataset | MSE with test dataset
--- | --- | ---
PCA(n_components='mle', svd_solver='full')|0.2399|0.2955
PCA(n_components=20)|0.3230|0.3850
PCA(n_components=10)|0.4314|0.4424
PCA(n_components=5)|0.6392|0.6044

*Comparing MSE of the predict results on test dataset, we found that PCA Linear regression models have different predicting MSE with different parameters. MSE with training dataset is close to MSE with test dataset, so there is no overfitting problem with PCA. Consider feature number and MSE, PCA with n_components=20 is relatively good.*

#### 2) Regularization
Regularization is a technique used to avoid the overfitting problem. It is a process of introducing additional information in order to prevent overfitting. Lasso is a Linear Model trained with L1 prior as regularizer. 

```python
from sklearn.linear_model import Lasso
X0 = X.copy()
las = Lasso()
las.fit(X0, y)
```
This is the default model. We can adjust parameter alpha，constant that multiplies the L1 term, to get the best model.

**Evaluation result of the model:**

Model | alpha |Number of non-zero coefficients| S^2 | MSE
--- | --- | --- | --- | --- |
Lasso()|1.0|13|0.6733|0.4608
Lasso(alpha=0.1)|0.1|16|0.7632|0.3340
Lasso(alpha=0.01)|0.01|26|0.7847|0.3037
Lasso(alpha=0.001)|0.001|35|0.8127|0.2642

From the evaluation result, we can see that alpha affect the quality of the models. The smaller alpha is, the better the model is. So Lasso with alpha=0.001 has highest score and lowest MSE.

**Predict test dataset with the model**

Model | MSE with training dataset | MSE with test dataset
--- | --- | ---
Lasso()|0.4554|0.4569
Lasso(alpha=0.1)|0.3248|0.3764
Lasso(alpha=0.01)|0.2971|0.3470
Lasso(alpha=0.001)|0.2594|0.2980

*From the predict results on test dataset, we found that Lasso models have different predicting MSE with different parameters. There is no overfitting problem with Lasso. MSE with training dataset is close to MSE with test dataset. Lasso with alpha=0.001 is the best model.*

#### 3) Random Forests
Random forests are among the most popular machine learning methods thanks to their relatively good accuracy, robustness and ease of use. They are often used for feature selection. The reason is because the tree-based strategies used by random forests naturally ranks by how well they improve the purity of the node. 

We can select features by compute the feature importances with Random Forests model. Here we check with the best Random forests model discussed before.
```python
# n_estimators = 200
rfr6 = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=50)
rfr6.fit(X0, y)
fi6 = pd.DataFrame(list(zip(X0.columns, rfr6.feature_importances_)), columns = ['features', 'Importance'])
fi6.sort_values(by='Importance', ascending=False).head(10)
```
The 10 most important features are as below:  
![rfr6_Importance](https://github.com/anxin16/Capstone-Project-2/blob/master/Figures/rfr6_Importance.png) 

### 3. Factors Affect Life Expectancy
We have two methods to determine the features that affect life expectancy most. One is Regularization with Lasso model, we can determine the important features with its non-zero coefficients. The other method is Random Forests Regressor model, we can decide the importance of features by its attribute feature_importances_. Here we use the best models discussed above.

#### 1) Result of Regularization with Lasso model
```python
# Lasso with alpha=0.001
las3 = Lasso(alpha=0.001)
las3.fit(X0, y)
print('Estimated intercept:', las3.intercept_)
las3_coef = pd.DataFrame(list(zip(X0.columns, las3.coef_)), columns = ['features', 'Coefficients'])
lc3 = las3_coef[las3_coef['Coefficients']!=0]
print('Number of non-zero coefficients:', len(lc3))
lc3.reindex(lc3.Coefficients.abs().sort_values(ascending = False).index)
```
Estimated intercept: 78.0900031141  
Number of non-zero coefficients: 35  
![f-coef-las3](https://github.com/anxin16/Capstone-Project-2/blob/master/Figures/f-coef-las3.png)   
(This picture is truncated because it's too long for screenshot.)

From result of Lasso model, we get the 10 most important features that affect life expectancy:

No. | features | Feature Description | Coefficients
--- | --- | --- | ---
1|cs_fam_wkidsinglemom|Fraction of Children with Single Mother|-2.531824
2|cur_smoke|Fraction Current Smokers|-2.377746
3|poor_share|Poverty Rate|1.753844e
4|cs_labforce|Labor Force Participation|-0.911347
5|frac_traveltime_lt15|Fraction with Commute < 15 Min|-0.586644
6|gini99|Gini Index Within Bottom 99%|0.4624287
7|cs_elf_ind_man|Share Working in Manufacturing|0.405938
8|lf_d_2000_1980|Percent Change in Labor Force 1980-2000|0.275496
9|cs_race_theil_2000|Racial Segregation|0.208809
10|mort_30day_hosp_z|30-day Hospital Mortality Rate Index|-0.140977

#### 2) Result of Random Forests Regressor model
From result of Random Forests Regressor model, we get the 10 most important features that affect life expectancy:

No. | features | Feature Description | Importance
--- | --- | --- | ---
1|median_house_value|Median House Value|0.244180
2|med_prev_qual_z|Mean of Z-Scores for Dartmouth Atlas Ambulatory Care Measures|0.131244
3|cur_smoke|Fraction Current Smokers|0.098932
4|cs_educ_ba|Percent College Grads|0.076523
5|puninsured2010|Percent Uninsured|0.074873
6|e_rank_b|Absolute Mobility (Expected Rank at p25)|0.037186
7|reimb_penroll_adj10|Medicare $ Per Enrollee|0.035520
8|bmi_obese|Fraction Obese|0.035098
9|mammogram_10|Percent Female Aged 67-69 with Mammogram|0.024484
10|amb_disch_per1000_10|Discharges for Ambulatory Care Sensitive Conditions Among Medicare Enrollees|0.019673

#### 3) Factors affect life expectancy of the lowest quartile
Next we do research on the factors that affect life expectancy of the lowest house income quartile people (Q1). The dataset include features relevant with Q1. We'll try Lasso and Random Forests Regressor.

```python
# Regularization with Lasso model
las_Q1 = Lasso(alpha=0.001)
las_Q1.fit(X, y)
print('Coefficient of determination R^2 of the prediction:', las_Q1.score(X,y))
print('Mean squared error (MSE):', np.mean((y-las_Q1.predict(X))**2))
```
Coefficient of determination R^2 of the prediction: 0.555170598503  
Mean squared error (MSE): 0.6240216768129673

The R^2 score of this model is not high. So we won't consider this model.

```python
# Random Forests Regressor
rfr_Q1 = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=50)
rfr_Q1.fit(X, y)
print('Coefficient of determination R^2 of the prediction:', rfr_Q1.score(X,y))
print('Mean squared error (MSE):', np.mean((y-rfr_Q1.predict(X))**2))
```
Coefficient of determination R^2 of the prediction: 0.929690144901  
Mean squared error (MSE): 0.0986330344350064

This is a very good model. We'll use this model for feature selection.
```python
fi_Q1 = pd.DataFrame(list(zip(X.columns, rfr_Q1.feature_importances_)), columns = ['features', 'Importance'])
fi_Q1.sort_values(by='Importance', ascending=False).head(10)
```
![f-imp-rfrQ1](https://github.com/anxin16/Capstone-Project-2/blob/master/Figures/f-imp-rfrQ1.png) 

Below are the 10 most important features that affect life expectancy of the lowest household income quartile people:

No. | features | Feature Description | Importance
--- | --- | --- | ---
1|median_house_value|Median House Value|0.140114
2|reimb_penroll_adj10|Medicare $ Per Enrollee|0.088326
3|cur_smoke_q1|Fraction Current Smokers in Q1|0.057331
4|cs_frac_black|Percent Black|0.044545
5|mammogram_10|Percent Female Aged 67-69 with Mammogram|0.033224
6|amb_disch_per1000_10|Discharges for Ambulatory Care Sensitive Conditions Among Medicare Enrollees|0.028584
7|med_prev_qual_z|Mean of Z-Scores for Dartmouth Atlas Ambulatory Care Measures|0.024302
8|adjmortmeas_pnall30day|30-day Mortality for Pneumonia|0.023432
9|frac_middleclass|Fraction Middle Class (p25-p75)|0.020439
10|lf_d_2000_1980|Percent Change in Labor Force 1980-2000|0.018807

From the research result, we can see that features affect the lowest income people are not totally same with that of all people. Several features like Percent Black, Fraction Middle Class (p25-p75), Percent Change in Labor Force 1980-2000 are more important for lowest income people.

## VI. Results and Discussion

## VII. Future Work

# Reference


























