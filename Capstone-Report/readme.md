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

## III. Data Wrangling

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
feature selection is the process of selecting a subset of relevant features for use in model construction. Feature selection techniques are used for four reasons:
* simplification of models to make them easier to interpret by users,
* shorter training times,
* to avoid the curse of dimensionality,
* enhanced generalization by reducing overfitting

In this project, X has 53 features all together. We use feature selection methods to reduce X's dimention while try to get as good models. 

#### 1) Principal Component Analysis

#### 2) Regularization

#### 3) Random Forests

### 3. Factors Affect Life Expectancy






## VI. Results and Discussion

## VII. Future Work

# Reference


























