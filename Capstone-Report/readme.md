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




## V. Data Modelingthe 
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
LinearRegression()|53|0.8249|0.2469

**Predice test dataset with the model**
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

#### 2) Support Vector Regression
```python

```

#### 3) Random Forest Regressor
```python
```

### 2. Feature Selection Methods

#### 1) Principal Component Analysis

#### 2) Regularization

#### 3) Random Forests

### 3. Factors Affect Life Expectancy






## VI. Results and Discussion

## VII. Future Work

# Reference


























