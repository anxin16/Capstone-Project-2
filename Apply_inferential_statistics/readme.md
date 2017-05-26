## Apply Inferential Statistics

To apply inferential statistics in my capstone project, I performed the following steps:

### Step 1: Read original csv files to pandas dataframe
Use pd.read_csv() to read csv files of table 1 and table 2. These tables have been cleaned after data wrangling.

### Step 2: Check the difference of life expectancy between males and females
In this step, I used null hypothesis method to check the difference of life expectancy between males and females.  

![boxplot](https://github.com/anxin16/Capstone-Project-2/blob/master/Figures/boxplot.png)

From the boxplot of the life expectancy of males and feles, we can see that the mean life expectancy of males is lower than that of females. But there's overlap for boxplots of males and females. We were not sure if the observed result was true for all the population. So I setup an hypothesis test to determine.

The null hypothesis is: Mean life expectancy of males are same as mean life expectancy of females. If the null hypothesis is correct, mean life expectancy of males equals to mean life expectancy of females and sample mean difference of males and females equals to zero. The alternative hypothesis is that mean life expectancy of males is lower than mean life expectancy of females.

I caculated the confidence interval of α = .01 and get the result that if null hypothesis is correct, we have 99% confidence that the difference of mean life expectancy between males and females is less than 0.92 . But the actural sample mean difference is 3.66, the possibility of getting this mean difference under H0 is nearly 0. So we strongly reject the null hypothesis. Mean life expectancy of females is higher that mean life expectancy of males. 

### Step 3: Research on the relationship between Life Expectancy and Household Income
__First, I made a scatter plot of Life Expectancy and Household Income as below:__

![scatter1](https://github.com/anxin16/Capstone-Project-2/blob/master/Figures/scatter1.png)

__second, I caculated correlation coefficient and r-squared of two lines in above scatter plot__

From the plot we can see that there's obvious relationship between Life Expectancy and Household Income. Because males' samples and females' sample are in different lines, I caculated their correlation coefficient and r-squared respectively.

![scatter2](https://github.com/anxin16/Capstone-Project-2/blob/master/Figures/scatter2.png)

For males sample,  
correlation coefficient: 0.990890962077  
r-squared: 0.981864898726  

![scatter3](https://github.com/anxin16/Capstone-Project-2/blob/master/Figures/scatter3.png)

For females sample,  
correlation coefficient: 0.986086324097  
r-squared: 0.972366238571  

__Finally, I get conclusion from the research__

Life expectanct has big relationship with household income for both males and females.
