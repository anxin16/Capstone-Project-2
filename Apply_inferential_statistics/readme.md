## Apply Inferential Statistics

To apply inferential statistics in my capstone project, I performed the following steps:

### Step 1: Read original csv files to pandas dataframe
Use pd.read_csv() to read csv files of table 1 and table 2. These tables have been cleaned after data wrangling.

### Step 2: Check the difference of life expectancy between males and females
In this step, I used null hypothesis method to check the difference of life expectancy between males and females.  

From the hitogram and boxplot of the life expectancy of males and feles, we can see that the mean life expectancy of males is lower than that of females. But there's overlap for boxplots of males and females. We were not sure if the observed result was true for all the population. So I setup an hypothesis test to determine.

The null hypothesis is: Mean life expectancy of males are same as mean life expectancy of females. If the null hypothesis is correct, mean life expectancy of males equals to mean life expectancy of females; the alternative hypothesis is that mean life expectancy of males is lower than mean life expectancy of females.


