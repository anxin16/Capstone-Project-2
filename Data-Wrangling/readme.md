## Data Wrangling
### Step 1: Read original csv files to pandas dataframe
Use pd.read_csv() to read all the csv files of the dataset.  
For table 12, there was error message at first. Then I add "encoding='latin-1'" as parameter of pd.read_csv and it's Ok.

### Step 2: Choose and keep useful columns for each table
For each table, remove the Unadjusted and Standard Error columns, keep Q1 or V1 columns.   
I found useful columns and Subset Variables (Columns).Then save new tables as csv files with to_csv().

### Step 3: Fill missing values in table 10 and table 12
A column is removed if there are more than 20% missing value. A state is removed if all the values of a column are missing.  
Then I caculate mean of each satae and fill missing values with the mean value of that state.

#### Please check Health-Wealth1.ipynb for code details and all the csv files are Data Wrangling results.
