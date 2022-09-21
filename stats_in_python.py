####3.1.1. Data representation and interaction

###3.1.1.2 The pandas data-frame
##Creating dataframes: reading data files or converting arrays

#Reading from a CSV File
#Separator: It is a CSV file, but the separator is “;”

import pandas as pd
data = pd.read_csv('descriptives-scipy/Data/brain_size.csv', sep=';', na_values=".")

#Creating from arrays
import numpy as np
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)
pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})

#Manipulating data
data.shape    # 40 rows and 8 columns
data.columns  # It has columns   
print(data['Gender'])  # Columns can be addressed by name   

# Simpler selector
data[data['Gender'] == 'Female']['VIQ'].mean()

#For a quick view on a large dataframe, use its describe method: 
#pd.DataFrame.describe()

#groupby: splitting a dataframe on values of categorical variables:
groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))
groupby_gender.mean()

#Plotting data
#Pandas comes with some plotting tools to display statistics of the data in dataframes:
from pandas.plotting import scatter_matrix
pd.plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])
pd.plotting.scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])   

####3.1.2. Hypothesis testing: comparing two groups

from scipy import stats

###3.1.2.1. Student’s t-test: the simplest statistical test
##1-sample t-test: testing the value of a population mean
stats.ttest_1samp(data['VIQ'], 0)

##2-sample t-test: testing for difference across populations
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq) 


###3.1.2.2. Paired tests: repeated measurements on the same individuals
##2 sample test:
stats.ttest_ind(data['FSIQ'], data['PIQ'])  
##“paired test”, or “repeated measures test
stats.ttest_rel(data['FSIQ'], data['PIQ']) 
##Wilcoxon signed-rank test
stats.wilcoxon(data['FSIQ'], data['PIQ']) 
##Mann–Whitney U test
stats.mannwhitneyu(data['FSIQ'], data['PIQ'])

####3.1.3. Linear models, multiple factors, and analysis of variance

###3.1.3.1. “formulas” to specify statistical models in Python
##A simple linear regression

x = np.linspace(-5, 5, 20)
np.random.seed(1)
# normal distributed noise
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
# Create a data frame containing all the relevant variables
data = pd.DataFrame({'x': x, 'y': y})
from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit()
print(model.summary())

##Categorical variables: comparing groups or multiple categories
data = pd.read_csv('descriptives-scipy/Data/brain_size.csv', sep=';', na_values=".")
model = ols("VIQ ~ Gender + 1", data).fit()
print(model.summary())

##Link to t-tests between different FSIQ and PIQ
data_fisq = pd.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pd.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pd.concat((data_fisq, data_piq))
print(data_long) 
model = ols("iq ~ type", data_long).fit()
print(model.summary())  
stats.ttest_ind(data['FSIQ'], data['PIQ']) 

###3.1.3.2. Multiple Regression: including multiple factors
data = pd.read_csv('descriptives-scipy/Data/iris.csv')
model = ols('sepal_width ~ name + petal_length', data).fit()
print(model.summary())


###3.1.3.3. Post-hoc hypothesis testing: analysis of variance (ANOVA)
print(model.f_test([0, 1, -1, 0]))

####3.1.4. More visualization: seaborn for statistical exploration
print(data) 

###3.1.4.1. Pairplot: scatter matrices

names = [
    'EDUCATION: Number of years of education',
    'SOUTH: 1=Person lives in South, 0=Person lives elsewhere',
    'SEX: 1=Female, 0=Male',
    'EXPERIENCE: Number of years of work experience',
    'UNION: 1=Union member, 0=Not union member',
    'WAGE: Wage (dollars per hour)',
    'AGE: years',
    'RACE: 1=Other, 2=Hispanic, 3=White',
    'OCCUPATION: 1=Management, 2=Sales, 3=Clerical, 4=Service, 5=Professional, 6=Other',
    'SECTOR: 0=Other, 1=Manufacturing, 2=Construction',
    'MARR: 0=Unmarried,  1=Married',
]
short_names = [n.split(':')[0] for n in names]
data = pd.read_csv('descriptives-scipy/Data/CPS_85_Wages',sep=None, engine='python', skiprows=27, skipfooter=6, header=None)
data.columns = short_names
import seaborn
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg') 

##Categorical variables can be plotted as the hue:
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg', hue='SEX')

#Look and feel and matplotlib settings
from matplotlib import pyplot as plt
plt.rcdefaults()

###3.1.4.2. lmplot: plotting a univariate regression
seaborn.lmplot(y='WAGE', x='EDUCATION', data=data) 

##Robust regression
##To compute a regression that is less sentive to outliers, one must use a robust model
#statsmodels.formula.api.rlm()

####3.1.5. Testing for interactions
#from statsmodels.formula.api import ols
import statsmodels as sm
from statsmodels.api import ols
 
result = data.ols(formula='WAGE ~ EDUCATION + GENDER + EDUCATION * GENDER',
                data=data).fit()    
print(result.summary())

