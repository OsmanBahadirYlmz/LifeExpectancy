# libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns


# In[1] 1. Reading data from a formatted file********
# https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
data=pd.read_csv("Life Expectancy Data.csv") #orginal data without filling
df = pd.read_csv("Life Expectancy Data.csv")


# data preprocessing
mean_by_country = df.groupby('Country').transform('mean')



#Filling data with mean of each country
for column in df.columns:
    try:
        if df[column].dtype != object:  # Exclude non-numeric columns
            df[column] = df[column].fillna(mean_by_country[column])
    except:
        continue

#filling the rest of data with avarage
df = df.fillna(data.mean())

print(df.isna().sum())


# In[2] 2.Descriptive statistics of the data
byCountry = df.groupby("Country").mean()
byCountry2 = df.groupby("Country").describe()


# Life expectancy
x_value = byCountry.iloc[:, 1]
plt.title("Histogram of Life Expectancy")
his = plt.hist(x_value, bins=10)
plt.xlabel("Years")
plt.ylabel("Counts")
plt.show()



# corolation between values

"""
coomment about coralation- in theese matrix it can be said that,
Schooling,Income composition of resources, BMI has direct coralation,
moreover, Adult Mortality and HIV/AIDS has negative coralation.
"""
coralation_matrix = df.corr()
coralation_matrix2 = data.corr()['Life expectancy '].sort_values()
coralation_matrix2.drop(labels=["Life expectancy "], axis=0, inplace=True)
plt.title("Corelation Table For Life Expectancy")
plt.bar(coralation_matrix2.index, coralation_matrix2[:])
plt.xticks(rotation=90)
plt.show()


# life expectancy vs features plots
for i in range(18):
    x_value = byCountry.iloc[:, 1]
    y_value = byCountry.iloc[:, i+2]
    plt.figure(i)
    plt.title("life Expectancy vs {}".format(byCountry.columns[i+2]))
    plt.xlabel("Life Expectancy")
    plt.ylabel(byCountry.columns[i+2])
    plt.scatter(x_value, y_value)
    plt.show()
"""
about above figures it can be inferred that
1-there is an inverse ratio between the adult mortality and life expactancy.
2-there is not linear corelation between expenditure on health. On the other hand
when percentage of expenditure obove 1500's, life expectancy rises proportionaly (dramatic increase)
3-Higher palio inmune makes life expantacy higher
4-diphttheria and life expactancy rises accordingly
5-HIV deaths drastically lower life expectancy 
6-above 10000 GDP increases life expactancy
7-thinness certainly has the effect of lowering life expectancy
8-there is direct corelation both income and schooling between life expactancy

"""
