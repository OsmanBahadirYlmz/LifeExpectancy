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
print(df.isna().sum()) #there is no null in df

duplicate_rows = df[df.duplicated()]
print(duplicate_rows) #there is no duplicate

# #OUTLIERS
# for column in df.columns:
#     z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    
#     # Define a threshold (e.g., z-score > 3) to identify outliers
#     threshold = 3
    
#     # Find outliers
#     outliers = df[z_scores > threshold]
    
#     # Print the outliers
#     print(outliers)



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
correlation_matrix = df.corr()
correlation_matrix2 = data.corr()['Life expectancy '].sort_values()
correlation_matrix2.drop(labels=["Life expectancy "], axis=0, inplace=True)
plt.title("Corelation Table For Life Expectancy")
plt.bar(correlation_matrix2.index, correlation_matrix2[:])
plt.xticks(rotation=90)
plt.show()

# Heatmap
plt.figure(figsize=(10, 8)) 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Heatmap just for life expectancy
plt.figure(figsize=(10, 8))
sorted_corr_matrix = correlation_matrix[['Life expectancy ']].sort_values(by='Life expectancy ',ascending=False)  
sns.heatmap(sorted_corr_matrix, annot=True, cmap='coolwarm')
plt.title('Life Expectancy Heatmap')
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
df.iloc[:,3:].apply(pd.to_numeric,errors='coerce')
df['Country']=df['Country'].astype('category').cat.codes
df['Status']=df['Status'].astype('category').cat.codes

df.corr()