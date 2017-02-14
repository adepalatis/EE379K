
#PROBLEM2
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('/Users/shammakabir/Downloads/CorrMat1.csv')
#data = pd.read_csv('/Users/shammakabir/Downloads/CorrMat3.csv') switch between to check 

pca = PCA(n_components=2)
results = pca.fit(data).transform(data)
r_df = pd.DataFrame(results)
print(r_df)
first = r_df.iloc[:,0]
second = r_df.iloc[:,1]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(first, second)

index = []
#find indexes of outliers 
for i in range(99):
    if second[i] >= 15000.0: 
        index.append(i)
    if second[i] <= -6500.0:
        index.append(i)
print(index)

#find which ones are more than 2 std away from col_mean
#create new data frame
fixed_df = pd.DataFrame()
mean = [] 
for i in range(99):
    temp = data.iloc[:,i]
    mean.append(temp.mean())
print(mean)
std = []
for i in range(99):
    temp = data.iloc[:,i]
    std.append(temp.std())
print(std)
for i in range(99):
    temp = data.iloc[:,i]
    for ind in index:
        if temp[ind] > 2*std[i]:
            temp[ind] = mean[i]
        if temp[ind] < 2*std[i]:
            temp[ind] = mean[i]
    fixed_df[i] = temp

fixed_df.replace(0, np.nan)
fixed_df = fixed_df.fillna(fixed_df.mean())
fixed_df.to_csv('/Users/shammakabir/Desktop/lolol.csv')

pca2 = PCA(n_components=2)
results1 = pca2.fit(fixed_df).transform(fixed_df)
r_df1 = pd.DataFrame(results1)
print(r_df1)
first1 = r_df.iloc[:,0]
second2 = r_df.iloc[:,1]
fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.scatter(first1, second2)
