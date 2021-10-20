# Databricks notebook source
# MAGIC %md
# MAGIC In this model we will use a latent factors model, a popular approach to [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering#:~:text=In%20the%20newer%2C%20narrower%20sense,from%20many%20users%20(collaborating).
# MAGIC 
# MAGIC We will be using a truncated [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition), because unlike a standard SVD, a truncated SVD produces a factorization where the number of columns can be specifies for a number of truncation, saving memory and processing power.
# MAGIC 
# MAGIC The sk-learn package has a module to build a truncated SVD. 

# COMMAND ----------

import pandas as pd
import numpy as np
import pickle

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# COMMAND ----------

#Read in data from a pickle file
df = pd.read_pickle("beer_data.pickle")

#Shorten the dataset for testing purposes
df = df.head(500000)

# COMMAND ----------

# MAGIC %md
# MAGIC We create a sparse pivot table which groups together the data in a meaningful way so we can train it later.
# MAGIC In this case, we want out pivot table to contain the users, the item they rated, and the rating value. The rest of the values will be filled with 0s.

# COMMAND ----------

# create a sparse pivot table
df_pivot = df.pivot_table(index='review_profilename', columns='beer_name', values='review_overall').fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC Next we need to determine the number of components we want to use for our truncated SVD.
# MAGIC 
# MAGIC We decide this value by using Catell's [scree test](https://en.wikipedia.org/wiki/Scree_plot). We'll plot the eigenvalues of different component values, and determine the number of components that would best suit our TSVD.

# COMMAND ----------

T = df_pivot.values.T
T.shape

# COMMAND ----------

def explained_variance(list_n_components):
    
    out = []
    
    for num in list_n_components:
        SVD = TruncatedSVD(n_components=num,random_state=num)
        SVD.fit_transform(T)
        evar = np.sum(SVD.explained_variance_ratio_)
        t = (num,evar)
        out.append(t)
    
    return out

# COMMAND ----------

n = [50,100,150,200,250,300,400,600,700,1000,1200,1400,1500]
exp_var = explained_variance(n)

# COMMAND ----------

x,y = zip(*exp_var)
plt.plot(x, y)

# COMMAND ----------

# MAGIC %md
# MAGIC Looking at the plot, we see that the number of components starts to level off at around 600 components.
# MAGIC 
# MAGIC So, we will create a TSVD with 600 components.

# COMMAND ----------

SVD = TruncatedSVD(n_components=600,random_state=600)
matrix = SVD.fit_transform(T)

# COMMAND ----------

#Create the correlation matrix
corr = np.corrcoef(matrix)

# COMMAND ----------

#Put all the beer names in a list
beer_names = df_pivot.columns
beer_names_list = list(beer_names)

# COMMAND ----------

#Takes in the name of the beer and returns the top n nunber of recommended beers

def beer_recs(beer_name, n):

    beer_idx = beer_names_list.index(beer_name)
    
    sim_idx = corr[beer_idx] #Get the similararity index of the input beer

    #Create a list of tuples (beer name, correlation coefficient)
    similar = []    
    for idx, coeff in enumerate(sim_idx):
        similar.append((beer_names_list[idx],coeff))
    
    similar.sort(key=lambda x: x[1], reverse=True)
    
    out = []
    
    for i in range(1,n+1):
        out.append(similar[i][0])
        
    return out

# COMMAND ----------

beer_recs('One Hop Wonder Version 12',5)

# COMMAND ----------

beer_recs('Lift Ticket Winter Ale',5)

# COMMAND ----------

beer_recs('Totem Pale',5)

# COMMAND ----------

beer_recs('Coors',5)

# COMMAND ----------


