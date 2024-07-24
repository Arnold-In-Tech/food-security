#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ### Import data: (Features = Independent Variables, Lables = Dependent)

# In[2]:


#Food security metrics
fies_A = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/regr/fies_r.csv")
fies_A = fies_A.iloc[:,1:]
print(fies_A.head())

fies_B = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/regr/FIES_se_B_r.csv")
fies_B = fies_B.iloc[:,1:]
fies_B = fies_B.replace(np.nan,0)  #replace nan with zero
print(fies_B.head())


# In[3]:



#A: Demographics
## Age
A_Age = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/regr/A_Age_r.csv")
A_Age = A_Age.iloc[:,1:]

## Gender
A_gender = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/regr/A_gender_r.csv")
A_gender = A_gender.iloc[:,1:]

## HH size
A_HH_size = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/regr/A_HH_size_r.csv")
A_HH_size = A_HH_size.iloc[:,1:]

## years of schooling
A_Schooling_Years = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/regr/A_Schooling_Years_r.csv")
A_Schooling_Years = A_Schooling_Years.iloc[:,1:]


#=====
#B: Sorghum FarmSize
B_S_FarmSize = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/regr/B_S_FarmSize_r.csv")
B_S_FarmSize = B_S_FarmSize.iloc[:,1:]


#=====
#C: Contract
C_contracted = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/regr/C_contracted_r.csv")
C_contracted = C_contracted.iloc[:,1:]

# M_Participation
C_M_participation = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/regr/C_M_participation_r.csv")
C_M_participation = C_M_participation.iloc[:,1:]


# D: coop_member_r
D_coop_member = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/regr/D_coop_member_r.csv")
D_coop_member = D_coop_member.iloc[:,1:]

# E: E_sorghum_sold_r
E_sorghum_sold = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/regr/E_sorghum_sold_r.csv")
E_sorghum_sold = E_sorghum_sold.iloc[:,1:]


# In[4]:


#Combine dataframes
features = pd.concat([A_Age, 
                    A_gender, 
                    A_HH_size, 
                    A_Schooling_Years, 
                    B_S_FarmSize, 
                    C_contracted, 
                    C_M_participation, 
                    D_coop_member, 
                    E_sorghum_sold], 
                   axis=1, 
                   join="inner")
features.head()


# ### Merge Food Security categories (# "Chronic" and "Occasional" to Zero, "Break-even" and "Food surplus" to One)

# In[5]:


# Covert labels to binary
# Chronic and Occasional to Zero, 
# "Break-even" and "Food surplus" to 1
# Slide the categories columns into one array/collumn

# Merge "Chronic" and "occasional"
fies_B['FIES_status_0']= fies_B[["Chronic", "Occasional"]].sum(axis=1)

# Merge "Break-even" and "Food surplus"
fies_B['FIES_status_1']= fies_B[["Break-even", "Food surplus"]].sum(axis=1)
fies_B

#Replace 1 and 2 with zero 
fies_B['FIES_status_0'] = fies_B['FIES_status_0'].replace(1,0)  #replace nan with zero
fies_B['FIES_status_0'] = fies_B['FIES_status_0'].replace(2,0)  #replace nan with zero

#Replace 3 and 4 with zero 
fies_B['FIES_status_1'] = fies_B['FIES_status_1'].replace(3,1)  #replace nan with zero
fies_B['FIES_status_1'] = fies_B['FIES_status_1'].replace(4,1)  #replace nan with zero


# Merge  "Chronic", "occasional", "Break-even" and "Food surplus" into one array with zeros and ones, 
# and overwrite the fies status column

fies_B['FIES_status']= fies_B[['FIES_status_0', 'FIES_status_1']].sum(axis=1)

fies_B = fies_B.iloc[:,-1:]

label = fies_B


# ## 1. Test for Multicollinearity (Pearson's Correlation)

# In[6]:


df_all = pd.concat([label,
                    features], 
                   axis=1, 
                   join="inner")
df_all.head()


# In[7]:


# storing into the excel file
#df_all.to_excel("output_data.xlsx")


# In[8]:


from matplotlib import pyplot as plt
import seaborn as sns

correlation = df_all.corr()
#Export correlation matrix
#correlation.to_csv("./correlation_.csv", sep=',', encoding='utf-8')

plt.rcParams.update({'font.size': 14})

plt.figure(figsize=(16, 5))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-0.65, vmax=0.65, cmap="RdBu_r")
#Export heatmap
#plt.savefig('Correlation.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=heatmap.get_facecolor())


# #### >> Pearsons Correlation shows "Contract", "Coop-membership" & "Sorghum sold" exert high impact on Food Security (They show poor correlation)

# ## 2. Pearsons Chi-Square Test
# ### Method 1

# In[9]:


import scipy.stats as stats

# Data to be used for chi-square calculation looks like 

observed = features

observed.head()  #Prints the first five lines of the table


# ##### Tests whether respondents (0-207) vary based on the demographic/Vertical/Horizontal factors (Age, Gender,...Sorghum sold)
# #### Hypotheses
#     ##### H0: (null hypothesis) The two variables are independent.
#     ##### H1: (alternative hypothesis) The two variables are not independent.

# In[10]:



#Perform the Chi-Square Test of Independence 

stat, p, dof, expected = stats.chi2_contingency(observed= observed)

# stat is the chi-square statistic
# p is the p-value
# dof is the degree of freedom
# The expected values for each cell in the contingency table

print("Chi-square statistic = ",stat)
print("P-value = ", p)
print("Degree of freedom = ", dof)
print("Expected values = \n", expected)


# #
# #### Since the p-value (1.0) of the test is not less than 0.05, we fail to reject the null hypothesis. This means we do not have sufficient evidence to say that there is an association between Respondents and Demographic/Vertical/Horizontal factors.
# 
# #### In other words, Respondents and Demographic/Vertical/Horizontal factors are independent.
# #

# ### Method 2

# In[11]:


df_all.head()  #complete table (including features (independent variables) and label (fies status))


# #### (i) Age vs FIES status
# 

# In[12]:


temp=df_all.groupby(['FIES_status', 'Age'])['Age'].agg(['count']).reset_index()

a_v_f = pd.pivot_table(temp,values='count',index='Age',columns='FIES_status', fill_value=0)

a_v_f


# In[13]:


#Perform the Chi-Square Test of Independence 

stat, p, dof, expected = stats.chi2_contingency(observed= a_v_f)

print(">> Age vs FIES status")
print("Chi-square statistic = ",stat)
print("P-value = ", p)
print("Degree of freedom = ", dof)
print("Expected values = \n", expected)


# #### (ii) Gender vs FIES status
# 

# In[14]:


temp=df_all.groupby(['FIES_status', 'Gender'])['Gender'].agg(['count']).reset_index()

g_v_f = pd.pivot_table(temp,values='count',index='Gender',columns='FIES_status', fill_value=0)

g_v_f


# In[15]:


#Perform the Chi-Square Test of Independence 

stat, p, dof, expected = stats.chi2_contingency(observed= g_v_f)

print(">> Gender vs FIES status")
print("Chi-square statistic = ",stat)
print("P-value = ", p)
print("Degree of freedom = ", dof)
print("Expected values = \n", expected)


# #### (iii) HouseholdSize vs FIES status
# 

# In[16]:


temp=df_all.groupby(['FIES_status', 'Household_Size'])['Household_Size'].agg(['count']).reset_index()

hh_v_f = pd.pivot_table(temp,values='count',index='Household_Size',columns='FIES_status', fill_value=0)

hh_v_f


# In[17]:


#Perform the Chi-Square Test of Independence 

stat, p, dof, expected = stats.chi2_contingency(observed= hh_v_f)

print(">> Household_Size vs FIES status")
print("Chi-square statistic = ",stat)
print("P-value = ", p)
print("Degree of freedom = ", dof)
print("Expected values = \n", expected)


# #### (iv) Schooling_Years vs FIES status
# 

# In[18]:


temp=df_all.groupby(['FIES_status', 'Schooling_Years'])['Schooling_Years'].agg(['count']).reset_index()

sy_v_f = pd.pivot_table(temp,values='count',index='Schooling_Years',columns='FIES_status', fill_value=0)

sy_v_f


# In[19]:


#Perform the Chi-Square Test of Independence 

stat, p, dof, expected = stats.chi2_contingency(observed= sy_v_f)

print(">> Schooling_Years vs FIES status")
print("Chi-square statistic = ",stat)
print("P-value = ", p)
print("Degree of freedom = ", dof)
print("Expected values = \n", expected)


# #### (v) S_FarmSize vs FIES status
# 

# In[20]:


temp=df_all.groupby(['FIES_status', 'S_FarmSize'])['S_FarmSize'].agg(['count']).reset_index()

sfs_v_f = pd.pivot_table(temp,values='count',index='S_FarmSize',columns='FIES_status', fill_value=0)

sfs_v_f


# In[21]:


#Perform the Chi-Square Test of Independence 

stat, p, dof, expected = stats.chi2_contingency(observed= sfs_v_f)

print(">> S_FarmSize vs FIES status")
print("Chi-square statistic = ",stat)
print("P-value = ", p)
print("Degree of freedom = ", dof)
print("Expected values = \n", expected)


# #### (vi) Contract vs FIES status
# 

# In[22]:


temp=df_all.groupby(['FIES_status', 'Contract'])['Contract'].agg(['count']).reset_index()

c_v_f = pd.pivot_table(temp,values='count',index='Contract',columns='FIES_status', fill_value=0)

c_v_f


# In[23]:


#Perform the Chi-Square Test of Independence 

stat, p, dof, expected = stats.chi2_contingency(observed= c_v_f)

print(">> Contract vs FIES status")
print("Chi-square statistic = ",stat)
print("P-value = ", p)
print("Degree of freedom = ", dof)
print("Expected values = \n", expected)


# #### (vii) O_market_participation vs FIES status
# 

# In[24]:


temp=df_all.groupby(['FIES_status', 'O_market_participation'])['O_market_participation'].agg(['count']).reset_index()

omp_v_f = pd.pivot_table(temp,values='count',index='O_market_participation',columns='FIES_status', fill_value=0)

omp_v_f


# In[25]:


#Perform the Chi-Square Test of Independence 

stat, p, dof, expected = stats.chi2_contingency(observed= omp_v_f)

print(">> O_market_participation vs FIES status")
print("Chi-square statistic = ",stat)
print("P-value = ", p)
print("Degree of freedom = ", dof)
print("Expected values = \n", expected)


# #### (viii) coop_membership vs FIES status
# 

# In[26]:


temp=df_all.groupby(['FIES_status', 'coop_membership'])['coop_membership'].agg(['count']).reset_index()

cm_v_f = pd.pivot_table(temp,values='count',index='coop_membership',columns='FIES_status', fill_value=0)

cm_v_f


# In[27]:


#Perform the Chi-Square Test of Independence 

stat, p, dof, expected = stats.chi2_contingency(observed= cm_v_f)

print(">> coop_membership vs FIES status")
print("Chi-square statistic = ",stat)
print("P-value = ", p)
print("Degree of freedom = ", dof)
print("Expected values = \n", expected)


# #### (viii) sorghum_sold vs FIES status
# 

# In[28]:


temp=df_all.groupby(['FIES_status', 'sorghum_sold'])['sorghum_sold'].agg(['count']).reset_index()

ss_v_f = pd.pivot_table(temp,values='count',index='sorghum_sold',columns='FIES_status', fill_value=0)

ss_v_f


# In[29]:


#Perform the Chi-Square Test of Independence 

stat, p, dof, expected = stats.chi2_contingency(observed= ss_v_f)

print(">> sorghum_sold vs FIES status")
print("Chi-square statistic = ",stat)
print("P-value = ", p)
print("Degree of freedom = ", dof)
print("Expected values = \n", expected)


# ## 3. Recursive feature elimmination

# ##### Transform feature data to between 0 and 1 (Normalize)

# In[30]:


from sklearn import preprocessing

# Feature data
X = df_all[[i for i in df_all if i != 'FIES_status']]

scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(X)

features = pd.DataFrame(features, columns=X.columns)


# #### Summary statistics

# In[31]:


print(features.describe())


# #### Use Recursive Feature Elimination (RFE) to identify relevant features 

# In[32]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


df_vars = df_all.columns.values.tolist()

y = df_all[['FIES_status']]
X = features

# Transform features & labels into numpy arrays
X = np.array(X)
y = np.array(y)

logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=9, step=1)
rfe = rfe.fit(X, y.ravel())
print(rfe.support_)
print(rfe.ranking_)


# #### >> RFE shows that all the features in our dataset are relevant for modeling 

# ## 4. Probit regression

# #### Use the statsmodels function to fit our Probit regression with our response variable (FIES_status) and predictor matrix (features)

# In[33]:


#Implementing the model
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit

X = features
y = df_all[['FIES_status']]

X = sm.add_constant(X, prepend = False)

model = Probit(y, X.astype(float))
probit_model = model.fit()
print(probit_model.summary())


# In[ ]:




