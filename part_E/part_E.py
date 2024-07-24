#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ### Import data

# In[2]:


#Food security
food_sec = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/food_security.csv")

#Demographics
## Age
age = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_A/1_age.csv")
## Gender
gender = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_A/2_gender.csv")
## No. of pple per HH
hh_numbers = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_A/3_household_numbers.csv")
## years of schooling
yrs_of_schlng = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_A/4_years_of_schooling.csv")


         

                
#HouseHold INFO
## subcounty
#times_harvested_E = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_E/1_times_harvested.csv")
## 2_kg_sorghum_sold_per_contr
kg_sorghum_sold_per_contr_E = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_E/2_kg_sorghum_sold_per_contr.csv")
## farmers planting sorghum
sorghum_sold_E = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_E/3_sorghum_sold.csv")
## No. of pple per HH
kgs_sorghum_sold_E = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_E/4_kgs_sorghum_sold.csv")
## years of schooling
sold_to_suppliers_E = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_E/5_sold_to_suppliers.csv")
## years of schooling
kgs_sold_to_suppliers_E = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_E/6_kgs_sold_to_suppliers.csv")


# ### Food Security
# #### Inspect food security data: 
# #### Replace both Yes (1) and No (2) columns with a single column containing both outcomes
# 

# In[3]:


print(food_sec.shape)
food_sec.head()


# In[4]:


#Slide No column into Yes column to create a single column of both Yes and No values

def col_merger(food_sec):
    count = 0
    columns = []
    for col in food_sec.columns:
        food_sec[col] = food_sec[col].fillna(0)
        count += 1
        columns.append(col)
        if count == 2:
            food_sec[columns[0]+'_'+columns[1]] = food_sec[columns[0]].astype(int) + food_sec[columns[1]].astype(int)
            count = 0
            columns.clear()
        else:
            continue
    return food_sec


# In[5]:


food_sec = col_merger(food_sec)

# Our merged data looks like
food_sec.tail()    


# In[6]:


#Select the merged data columns only, rename, and put in a new dataframe df

df = food_sec.iloc[:,16:]


df = df.rename(columns={"Worried_Yes_Worried_No": "Worried", 
                   "Healthy_Yes_Healthy_No": "Healthy", 
                   "FewFoods_Yes_FewFoods_No": "FewFoods",
                   "Skipped_Yes_Skipped_No": "Skipped",
                   "AteLess_Yes_AteLess_No": "AteLess",
                   "RanOut_Yes_RanOut_No": "RanOut",
                   "Hungry_Yes_Hungry_No": "Hungry",
                   "WholeDay_Yes_WholeDay_No": "WholeDay"})
df.head()


# ### 1(a) Age (demographic) 
# #### Replace any age value with 1 and empty spaces (NaN) with 2 

# In[7]:


print(age.shape)
age.head()


# #### Age: Replace NaN with 2 and any other age values with 1. 
# ##### > 1 indicates that the interviewee is part of the age group, 
# ##### > 2 indicates the interviewee is not part of that age group 

# In[8]:




import math

def replacer(age_data):
    """
    Replaces any value with 1; indicates that the person is part of the age group, 
    2 ; indicates that the person is NOT part of the age group
    """
    
    age_cpy = age_data.reset_index()  # make sure indexes pair with number of rows

    for col in age_cpy.columns: #retrieve column names

        for index, row in age_cpy.iterrows(): #Iterate through each row 

            if math.isnan(row[col]):    
                age_cpy.loc[index, col] = 2 #replace NaN with 2
            else:                           
                age_cpy.loc[index, col] = 1 #replace any number with 1
    return age_cpy

age_cpy = replacer(age)

age_cpy = age_cpy.drop(columns=['index']).astype(int)   #drop the newly added index column


# In[9]:


# The two data frames look like

print(df.tail())
print(age_cpy.tail())


# In[10]:


#Check if there are any other number besides 1 and 2 in the dataframe
print(food_sec.eq(0).any().any()) 
print(food_sec.eq(3).any().any())
print(food_sec.eq(4).any().any())

#There are zeros and 3s in our data
#Meaning: There are empty rows
#         There were entries with both 1s and 2s (both Yes and No)


# In[11]:


# Our merged data looks like

df_merged = pd.concat([age_cpy, df], axis=1)
df_merged.head(10)


# ### Generate a dataframe of all Age counts (Yes or No) grouped based each food security variable option (Yes or No) 
# 
# #### Demo

# In[12]:



# Group by food security variable(e.g Worried (Yes/No)
# Get sums of each age representative


#Worried
worried = df_merged.groupby(['Worried', '40-49 years'])['40-49 years'].agg(['count']).reset_index() 


# Select data where worried = 1 (Yes) 
#as well as age = 1 and put in a new dataframe

worried['outcome']  = [(int(i) == 1 and int(j) == 1) for i,j in zip(worried["Worried"], worried["40-49 years"])] 


#Extract True statements with associated data

if 'True' in str(set(worried['outcome'])):
    temp_out = worried[worried['outcome'].astype(str).str.contains('True')].reset_index(drop=True)
    #print(temp_out)
else:
    pass


print("Summary of total population counts grouped based on both worried and Age categories")

print("The rows containing both Yes outcomes(Worried=1 and Age=1, ie TRUE) is selected for graph representation")

worried

#Put the above code in a function to loop over other food security variable


# In[13]:



def food_sec_yeses(df, age_cpy, df_merged):
    
    """
    Groups by food security variable, sums Age by the groups, puts yeses sums only in new dataframe,
    joins all dataframes generated
    Returns the last grouping execution table and a dictionary of all Yeses retreived from the grouping 
    tables
    """
    
    count = 1
    
    curated_dict = {}
    
    for col_df in df.columns:
        
        temp_dict = {}
        
        for col_age in age_cpy.columns:
            
            temp1 = df_merged.groupby([col_df, col_age])[col_age].agg(['count']).reset_index() #group by food sec variable and age then count
            
            temp1['outcome']  = [(int(i) == 1 and int(j) == 1) for i,j in zip(temp1[col_df], temp1[col_age])] # Identify row where both food sec and age variables equal to 1 (Yes)
            
            #Extract True statements with associated data

            if 'True' in str(set(temp1['outcome'])):
                
                temp_out = temp1[temp1['outcome'].astype(str).str.contains('True')].reset_index(drop=True)
                
                temp_dict[col_age] = temp_out['count'][0]
                
            else:
                print(col_df, col_age)
                
                print("Yes statements for both variables not detected")
                
                temp_dict[col_age] = int(0)
            
        temp_dict = pd.DataFrame(temp_dict.items()) #store the extracted dictionary in a dataframe
        
        temp_dict = temp_dict.set_index([temp_dict.columns[0]]) #Set 1st column as index
        
        temp_dict.columns = [col_df] # Assign new header
        
        curated_dict['x{0}'.format(count)] = temp_dict #Store the dataframes in the merged dictionary
        
        count += 1
        
    return temp1, curated_dict

                


# In[14]:


curated_dict = food_sec_yeses(df, age_cpy, df_merged)
curated_dict = curated_dict[1]
curated_dict


# In[15]:


# Convert the dictionary into a dataframe
#Side_join all the dataframes
                    
curated_df = pd.concat(curated_dict.values(), axis=1)
curated_df


# #### Calculate percentage representation out of sample size (175) and transpose

# In[16]:


# Calculate percentage representation for each year
curated_perc = curated_df/ 175 * 100
curated_perc
#curated_perc.index.name = 'Age'

# Transpose the matrix
curated_transposed = curated_perc.T
curated_transposed

curated_transposed = curated_transposed.reset_index()  #convert index of the dataframe into a column

#rename the index column
curated_transposed = curated_transposed.rename({'index': 'FoodSecurity'}, axis=1)
curated_transposed


# ### generate plot for the above table

# In[17]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


#plt.rcParams.update({'axes.facecolor':'white'})

plt.figure(facecolor='white') 

ax = curated_transposed.plot(x='FoodSecurity', 
                             y=list(curated_transposed.columns[1:]), 
                             #height=100,
                             ylim=(0,30),
                             kind="bar",
                             width=0.9,
                             rot = 0)


ax.set_ylabel("Percentage population", fontsize = 12)
ax.set_xlabel("Food Security", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Age", bbox_to_anchor=(1.43, 1.03), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

#plt.savefig('fig4.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#== N/B facecolor=ax.get_facecolor() sets the backgrund as white


# ## 2. Gender (demographic)

# In[18]:


# Our gender data looks like
gender.head()


# In[19]:


# Merger the two columns into a single column representation 
temp = col_merger(gender)

#Select the merged data column, rename, and put in a new dataframe df
temp = temp.iloc[:,-1:]
gender_cpy = temp.rename(columns={"Male_Female": "Gender"}) 
gender_cpy.head()


# In[20]:


# Join both gender and food security dataframes

df_merged = pd.concat([gender_cpy, df], axis=1)
df_merged.head()


# In[21]:


# Group by both food security and gender
# Extract foodsecure (1) for BOTH genders


def food_sec_binary_yeses(df, gender_cpy, df_merged, gender):
    # df and gender are only used to obtain original column names
    
    """
    Groups by food security variable, sums Age by the groups, puts yeses sums only in new dataframe,
    joins all dataframes generated
    Returns the last grouping execution table and a dictionary of all Yeses retreived from the grouping 
    tables
    """
    
    count = 1
    
    curated_dict = {}
    
    for col_df in df.columns:
        
        temp_dict = {}
        
        for col_age in gender_cpy.columns:
            
            temp1 = df_merged.groupby([col_df, col_age])[col_age].agg(['count']).reset_index() #group by food sec variable and age then count
            
            temp1['outcome1']  = [(int(i) == 1 and int(j) == 1) for i,j in zip(temp1[col_df], temp1[col_age])] # Identify row where both food sec and age variables equal to 1 (Yes)

            temp1['outcome2']  = [(int(i) == 1 and int(j) == 2) for i,j in zip(temp1[col_df], temp1[col_age])] # Identify row where both food sec and age variables equal to 1 (Yes)
            
            temp1['outcome3']  = [(int(i) == 1 and int(j) == 3) for i,j in zip(temp1[col_df], temp1[col_age])] # Identify row where both food sec and age variables equal to 1 (Yes)
            
            temp1['outcome4']  = [(int(i) == 1 and int(j) == 4) for i,j in zip(temp1[col_df], temp1[col_age])] # Identify row where both food sec and age variables equal to 1 (Yes)

            temp1['outcome5']  = [(int(i) == 1 and int(j) == 5) for i,j in zip(temp1[col_df], temp1[col_age])] # Identify row where both food sec and age variables equal to 1 (Yes)

            #Extract True statements with associated data

            if 'True' in str(set(temp1['outcome1'])):
                
                temp_out = temp1[temp1['outcome1'].astype(str).str.contains('True')].reset_index(drop=True)
                
                temp_dict[gender.columns[0]] = temp_out['count'][0] #assign male
                
                
            if 'True' in str(set(temp1['outcome2'])):
                
                temp_out = temp1[temp1['outcome2'].astype(str).str.contains('True')].reset_index(drop=True)
                
                temp_dict[gender.columns[1]] = temp_out['count'][0] #assign female from column name of gender
            
            
            if 'True' in str(set(temp1['outcome3'])):
                
                temp_out = temp1[temp1['outcome3'].astype(str).str.contains('True')].reset_index(drop=True)
                
                temp_dict[gender.columns[2]] = temp_out['count'][0] #assign female from column name of gender
                
            if 'True' in str(set(temp1['outcome4'])):
                
                temp_out = temp1[temp1['outcome4'].astype(str).str.contains('True')].reset_index(drop=True)
                
                temp_dict[gender.columns[3]] = temp_out['count'][0] #assign female from column name of gender
            
            if 'True' in str(set(temp1['outcome5'])):
                
                temp_out = temp1[temp1['outcome5'].astype(str).str.contains('True')].reset_index(drop=True)
                
                temp_dict[gender.columns[4]] = temp_out['count'][0] #assign female from column name of gender
            
            else:
                print(col_df, col_age)
                
                print("Yes statements for both variables not detected")
                
                temp_dict[col_age] = int(0)
            
        temp_dict = pd.DataFrame(temp_dict.items()) #store the extracted dictionary in a dataframe
        
        temp_dict = temp_dict.set_index([temp_dict.columns[0]]) #Set 1st column as index
        
        temp_dict.columns = [col_df] # Assign new header
        
        curated_dict['x{0}'.format(count)] = temp_dict #Store the dataframes in the merged dictionary
        
        count += 1
        
    return temp1, curated_dict


# In[22]:


# Group by both gender and food security and count
# Select males (1) and worried (1) plus Females (2) and worried (1)
# N/B pass in the gender to obtain both names (male female)
curated_dict = food_sec_binary_yeses(df, gender_cpy, df_merged, gender)
#curated_dict = curated_dict[1]
curated_dict


# In[23]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[24]:


# Calculate percentage representation for each year
curated_perc = curated_df/ 175 * 100
curated_perc
#curated_perc.index.name = 'Age'

# Transpose the matrix
curated_transposed = curated_perc.T
curated_transposed

curated_transposed = curated_transposed.reset_index()  #convert index of the dataframe into a column

#rename the index column
curated_transposed = curated_transposed.rename({'index': 'FoodSecurity'}, axis=1)
curated_transposed


# In[25]:


### Generate plot


# In[26]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


#plt.rcParams.update({'axes.facecolor':'white'})

plt.figure(facecolor='white') 

ax = curated_transposed.plot(x='FoodSecurity', 
                             y=list(curated_transposed.columns[1:]), 
                             #height=100,
                             ylim=(0,50),
                             kind="bar",
                             width=0.9,
                             color=['#069AF3','#E50000'],
                             rot = 0)


ax.set_ylabel("% Percentage population", fontsize = 12)
ax.set_xlabel("Food Security", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Gender", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

#plt.savefig('fig5.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#N/B facecolor=ax.get_facecolor() sets the backgrund as white


# ### 3. No. of people per HH

# In[27]:


hh_numbers.head()


# In[28]:


# Inspect data
hh_numbers.describe()


# ##### Generate groups/class from the above data

# #### CLASSES
# ##### 0-5 people
# ##### 5-10 people
# 

# In[29]:


# Categorise the groups with unique decimal numbers (ie, 1, 2, 3, etc)

hh_numbers_cpy = hh_numbers
identifiers = {}
for index, row in hh_numbers_cpy.iterrows():
    if row['people_per_HH'] in list(range(1, 6, 1)):
        identifiers[index] = 1 
        
    elif row['people_per_HH'] in list(range(6, 11, 1)):
        identifiers[index] = 2 
        
    else:
        print(row['people_per_HH'], "Not in range")


# Convert identifiers dictionary to dataframe 
IDs = pd.DataFrame(identifiers.items(), columns=['index', 'Group_identifiers'])
IDs.set_index('index')

#Join with hh_numbers_cpy
hh_numbers_cpy = pd.concat([hh_numbers_cpy, IDs], axis=1).drop(['index'], axis=1)

# Our new dataframe looks like
hh_numbers_cpy


# In[30]:


# Drop the people_per_HH column and only retain Group identifies ror grouping purposes
del hh_numbers_cpy['people_per_HH'] 


# In[31]:


# Join with the food security data

df_merged = pd.concat([hh_numbers_cpy['Group_identifiers'], df], axis=1)
df_merged.head()


# In[32]:


#Generate an empty dataframe with the following column names
#0-5 people =< 5
#5-10 people = 5 - 10

columns = ["0-5 people", "5-10 people"]
index = [0]
hh_numbers_groupnames = pd.DataFrame(index=index, columns=columns)
hh_numbers_groupnames


# In[33]:


# Group by both food security and groups
# Extract foodsecure (1) for BOTH groups: 0-5 people (1)  5-10 people (2)

curated_dict = food_sec_binary_yeses(df, hh_numbers_cpy, df_merged, hh_numbers_groupnames)


# In[34]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[35]:


# Calculate percentage representation for each year
curated_perc = curated_df/ 175 * 100
curated_perc
#curated_perc.index.name = 'Age'

# Transpose the matrix
curated_transposed = curated_perc.T
curated_transposed

curated_transposed = curated_transposed.reset_index()  #convert index of the dataframe into a column

#rename the index column
curated_transposed = curated_transposed.rename({'index': 'FoodSecurity'}, axis=1)
curated_transposed


# In[36]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


#plt.rcParams.update({'axes.facecolor':'white'})

plt.figure(facecolor='white') 

ax = curated_transposed.plot(x='FoodSecurity', 
                             y=list(curated_transposed.columns[1:]), 
                             #height=100,
                             ylim=(0,80),
                             kind="bar",
                             width=0.9,
                             color=['#069AF3','#E50000'],
                             rot = 0)


ax.set_ylabel("% Percentage population", fontsize = 12)
ax.set_xlabel("Food Security", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "People per HH", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

#plt.savefig('fig6.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#N/B facecolor=ax.get_facecolor() sets the backgrund as white


# #### 4. Years of schooling

# In[37]:


yrs_of_schlng.head()


# In[38]:


#Replace Nan with 2, and any value with 1

yrs_of_schlng_cpy = replacer(yrs_of_schlng)

yrs_of_schlng_cpy = yrs_of_schlng_cpy.drop(columns=['index']).astype(int)   #drop the newly added index column

yrs_of_schlng_cpy.head()


# In[39]:


# Our merged data looks like

df_merged = pd.concat([yrs_of_schlng_cpy, df], axis=1)
df_merged.head()


# In[40]:


curated_dict = food_sec_yeses(df, yrs_of_schlng_cpy, df_merged)
curated_dict = curated_dict[1]
curated_dict


# In[41]:


# Convert the dictionary into a dataframe
# Side_join all the dataframes
                    
curated_df = pd.concat(curated_dict.values(), axis=1)
curated_df


# In[42]:


# Calculate percentage representation for each year
curated_perc = curated_df/ 175 * 100
curated_perc
#curated_perc.index.name = 'Age'

# Transpose the matrix
curated_transposed = curated_perc.T
curated_transposed

curated_transposed = curated_transposed.reset_index()  #convert index of the dataframe into a column

#rename the index column
curated_transposed = curated_transposed.rename({'index': 'FoodSecurity'}, axis=1)
curated_transposed


# In[43]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


#plt.rcParams.update({'axes.facecolor':'white'})

plt.figure(facecolor='white') 

ax = curated_transposed.plot(x='FoodSecurity', 
                             y=list(curated_transposed.columns[1:]), 
                             #height=100,
                             ylim=(0,30),
                             kind="bar",
                             width=0.9,
                             rot = 0)


ax.set_ylabel("% Percentage population", fontsize = 12)
ax.set_xlabel("Food Security", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Years of schooling", bbox_to_anchor=(1.43, 1.03), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

#plt.savefig('fig8.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())


# #==============  
# 
# # PART E
# 
# #==============  
# 

# ### B.(i) No of times sorghum harvested
# 

# In[44]:


# print(times_harvested_E.shape)
# times_harvested_E.head()


# In[45]:


# # Inspect data
# times_harvested_E.describe()


# ##### Generate groups/class from the above data

# #### CLASSES
# ##### 0-3 times
# ##### 3-6 times
# 

# In[46]:


# # Categorise the groups with unique decimal numbers (ie, 1, 2, 3, etc)

# times_harvested_E_cpy = times_harvested_E
# identifiers = {}
# for index, row in times_harvested_E_cpy.iterrows():
#     if row['Times harvested'] in list(range(0, 4, 1)):
#         identifiers[index] = 1 
        
#     elif row['Times harvested'] in list(range(4, 7, 1)):
#         identifiers[index] = 2 
        
#     else:
#         print(row['Times harvested'], "Not in range")


# # Convert identifiers dictionary to dataframe 
# IDs = pd.DataFrame(identifiers.items(), columns=['index', 'Group_identifiers'])
# IDs.set_index('index')

# #Join with hh_numbers_cpy
# times_harvested_E_cpy = pd.concat([times_harvested_E_cpy, IDs], axis=1).drop(['index'], axis=1)

# # Our new dataframe looks like
# times_harvested_E_cpy


# In[47]:


# # Drop the people_per_HH column and only retain Group identifies ror grouping purposes
# del times_harvested_E_cpy['Times harvested'] 


# In[48]:


# # Join with the food security data

# df_merged = pd.concat([times_harvested_E_cpy['Group_identifiers'], df], axis=1)
# df_merged = df_merged.dropna()
# df_merged.head()


# In[49]:


# #Generate an empty dataframe with the following column names
# #0-5 people =< 5
# #5-10 people = 5 - 10

# columns = ["0-3 times", "4-6 times"]
# index = [0]
# times_harvested_E_groupnames = pd.DataFrame(index=index, columns=columns)
# times_harvested_E_groupnames


# In[50]:


# # Group by both food security and groups
# # Extract foodsecure (1) for BOTH groups: 0-5 people (1)  5-10 people (2)
# curated_dict = food_sec_binary_yeses(df, times_harvested_E_cpy, df_merged, times_harvested_E_groupnames)
# curated_dict


# In[51]:


# # Convert the dictionary into a dataframe
                    
# curated_df = pd.concat(curated_dict[1].values(), axis=1)
# curated_df


# In[52]:


# # Delete the Group identifiers row
# curated_df = curated_df.drop('Group_identifiers')
# curated_df


# In[53]:



# curated_transposed.to_excel("times_harvested_E_.xlsx", sheet_name='times_harvested_vs_food_sec_B')


# In[54]:


# # Calculate percentage representation for each year
# curated_perc = curated_df/ 175 * 100
# curated_perc
# #curated_perc.index.name = 'Age'

# # Transpose the matrix
# curated_transposed = curated_perc.T
# curated_transposed

# curated_transposed = curated_transposed.reset_index()  #convert index of the dataframe into a column

# #rename the index column
# curated_transposed = curated_transposed.rename({'index': 'FoodSecurity'}, axis=1)
# curated_transposed


# In[55]:


# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure


# #plt.rcParams.update({'axes.facecolor':'white'})

# plt.figure(facecolor='white') 

# ax = curated_transposed.plot(x='FoodSecurity', 
#                              y=list(curated_transposed.columns[1:]), 
#                              #height=100,
#                              ylim=(0,80),
#                              kind="bar",
#                              width=0.9,
#                              color=['#069AF3','#E50000'],
#                              rot = 0)


# ax.set_ylabel("% Percentage population", fontsize = 12)
# ax.set_xlabel("Food Security", fontsize = 12)

# plt.rcParams.update({'font.size': 14})

# ax.legend(ncol=1, title = "times sorghum harvested\n per year", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

# plt.rcParams.update({'font.size': 8.8})

# plt.savefig('times_harvested_E.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

# #N/B facecolor=ax.get_facecolor() sets the backgrund as white


# #### E (ii) kg_sorghum_produced

# In[56]:


kg_sorghum_sold_per_contr_E.head()


# In[57]:


# Inspect data
kg_sorghum_sold_per_contr_E.describe()


# In[58]:


# Drop rows with zeros and count the population
df_without_zeros = kg_sorghum_sold_per_contr_E[(kg_sorghum_sold_per_contr_E.T != 0).any()]
df_without_zeros.shape


# In[59]:


# Categorise the groups with unique decimal numbers (ie, 1, 2, 3, etc)

kg_sorghum_sold_per_contr_E_cpy = kg_sorghum_sold_per_contr_E
identifiers = {}
for index, row in kg_sorghum_sold_per_contr_E_cpy.iterrows():
    if row['Avg Kgs'] in list(range(1, 91, 1)):
        identifiers[index] = 1 
        
    elif row['Avg Kgs'] in list(range(91, 181, 1)):
        identifiers[index] = 2 

    elif row['Avg Kgs'] in list(range(181, 271, 1)):
        identifiers[index] = 3 
    
    elif row['Avg Kgs'] in list(range(271, 361, 1)):
        identifiers[index] = 4 
        
    elif row['Avg Kgs'] in list(range(361, 4001, 1)):
        identifiers[index] = 5 

    else:
        print(row['Avg Kgs'], "Not in range")


# Convert identifiers dictionary to dataframe 
IDs = pd.DataFrame(identifiers.items(), columns=['index', 'Group_identifiers'])
IDs.set_index('index')

#Join with hh_numbers_cpy
kg_sorghum_sold_per_contr_E_cpy = pd.concat([kg_sorghum_sold_per_contr_E_cpy, IDs], axis=1).drop(['index'], axis=1)

# Our new dataframe looks like
kg_sorghum_sold_per_contr_E_cpy


# In[60]:


# Drop the Avg Kgs column and only retain Group identifies ror grouping purposes
del kg_sorghum_sold_per_contr_E_cpy['Avg Kgs'] 


# In[61]:


# Join with the food security data

df_merged = pd.concat([kg_sorghum_sold_per_contr_E_cpy['Group_identifiers'], df], axis=1)
df_merged.head()


# In[62]:


#Generate an empty dataframe with the following column names
#0-5 people =< 5
#5-10 people = 5 - 10

columns = ["0-90 Kgs", "90-180 Kgs", "180-270 Kgs", "270-360 Kgs", "Above 360 Kgs"]
index = [0]
kg_sorghum_sold_per_contr_E_groupnames = pd.DataFrame(index=index, columns=columns)
kg_sorghum_sold_per_contr_E_groupnames


# In[63]:


# Group by both food security and groups
# Extract foodsecure (1) for BOTH groups: 0-5 people (1)  5-10 people (2)

curated_dict = food_sec_binary_yeses(df, kg_sorghum_sold_per_contr_E_cpy, df_merged, kg_sorghum_sold_per_contr_E_groupnames)


# In[64]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[65]:


# Calculate percentage representation for each year
curated_perc = curated_df/ 83 * 100
curated_perc
#curated_perc.index.name = 'Age'

# Transpose the matrix
curated_transposed = curated_perc.T
curated_transposed

curated_transposed = curated_transposed.reset_index()  #convert index of the dataframe into a column

#rename the index column
curated_transposed = curated_transposed.rename({'index': 'FoodSecurity'}, axis=1)
curated_transposed


# In[66]:



curated_transposed.to_excel("kg_sorghum_sold_per_contr_E.xlsx", sheet_name='kg_sold_p_contr_E_vs_food_sec_B')


# In[67]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from bokeh.palettes import Magma, Inferno, Plasma, Viridis, Cividis

plt.figure(facecolor='white') 

ax = curated_transposed.plot(x='FoodSecurity', 
                             y=list(curated_transposed.columns[1:]), 
                             #height=100,
                             ylim=(0,30),
                             kind="bar",
                             width=0.9,
                             color=Magma[5],
                             rot = 0)

ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Av. Kgs of sorghum sold\nper contract/cooperative\narrangement per harvest", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

plt.savefig('kg_sorghum_sold_per_contr_E.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#N/B facecolor=ax.get_facecolor() sets the backgrund as white


# #### (iv) sorghum_sold
# 

# In[55]:


print(sorghum_sold_E.shape)
sorghum_sold_E.head()


# In[56]:


# Merger the two columns into a single column representation 
temp = col_merger(sorghum_sold_E)

#Select the merged data column, rename, and put in a new dataframe df
temp = temp.iloc[:,-1:]
sorghum_sold_E_cpy = temp.rename(columns={"Yes_No": "sorghum_sold"}) 
sorghum_sold_E_cpy.head()


# In[57]:


#Write out sorghum sold (for regr)
sorghum_sold_E_cpy['sorghum_sold'].to_csv("../regr/E_sorghum_sold_r.csv", sep=',', encoding='utf-8')


# In[58]:


# Join both gender and food security dataframes

df_merged = pd.concat([sorghum_sold_E_cpy, df], axis=1)
df_merged.head()


# In[59]:


#Generate an empty dataframe with the following column names
#0-5 people =< 5
#5-10 people = 5 - 10

columns = ["Yes", "No"]
index = [0]
sorghum_sold_E_groupnames = pd.DataFrame(index=index, columns=columns)
sorghum_sold_E_groupnames


# In[60]:


# Group by both gender and food security and count
# Select males (1) and worried (1) plus Females (2) and worried (1)
# N/B pass in the gender to obtain both names (male female)
curated_dict = food_sec_binary_yeses(df, sorghum_sold_E_cpy, df_merged, sorghum_sold_E_groupnames)


# In[61]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[62]:


# Delete sorghum sold row
curated_df = curated_df.drop('sorghum_sold')
curated_df


# In[63]:


# Calculate percentage representation for each year
curated_perc = curated_df/ 207 * 100
curated_perc
#curated_perc.index.name = 'Age'

# Transpose the matrix
curated_transposed = curated_perc.T
curated_transposed

curated_transposed = curated_transposed.reset_index()  #convert index of the dataframe into a column

#rename the index column
curated_transposed = curated_transposed.rename({'index': 'FoodSecurity'}, axis=1)
curated_transposed


# In[76]:



curated_transposed.to_excel("sorghum_sold.xlsx", sheet_name='sorghum_sold_vs_food_sec_B')


# In[77]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


#plt.rcParams.update({'axes.facecolor':'white'})

plt.figure(facecolor='white') 

ax = curated_transposed.plot(x='FoodSecurity', 
                             y=list(curated_transposed.columns[1:]), 
                             #height=100,
                             ylim=(0,60),
                             kind="bar",
                             width=0.9,
                             color=Magma[3],
                             rot = 0)

ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Sorghum sold", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

plt.savefig('sorghum_sold_E.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#N/B facecolor=ax.get_facecolor() sets the backgrund as white


# #### E (v) kgs_sorghum_sold_

# In[78]:


kgs_sorghum_sold_E.describe()


# In[79]:


# Drop rows with zeros and count the population
df_without_zeros = kgs_sorghum_sold_E[(kgs_sorghum_sold_E.T != 0).any()]
df_without_zeros.shape


# In[80]:


# Categorise the groups with unique decimal numbers (ie, 1, 2, 3, etc)

kgs_sorghum_sold_E_cpy = kgs_sorghum_sold_E
identifiers = {}
for index, row in kgs_sorghum_sold_E_cpy.iterrows():
    if row['Avg Kgs'] in list(range(1, 91, 1)):
        identifiers[index] = 1 
        
    elif row['Avg Kgs'] in list(range(91, 181, 1)):
        identifiers[index] = 2 

    elif row['Avg Kgs'] in list(range(181, 271, 1)):
        identifiers[index] = 3 
    
    elif row['Avg Kgs'] in list(range(271, 361, 1)):
        identifiers[index] = 4 
        
    elif row['Avg Kgs'] in list(range(361, 4001, 1)):
        identifiers[index] = 5 

    else:
        print(row['Avg Kgs'], "Not in range")


# Convert identifiers dictionary to dataframe 
IDs = pd.DataFrame(identifiers.items(), columns=['index', 'Group_identifiers'])
IDs.set_index('index')

#Join with hh_numbers_cpy
kgs_sorghum_sold_E_cpy = pd.concat([kgs_sorghum_sold_E_cpy, IDs], axis=1).drop(['index'], axis=1)

# Our new dataframe looks like
kgs_sorghum_sold_E_cpy


# In[81]:


# Drop the people_per_HH column and only retain Group identifies ror grouping purposes
del kgs_sorghum_sold_E_cpy['Avg Kgs'] 


# In[82]:


# Join with the food security data

df_merged = pd.concat([kgs_sorghum_sold_E_cpy['Group_identifiers'], df], axis=1)
df_merged.head()


# In[83]:


#Generate an empty dataframe with the following column names
#0-5 people =< 5
#5-10 people = 5 - 10

columns = ["0-90 Kgs", "90-180 Kgs", "180-270 Kgs", "270-360 Kgs", "Above 360 Kgs"]
index = [0]
kgs_sorghum_sold_E_groupnames = pd.DataFrame(index=index, columns=columns)
kgs_sorghum_sold_E_groupnames


# In[84]:


# Group by both food security and groups
# Extract foodsecure (1) for BOTH groups: 0-5 people (1)  5-10 people (2)

curated_dict = food_sec_binary_yeses(df, kgs_sorghum_sold_E_cpy, df_merged, kgs_sorghum_sold_E_groupnames)


# In[85]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[86]:


# Calculate percentage representation for each year
curated_perc = curated_df/ 115 * 100
curated_perc
#curated_perc.index.name = 'Age'

# Transpose the matrix
curated_transposed = curated_perc.T
curated_transposed

curated_transposed = curated_transposed.reset_index()  #convert index of the dataframe into a column

#rename the index column
curated_transposed = curated_transposed.rename({'index': 'FoodSecurity'}, axis=1)
curated_transposed


# In[87]:



curated_transposed.to_excel("Av_Kgs_sorghum_sold_per_harvest.xlsx", sheet_name='AvKgs_per_harv_vs_food_sec_B')


# In[88]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


#plt.rcParams.update({'axes.facecolor':'white'})

plt.figure(facecolor='white') 

ax = curated_transposed.plot(x='FoodSecurity', 
                             y=list(curated_transposed.columns[1:]), 
                             #height=100,
                             ylim=(0,20),
                             kind="bar",
                             width=0.9,
                             color=Magma[5],
                             rot = 0)

ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Av. Kgs of sorghum\n sold per harvest", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

plt.savefig('Kgs_sorghum_sold_per_harvest_E.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#N/B facecolor=ax.get_facecolor() sets the backgrund as white


# #### E. (iv) sold_to_suppliers

# In[89]:


print(sold_to_suppliers_E.shape)
sold_to_suppliers_E.head()


# In[90]:


# Merger the two columns into a single column representation 
temp = col_merger(sold_to_suppliers_E)

#Select the merged data column, rename, and put in a new dataframe df
temp = temp.iloc[:,-1:]
sold_to_suppliers_E_cpy = temp.rename(columns={"Yes_No": "sorghum_sold"}) 
sold_to_suppliers_E_cpy.head()


# In[91]:


# Join both gender and food security dataframes

df_merged = pd.concat([sold_to_suppliers_E_cpy, df], axis=1)
df_merged.head()


# In[92]:


#Generate an empty dataframe with the following column names
#0-5 people =< 5
#5-10 people = 5 - 10

columns = ["Yes", "No"]
index = [0]
sold_to_suppliers_E_groupnames = pd.DataFrame(index=index, columns=columns)
sold_to_suppliers_E_groupnames


# In[93]:


# Group by both gender and food security and count
# Select males (1) and worried (1) plus Females (2) and worried (1)
# N/B pass in the gender to obtain both names (male female)
curated_dict = food_sec_binary_yeses(df, sold_to_suppliers_E_cpy, df_merged, sold_to_suppliers_E_groupnames)


# In[94]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[95]:


# Delete sorghum sold row
curated_df = curated_df.drop('sorghum_sold')
curated_df


# In[96]:


# Calculate percentage representation for each year
curated_perc = curated_df/ 207 * 100
curated_perc
#curated_perc.index.name = 'Age'

# Transpose the matrix
curated_transposed = curated_perc.T
curated_transposed

curated_transposed = curated_transposed.reset_index()  #convert index of the dataframe into a column

#rename the index column
curated_transposed = curated_transposed.rename({'index': 'FoodSecurity'}, axis=1)
curated_transposed


# In[97]:



curated_transposed.to_excel("sorghum_sold_to_suppliers.xlsx", sheet_name='sold_to_suppliers_vs_food_sec_B')


# In[98]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


#plt.rcParams.update({'axes.facecolor':'white'})

plt.figure(facecolor='white') 

ax = curated_transposed.plot(x='FoodSecurity', 
                             y=list(curated_transposed.columns[1:]), 
                             #height=100,
                             ylim=(0,60),
                             kind="bar",
                             width=0.9,
                             color=Magma[3],
                             rot = 0)

ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Sorghum sold\n to suppliers", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

plt.savefig('sold_to_suppliers_E.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#N/B facecolor=ax.get_facecolor() sets the backgrund as white


# #### E. (vi) 6_kgs_sold_to_suppliers

# In[99]:


kgs_sold_to_suppliers_E.head()


# In[100]:


kgs_sold_to_suppliers_E.describe()


# In[101]:


# Drop rows with zeros and count the population
df_without_zeros = kgs_sold_to_suppliers_E[(kgs_sold_to_suppliers_E.T != 0).any()]
df_without_zeros.shape


# In[102]:


# Categorise the groups with unique decimal numbers (ie, 1, 2, 3, etc)

kgs_sold_to_suppliers_E_cpy = kgs_sold_to_suppliers_E
identifiers = {}
for index, row in kgs_sold_to_suppliers_E_cpy.iterrows():
    if row['Avg Kgs'] in list(range(1, 91, 1)):
        identifiers[index] = 1 
        
    elif row['Avg Kgs'] in list(range(91, 181, 1)):
        identifiers[index] = 2 

    elif row['Avg Kgs'] in list(range(181, 271, 1)):
        identifiers[index] = 3 
    
    elif row['Avg Kgs'] in list(range(271, 361, 1)):
        identifiers[index] = 4 
        
    elif row['Avg Kgs'] in list(range(361, 4001, 1)):
        identifiers[index] = 5 

    else:
        print(row['Avg Kgs'], "Not in range")


# Convert identifiers dictionary to dataframe 
IDs = pd.DataFrame(identifiers.items(), columns=['index', 'Group_identifiers'])
IDs.set_index('index')

#Join with hh_numbers_cpy
kgs_sold_to_suppliers_E_cpy = pd.concat([kgs_sold_to_suppliers_E_cpy, IDs], axis=1).drop(['index'], axis=1)

# Our new dataframe looks like
kgs_sold_to_suppliers_E_cpy


# In[103]:


# Drop the people_per_HH column and only retain Group identifies ror grouping purposes
del kgs_sold_to_suppliers_E_cpy['Avg Kgs'] 


# In[104]:


# Join with the food security data

df_merged = pd.concat([kgs_sold_to_suppliers_E_cpy['Group_identifiers'], df], axis=1)
df_merged.head()


# In[105]:


#Generate an empty dataframe with the following column names
#0-5 people =< 5
#5-10 people = 5 - 10

columns = ["1-90 Kgs", "90-180 Kgs", "180-270 Kgs", "270-360 Kgs", "Above 360 Kgs"]
index = [0]
kgs_sold_to_suppliers_E_groupnames = pd.DataFrame(index=index, columns=columns)
kgs_sold_to_suppliers_E_groupnames


# In[106]:


# Group by both food security and groups
# Extract foodsecure (1) for BOTH groups: 0-5 people (1)  5-10 people (2)

curated_dict = food_sec_binary_yeses(df, kgs_sold_to_suppliers_E_cpy, df_merged, kgs_sold_to_suppliers_E_groupnames)


# In[107]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[108]:


# Calculate percentage representation for each year
curated_perc = curated_df/ 37 * 100
curated_perc
#curated_perc.index.name = 'Age'

# Transpose the matrix
curated_transposed = curated_perc.T
curated_transposed

curated_transposed = curated_transposed.reset_index()  #convert index of the dataframe into a column

#rename the index column
curated_transposed = curated_transposed.rename({'index': 'FoodSecurity'}, axis=1)
curated_transposed


# In[109]:



curated_transposed.to_excel("Av_Kgs_sorghum_sold_to_suppliers.xlsx", sheet_name='Avkgs_soldto_supp_vs_food_sec_B')


# In[110]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


#plt.rcParams.update({'axes.facecolor':'white'})

plt.figure(facecolor='white') 

ax = curated_transposed.plot(x='FoodSecurity', 
                             y=list(curated_transposed.columns[1:]), 
                             #height=100,
                             ylim=(0,40),
                             kind="bar",
                             width=0.9,
                             color=Magma[5],
                             rot = 0)

ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Av. Kgs of sorghum\n sold to suppliers\n per harvest", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

plt.savefig('Kgs_sold_to_suppliers_per_harvest_E.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#N/B facecolor=ax.get_facecolor() sets the backgrund as white


# In[ ]:




