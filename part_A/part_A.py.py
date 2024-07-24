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
## Age
age = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_A/1_age.csv")
## Gender
gender = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_A/2_gender.csv")
## No. of pple per HH
hh_numbers = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_A/3_household_numbers.csv")
## years of schooling
yrs_of_schlng = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_A/4_years_of_schooling.csv")


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


# In[10]:


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

#Write out FIES data (for regr)
df.to_csv("./regr/fies.csv", sep=',', encoding='utf-8')


# ### 1(a) Age (demographic) 
# #### Replace any age value with 1 and empty spaces (NaN) with 2 

# In[11]:


print(age.shape)
age.head()


# #### Age: Replace NaN with 2 and any other age values with 1. 
# ##### > 1 indicates that the interviewee is part of the age group, 
# ##### > 2 indicates the interviewee is not part of that age group 

# In[35]:




import math

def replacer(age):
    """
    Replaces any value with 1; indicates that the person is part of the age group, 
    2 ; indicates that the person is NOT part of the age group
    """
    
    age_cpy = age.reset_index()  # make sure indexes pair with number of rows

    for col in age_cpy.columns: #retrieve column names

        for index, row in age_cpy.iterrows(): #Iterate through each row 

            if math.isnan(row[col]):    
                age_cpy.loc[index, col] = 2 #replace NaN with 2
            else:                           
                age_cpy.loc[index, col] = 1 #replace any number with 1
    return age_cpy

age_cpy = replacer(age)

age_cpy = age_cpy.drop(columns=['index']).astype(int)   #drop the newly added index column


# In[36]:


# The two data frames look like

print(df.tail())
print(age_cpy.head())


# In[37]:


#Slide Ages into one column representing the 5 categories with 1, 2, 3, 4, 5
from io import StringIO

age_temp = age_cpy.copy()

count = 1
for i in age_temp.columns:
    age_temp[i] = age_temp[i].replace(2, 0)
    age_temp[i] = age_temp[i].replace(1, count)
    count += 1

#slide the columns into one
age_temp['Ages']= age_temp.sum(axis=1)

#Write out Ages (for regr)
age_temp['Ages'].to_csv("./regr/Age_r.csv", sep=',', encoding='utf-8')


# In[39]:


#Check if there are any other number besides 1 and 2 in the dataframe

print(df.eq(3).any().any())  #check for any 3'sZ

print(len(df.loc[df['Worried'].isin([1])].index)) #returns count of row indices with 1
print(len(df.loc[df['Worried'].isin([2])].index)) #returns count of row indices with 2
print(len(df.loc[df['Worried'].isin([3])].index)) #returns count of row indices with 3
print(df.loc[df['Worried'].isin([3])].index)


# In[40]:


# Our merged data looks like

print(age_cpy.shape, df.shape)  #check shape of both matrices

df_merged = pd.concat([age_cpy, df], axis=1)
df_merged.head()


# ### Generate a dataframe of all Age counts (Yes or No) grouped based each food security variable option (Yes or No) 
# 
# #### Demonstration

# In[41]:



# Group by food security variable "Worried" as well as each age "40-49 years"
# Get counts of each age representative


#Worried
worried = df_merged.groupby(['Worried', '40-49 years'])['40-49 years'].agg(['count']).reset_index() 

print(worried)

print("\nThe total count for all permutations equals", worried["count"].sum())
print("\nThe total count for persons with age 40-49 years (worried and not worried) equals", len(df_merged.loc[df_merged["40-49 years"] == 1]), '\n')

# Select data where worried = 1 (Yes) 
#as well as age = 1 and put in a new dataframe

worried['outcome']  = [(int(i) == 1 and int(j) == 1) for i,j in zip(worried["Worried"], worried["40-49 years"])] 


#Extract True statements with associated data

if 'True' in str(set(worried['outcome'])):
    temp_out = worried[worried['outcome'].astype(str).str.contains('True')].reset_index(drop=True)
    #print(temp_out)
else:
    pass


print("Summary of total population counts grouped based on Worried and Age categories")

print("The rows containing both Yes outcomes(Worried=1 and Age=1, ie TRUE) is selected for graph representation \n")

worried


# ##### The above code put in a function to loop over each food security variable looks like:
# 

# In[42]:



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
            
            # Group by each FIES variable and age then count
            temp1 = df_merged.groupby([col_df, col_age])[col_age].agg(['count']).reset_index() 
            
            # Identify row where both FIES and age variables equal to 1 (Yes)
            temp1['outcome']  = [(int(i) == 1 and int(j) == 1) for i,j in zip(temp1[col_df], temp1[col_age])] 
            
            ##print("\nThe total count for persons with age ", col_age, " (both ", col_df, " and not) equals", len(df_merged.loc[df_merged[col_age] == 1]), '\n')

            #Extract True statements from the above with its associated data

            if 'True' in str(set(temp1['outcome'])):
                
                temp_out = temp1[temp1['outcome'].astype(str).str.contains('True')].reset_index(drop=True)
                
                temp_dict[col_age] = temp_out['count'][0]
                
            else:
                print(col_df, col_age)
                
                print("Yes statements for both variables not detected")
                
                temp_dict[col_age] = int(0)
                
        # Store the extracted dictionary in a dataframe    
        temp_dict = pd.DataFrame(temp_dict.items()) 
        
        # Set 1st column as index
        temp_dict = temp_dict.set_index([temp_dict.columns[0]]) 
        
        # Assign new header
        temp_dict.columns = [col_df] 
        
        # Store the dataframes in the merged dictionary
        curated_dict['x{0}'.format(count)] = temp_dict 
        
        count += 1
        
    return temp1, curated_dict

                


# In[43]:


curated_dict = food_sec_yeses(df, age_cpy, df_merged)
curated_dict = curated_dict[1]
curated_dict


# In[44]:


# Convert the dictionary into a dataframe
#Side_join all the dataframes
                    
curated_df = pd.concat(curated_dict.values(), axis=1)
curated_df


# In[45]:


# Check the total count row per colomn

curated_df["Worried"].sum()


# #### Calculate percentage representation out of sample size (175) and transpose

# In[46]:


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


# In[23]:



#curated_transposed.to_excel("Age_demog.xlsx", sheet_name='Age_demo_vs_food_sec')


# ### generate plot for the above table

# In[31]:


import os
# Install bokeh package for color palletes
#os.system('pip install bokeh')


# In[19]:


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
                             rot = 0,
                             color = Viridis[5])


ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Age", bbox_to_anchor=(1.43, 1.03), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

#plt.savefig('age_fig4.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#== N/B facecolor=ax.get_facecolor() sets the backgrund as white


# ## 2. Gender (demographic)

# In[47]:


# Our gender data looks like
gender.head()


# In[48]:


# Merger the two columns into a single column representation 
temp = col_merger(gender)

#Select the merged data column, rename, and put in a new dataframe df
temp = temp.iloc[:,-1:]
gender_cpy = temp.rename(columns={"Male_Female": "Gender"}) 
gender_cpy.head()


# In[49]:


#Write out gender for regression analysis

gender_cpy.to_csv("./regr/gender_r.csv", sep=',', encoding='utf-8')


# In[50]:


# Join both gender and food security dataframes

df_merged = pd.concat([gender_cpy, df], axis=1)
df_merged.head()


# In[51]:


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
            
            #Extract True statements with associated data

            if 'True' in str(set(temp1['outcome1'])):
                
                temp_out = temp1[temp1['outcome1'].astype(str).str.contains('True')].reset_index(drop=True)
                
                temp_dict[gender.columns[0]] = temp_out['count'][0] #assign male
                
                
            if 'True' in str(set(temp1['outcome2'])):
                
                temp_out = temp1[temp1['outcome2'].astype(str).str.contains('True')].reset_index(drop=True)
                
                temp_dict[gender.columns[1]] = temp_out['count'][0] #assign female from column name of gender
                
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


# In[52]:


# Group by both gender and food security and count
# Select males (1) and worried (1) plus Females (2) and worried (1)
# N/B pass in the gender to obtain both names (male female)
curated_dict = food_sec_binary_yeses(df, gender_cpy, df_merged, gender)
#curated_dict = curated_dict[1]
curated_dict


# In[53]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[54]:


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


# In[55]:



curated_transposed.to_excel("Gender_Demog.xlsx", sheet_name='Gender_Demo_vs_food_sec')


# In[56]:


### Generate plot


# In[36]:


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
                             color = Viridis[3],
                             rot = 0)

ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Gender", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

plt.savefig('fig5.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#N/B facecolor=ax.get_facecolor() sets the backgrund as white


# ### 3. No. of people per HH

# In[83]:


hh_numbers.head()


# In[84]:


# Inspect data
hh_numbers.describe()


# ##### Generate groups/class from the above data

# #### CLASSES
# ##### 0-5 people
# ##### 5-10 people
# 

# In[85]:


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


# In[86]:


# Drop the people_per_HH column and only retain Group identifies ror grouping purposes
del hh_numbers_cpy['people_per_HH'] 


# In[87]:


#Write out household size for regr
hh_numbers_temp = hh_numbers_cpy.copy()
hh_numbers_temp.columns = ["Household_Size"]
hh_numbers_temp.to_csv("./regr/HH_size_r.csv", sep=',', encoding='utf-8')


# In[88]:


# Join with the food security data

df_merged = pd.concat([hh_numbers_cpy['Group_identifiers'], df], axis=1)
df_merged.head()


# In[89]:


#Generate an empty dataframe with the following column names
#0-5 people =< 5
#5-10 people = 5 - 10

columns = ["0-5 people", "5-10 people"]
index = [0]
hh_numbers_groupnames = pd.DataFrame(index=index, columns=columns)
hh_numbers_groupnames


# In[90]:


# Group by both food security and groups
# Extract foodsecure (1) for BOTH groups: 0-5 people (1)  5-10 people (2)

curated_dict = food_sec_binary_yeses(df, hh_numbers_cpy, df_merged, hh_numbers_groupnames)


# In[91]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[92]:


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


# In[46]:



curated_transposed.to_excel("ppl_per_HH_demog.xlsx", sheet_name='ppl_per_HH_vs_food_sec')


# In[47]:


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
                             color=Viridis[3],
                             rot = 0)


ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "People per HH", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

#plt.savefig('fig6.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#N/B facecolor=ax.get_facecolor() sets the backgrund as white


# #### 4. Years of schooling

# In[93]:


yrs_of_schlng.head()


# In[94]:


#Replace Nan with 2, and any value with 1

yrs_of_schlng_cpy = replacer(yrs_of_schlng)

yrs_of_schlng_cpy = yrs_of_schlng_cpy.drop(columns=['index']).astype(int)   #drop the newly added index column

yrs_of_schlng_cpy.head()


# In[96]:


#Slide Years of schooling into one column representing the 5 categories with 1, 2, 3, 4, 5

yrs_of_schlng_temp = yrs_of_schlng_cpy.copy()

count = 1
for i in yrs_of_schlng_temp.columns:
    yrs_of_schlng_temp[i] = yrs_of_schlng_temp[i].replace(2, 0)
    yrs_of_schlng_temp[i] = yrs_of_schlng_temp[i].replace(1, count)
    count += 1

#slide the columns into one
yrs_of_schlng_temp['Schooling_Years']= yrs_of_schlng_temp.sum(axis=1)

#Write out Ages (for regr)
yrs_of_schlng_temp['Schooling_Years'].to_csv("./regr/Schooling_Years_r.csv", sep=',', encoding='utf-8')


# In[97]:


# Our merged data looks like

df_merged = pd.concat([yrs_of_schlng_cpy, df], axis=1)
df_merged.head()


# In[98]:


curated_dict = food_sec_yeses(df, yrs_of_schlng_cpy, df_merged)
#curated_dict = curated_dict[1]
curated_dict


# In[52]:


# Convert the dictionary into a dataframe
# Side_join all the dataframes
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[53]:


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


# In[54]:



curated_transposed.to_excel("yrs_of_scl_demog.xlsx", sheet_name='yrs_of_schl_vs_food_sec')


# In[56]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


#plt.rcParams.update({'axes.facecolor':'white'})

plt.figure(facecolor='white') 

ax = curated_transposed.plot(x='FoodSecurity', 
                             y=list(curated_transposed.columns[1:]), 
                             #height=100,
                             ylim=(0,25),
                             kind="bar",
                             width=0.9,
                             color = Viridis[8],
                             rot = 0)


ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Years of schooling", bbox_to_anchor=(1.43, 1.03), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

plt.savefig('fig8.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())


# In[ ]:




