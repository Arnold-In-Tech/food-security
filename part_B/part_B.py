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
subcounty_B = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_B/1_subcounty.csv")
## sorghum_farmland_size
sorghum_farmland_size_B = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_B/2_sorghum_farmland_size.csv")
## farmers planting sorghum
#sorghum_farmers_B = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_B/3_sorghum_farmers.csv")
## No. of pple per HH
#age_B = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_B/4_Age.csv")
## years of schooling
#yrs_of_schlng_B = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_B/5_schooling_yrs.csv")
## years of schooling
#other_cereals_farmed_B = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_B/6_other_cereals_farmed.csv")


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

# In[18]:


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

# In[17]:


# Our gender data looks like
gender.head()


# In[18]:


# Merger the two columns into a single column representation 
temp = col_merger(gender)

#Select the merged data column, rename, and put in a new dataframe df
temp = temp.iloc[:,-1:]
gender_cpy = temp.rename(columns={"Male_Female": "Gender"}) 
gender_cpy.head()


# In[19]:


# Join both gender and food security dataframes

df_merged = pd.concat([gender_cpy, df], axis=1)
df_merged.head()


# In[20]:


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


# In[21]:


# Group by both gender and food security and count
# Select males (1) and worried (1) plus Females (2) and worried (1)
# N/B pass in the gender to obtain both names (male female)
curated_dict = food_sec_binary_yeses(df, gender_cpy, df_merged, gender)
#curated_dict = curated_dict[1]
curated_dict


# In[22]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[23]:


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


# In[24]:


### Generate plot


# In[25]:


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

# In[26]:


hh_numbers.head()


# In[27]:


# Inspect data
hh_numbers.describe()


# ##### Generate groups/class from the above data

# #### CLASSES
# ##### 0-5 people
# ##### 5-10 people
# 

# In[28]:


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


# In[29]:


# Drop the people_per_HH column and only retain Group identifies ror grouping purposes
del hh_numbers_cpy['people_per_HH'] 


# In[30]:


# Join with the food security data

df_merged = pd.concat([hh_numbers_cpy['Group_identifiers'], df], axis=1)
df_merged.head()


# In[31]:


#Generate an empty dataframe with the following column names
#0-5 people =< 5
#5-10 people = 5 - 10

columns = ["0-5 people", "5-10 people"]
index = [0]
hh_numbers_groupnames = pd.DataFrame(index=index, columns=columns)
hh_numbers_groupnames


# In[32]:


# Group by both food security and groups
# Extract foodsecure (1) for BOTH groups: 0-5 people (1)  5-10 people (2)

curated_dict = food_sec_binary_yeses(df, hh_numbers_cpy, df_merged, hh_numbers_groupnames)


# In[33]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[34]:


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


# In[35]:


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

# In[36]:


yrs_of_schlng.head()


# In[37]:


#Replace Nan with 2, and any value with 1

yrs_of_schlng_cpy = replacer(yrs_of_schlng)

yrs_of_schlng_cpy = yrs_of_schlng_cpy.drop(columns=['index']).astype(int)   #drop the newly added index column

yrs_of_schlng_cpy.head()


# In[38]:


# Our merged data looks like

df_merged = pd.concat([yrs_of_schlng_cpy, df], axis=1)
df_merged.head()


# In[39]:


curated_dict = food_sec_yeses(df, yrs_of_schlng_cpy, df_merged)
curated_dict = curated_dict[1]
curated_dict


# In[40]:


# Convert the dictionary into a dataframe
# Side_join all the dataframes
                    
curated_df = pd.concat(curated_dict.values(), axis=1)
curated_df


# In[41]:


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


# In[44]:


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
# # PART B
# 
# #==============  
# 

# ### B.(i) Age (HH info)
# 

# In[44]:


# print(age_B.shape)
# age_B.head()


# In[45]:


# age_cpy = replacer(age_B)

# age_cpy = age_cpy.drop(columns=['index']).astype(int)   #drop the newly added index column


# In[46]:


# # Coded age data looks like
# age_cpy.head()


# In[47]:


# # Our merged data looks like

# df_merged = pd.concat([age_cpy, df], axis=1)
# df_merged.head()


# In[48]:


# #Group by
# curated_dict = food_sec_yeses(df, age_cpy, df_merged)
# curated_dict = curated_dict[1]


# In[49]:


# # Convert the dictionary into a dataframe
# #Side_join all the dataframes
                    
# curated_df = pd.concat(curated_dict.values(), axis=1)
# curated_df


# In[50]:


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


# In[51]:



# curated_transposed.to_excel("Age_HH.xlsx", sheet_name='Age_HH_vs_food_sec')


# In[45]:


# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure


# #plt.rcParams.update({'axes.facecolor':'white'})

# plt.figure(facecolor='white') 

# ax = curated_transposed.plot(x='FoodSecurity', 
#                              y=list(curated_transposed.columns[1:]), 
#                              #height=100,
#                              ylim=(0,40),
#                              kind="bar",
#                              width=0.9,
#                              rot = 0)


# ax.set_ylabel("Percentage population", fontsize = 12)
# ax.set_xlabel("Food Security", fontsize = 12)

# plt.rcParams.update({'font.size': 14})

# ax.legend(ncol=1, title = "Age", bbox_to_anchor=(1.43, 1.03), fontsize = 12)

# plt.rcParams.update({'font.size': 8.8})

# plt.savefig('age_B_fig.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

# #== N/B facecolor=ax.get_facecolor() sets the backgrund as white


# #### B (ii) Subcounty

# In[46]:



subcounty_B.head()


# In[47]:



subcounty_cpy = replacer(subcounty_B)

subcounty_cpy = subcounty_cpy.drop(columns=['index']).astype(int)   #drop the newly added index column


# In[48]:


# Our merged data looks like

df_merged = pd.concat([subcounty_cpy, df], axis=1)
df_merged.head()


# In[49]:


#Group by
curated_dict = food_sec_yeses(df, subcounty_cpy, df_merged)
curated_dict = curated_dict[1]


# In[50]:


# Convert the dictionary into a dataframe
#Side_join all the dataframes
                    
curated_df = pd.concat(curated_dict.values(), axis=1)
curated_df


# In[55]:


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


# In[56]:



curated_transposed.to_excel("subcounty_HH.xlsx", sheet_name='subcounty_HH_vs_food_sec')


# In[57]:


# Rearrange columns to have Mwala at the end
cols = curated_transposed.columns
cols = list(cols[:-2]) + list(cols[-1:]) + list(cols[-2:-1])
curated_transposed = curated_transposed[cols]

curated_transposed.head()


# In[59]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from bokeh.palettes import Magma, Inferno, Plasma, Viridis, Cividis

plt.figure(facecolor='white') 

ax = curated_transposed.plot(x='FoodSecurity', 
                             y=list(curated_transposed.columns[1:]), 
                             #height=100,
                             ylim=(0,17.5),
                             kind="bar",
                             width=0.9,
                             color = Viridis[8],
                             rot = 0)


ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Subcounty", bbox_to_anchor=(1.0, 1.03), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

plt.savefig('subcounty_B_fig.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#== N/B facecolor=ax.get_facecolor() sets the backgrund as white


# In[42]:


#### (iii) sorghum_farmland_size_B


# In[43]:


sorghum_farmland_size_B.head()


# In[44]:



sorghum_farmland_size_B_cpy = replacer(sorghum_farmland_size_B)

sorghum_farmland_size_B_cpy = sorghum_farmland_size_B_cpy.drop(columns=['index']).astype(int)   #drop the newly added index column


# In[48]:


#Slide farmland size into one column representing the 6 categories with 1, 2, 3, 4, 5, 6

sorghum_farmland_size_B_temp = sorghum_farmland_size_B_cpy.copy()

count = 1
for i in sorghum_farmland_size_B_temp.columns:
    sorghum_farmland_size_B_temp[i] = sorghum_farmland_size_B_temp[i].replace(2, 0)
    sorghum_farmland_size_B_temp[i] = sorghum_farmland_size_B_temp[i].replace(1, count)
    count += 1

#slide the columns into one
sorghum_farmland_size_B_temp['S_FarmSize']= sorghum_farmland_size_B_temp.sum(axis=1)

#Write out Ages (for regr)
sorghum_farmland_size_B_temp['S_FarmSize'].to_csv("../regr/S_FarmSize_r.csv", sep=',', encoding='utf-8')
sorghum_farmland_size_B_temp


# In[45]:


# Our merged data looks like

df_merged = pd.concat([sorghum_farmland_size_B_cpy, df], axis=1)
df_merged.head()


# In[64]:


#Group by
curated_dict = food_sec_yeses(df, sorghum_farmland_size_B_cpy, df_merged)
curated_dict = curated_dict[1]


# In[65]:


# Convert the dictionary into a dataframe
#Side_join all the dataframes
                    
curated_df = pd.concat(curated_dict.values(), axis=1)
curated_df


# In[66]:


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


# In[67]:



curated_transposed.to_excel("sorghum_far_size_HH.xlsx", sheet_name='frm_sz_HH_vs_food_sec')


# In[69]:


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
                             color=Viridis[6],
                             rot = 0)


ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Sorghum farmland size", bbox_to_anchor=(1.0, 1.03), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

plt.savefig('sorghum_farmland_size_B_fig.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#== N/B facecolor=ax.get_facecolor() sets the backgrund as white


# #### (iii) sorghum_farmers_per_HH
# 

# In[71]:


#sorghum_farmers_B.head()


# In[72]:


# Inspect data
#sorghum_farmers_B.describe()


# In[ ]:





# In[73]:


# # Categorise the groups with unique decimal numbers (ie, 1, 2, 3, etc)

# sorghum_farmers_B_cpy = sorghum_farmers_B
# identifiers = {}
# for index, row in sorghum_farmers_B_cpy.iterrows():
#     if row['sorghum farmers per HH'] in list(range(1, 6, 1)):
#         identifiers[index] = 1 
        
#     elif row['sorghum farmers per HH'] in list(range(6, 11, 1)):
#         identifiers[index] = 2 
        
#     else:
#         print(row['sorghum farmers per HH'], "Not in range")


# # Convert identifiers dictionary to dataframe 
# IDs = pd.DataFrame(identifiers.items(), columns=['index', 'Group_identifiers'])
# IDs.set_index('index')

# #Join with hh_numbers_cpy
# sorghum_farmers_B_cpy = pd.concat([sorghum_farmers_B_cpy, IDs], axis=1).drop(['index'], axis=1)

# # Our new dataframe looks like
# sorghum_farmers_B_cpy


# In[74]:


# # Drop the people_per_HH column and only retain Group identifies ror grouping purposes
# del sorghum_farmers_B_cpy['sorghum farmers per HH'] 


# In[75]:


# # Join with the food security data

# df_merged = pd.concat([sorghum_farmers_B_cpy['Group_identifiers'], df], axis=1)
# df_merged.head()


# In[76]:


# #Generate an empty dataframe with the following column names
# #0-5 people =< 5
# #5-10 people = 5 - 10

# columns = ["0-5 people", "5-10 people"]
# index = [0]
# hh_numbers_groupnames = pd.DataFrame(index=index, columns=columns)
# hh_numbers_groupnames


# In[77]:


# # Group by both food security and groups
# # Extract foodsecure (1) for BOTH groups: 0-5 people (1)  5-10 people (2)

# curated_dict = food_sec_binary_yeses(df, sorghum_farmers_B_cpy, df_merged, hh_numbers_groupnames)


# In[78]:


# # Convert the dictionary into a dataframe
                    
# curated_df = pd.concat(curated_dict[1].values(), axis=1)
# curated_df


# In[79]:


# # Calculate percentage representation for each year
# curated_perc = curated_df/ 173 * 100
# curated_perc
# #curated_perc.index.name = 'Age'

# # Transpose the matrix
# curated_transposed = curated_perc.T
# curated_transposed

# curated_transposed = curated_transposed.reset_index()  #convert index of the dataframe into a column

# #rename the index column
# curated_transposed = curated_transposed.rename({'index': 'FoodSecurity'}, axis=1)
# curated_transposed


# In[80]:



# curated_transposed.to_excel("sorg_farmrs_per_HH.xlsx", sheet_name='sog_frm_per_HH_vs_food_sec')


# In[81]:


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

# ax.legend(ncol=1, title = "Sorghum farmers per HH", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

# plt.rcParams.update({'font.size': 8.8})

# plt.savefig('sorghum_farmers_B.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

# #N/B facecolor=ax.get_facecolor() sets the backgrund as white


# #### (iv) Years of schooling
# 

# In[82]:


# yrs_of_schlng_B.head()


# In[83]:


# #Replace Nan with 2, and any value with 1

# yrs_of_schlng_B_cpy = replacer(yrs_of_schlng_B)

# yrs_of_schlng_B_cpy = yrs_of_schlng_B_cpy.drop(columns=['index']).astype(int)   #drop the newly added index column

# yrs_of_schlng_B_cpy.head()


# In[84]:


# # Our merged data looks like

# df_merged = pd.concat([yrs_of_schlng_B_cpy, df], axis=1)
# df_merged.head()


# In[85]:


# curated_dict = food_sec_yeses(df, yrs_of_schlng_B_cpy, df_merged)
# curated_dict = curated_dict[1]
# #curated_dict


# In[86]:


# # Convert the dictionary into a dataframe
# # Side_join all the dataframes
                    
# curated_df = pd.concat(curated_dict.values(), axis=1)
# curated_df


# In[87]:


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


# In[88]:



# curated_transposed.to_excel("yrs_of_sclng_HH.xlsx", sheet_name='yrs_of_sch_vs_food_sec')


# In[89]:


# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure


# #plt.rcParams.update({'axes.facecolor':'white'})

# plt.figure(facecolor='white') 

# ax = curated_transposed.plot(x='FoodSecurity', 
#                              y=list(curated_transposed.columns[1:]), 
#                              #height=100,
#                              ylim=(0,40),
#                              kind="bar",
#                              width=0.9,
#                              rot = 0)


# ax.set_ylabel("% Percentage population", fontsize = 12)
# ax.set_xlabel("Food Security", fontsize = 12)

# plt.rcParams.update({'font.size': 14})

# ax.legend(ncol=1, title = "Years of schooling", bbox_to_anchor=(1.43, 1.03), fontsize = 12)

# plt.rcParams.update({'font.size': 8.8})

# plt.savefig('yrs_of_schlng_B.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())


# #### (v) other cereals farmed

# In[90]:


# other_cereals_farmed_B.head()


# In[91]:



# other_cereals_farmed_B_cpy = replacer(other_cereals_farmed_B)

# other_cereals_farmed_B_cpy = other_cereals_farmed_B_cpy.drop(columns=['index']).astype(int)   #drop the newly added index column


# In[92]:


# # Our merged data looks like

# df_merged = pd.concat([other_cereals_farmed_B_cpy, df], axis=1)
# df_merged.head()


# In[93]:


# #Group by
# curated_dict = food_sec_yeses(df, other_cereals_farmed_B_cpy, df_merged)
# curated_dict = curated_dict[1]


# In[94]:


# # Convert the dictionary into a dataframe
# #Side_join all the dataframes
                    
# curated_df = pd.concat(curated_dict.values(), axis=1)
# curated_df


# In[95]:


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


# In[96]:



# curated_transposed.to_excel("other_cereals_farmed_HH.xlsx", sheet_name='other_cer_frm_vs_food_sec')


# In[97]:


# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure


# #plt.rcParams.update({'axes.facecolor':'white'})

# plt.figure(facecolor='white') 

# ax = curated_transposed.plot(x='FoodSecurity', 
#                              y=list(curated_transposed.columns[1:]), 
#                              #height=100,
#                              ylim=(0,70),
#                              kind="bar",
#                              width=0.9,
#                              rot = 0)


# ax.set_ylabel("Percentage population", fontsize = 12)
# ax.set_xlabel("Food Security", fontsize = 12)

# plt.rcParams.update({'font.size': 14})

# ax.legend(ncol=1, title = "Other cereals farmed", bbox_to_anchor=(1.0, 1.03), fontsize = 12)

# plt.rcParams.update({'font.size': 8.8})

# plt.savefig('other_cereals_farmed_fig.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

# #== N/B facecolor=ax.get_facecolor() sets the backgrund as white


# In[ ]:




