#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd
import numpy as np


# ### Import data

# In[80]:


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



#Horizontal Int (Contracts)
## Coop
coop_in_place_D = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_D/1_coop_in_place.csv")
## Coop membership
coop_member_D = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_D/2_membership.csv")
## market participation
market_participation_D = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_D/4_market_participation.csv")
## Receipt to other_farm_services due to membership
other_farm_services_in_contract_D = pd.read_csv("/home/arnold/Desktop/temp_file/ray_plots/part_D/5_other_farm_services.csv")


# ### Food Security
# #### Inspect food security data: 
# #### Replace both Yes (1) and No (2) columns with a single column containing both outcomes
# 

# In[81]:


print(food_sec.shape)
food_sec.head()


# In[82]:


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


# In[83]:


food_sec = col_merger(food_sec)

# Our merged data looks like
food_sec.tail()    


# In[84]:


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

# In[85]:


print(age.shape)
age.head()


# #### Age: Replace NaN with 2 and any other age values with 1. 
# ##### > 1 indicates that the interviewee is part of the age group, 
# ##### > 2 indicates the interviewee is not part of that age group 

# In[86]:




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


# In[87]:


# The two data frames look like

print(df.tail())
print(age_cpy.tail())


# In[88]:


#Check if there are any other number besides 1 and 2 in the dataframe
print(food_sec.eq(0).any().any()) 
print(food_sec.eq(3).any().any())
print(food_sec.eq(4).any().any())

#There are zeros and 3s in our data
#Meaning: There are empty rows
#         There were entries with both 1s and 2s (both Yes and No)


# In[89]:


# Our merged data looks like

df_merged = pd.concat([age_cpy, df], axis=1)
df_merged.head(10)


# ### Generate a dataframe of all Age counts (Yes or No) grouped based each food security variable option (Yes or No) 
# 
# #### Demo

# In[90]:



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


# In[91]:



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

                


# In[92]:


curated_dict = food_sec_yeses(df, age_cpy, df_merged)
curated_dict = curated_dict[1]
curated_dict


# In[93]:


# Convert the dictionary into a dataframe
#Side_join all the dataframes
                    
curated_df = pd.concat(curated_dict.values(), axis=1)
curated_df


# #### Calculate percentage representation out of sample size (175) and transpose

# In[94]:


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


# ### generate plot for the above table

# In[95]:


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

# In[96]:


# Our gender data looks like
gender.head()


# In[97]:


# Merger the two columns into a single column representation 
temp = col_merger(gender)

#Select the merged data column, rename, and put in a new dataframe df
temp = temp.iloc[:,-1:]
gender_cpy = temp.rename(columns={"Male_Female": "Gender"}) 
gender_cpy.head()


# In[98]:


# Join both gender and food security dataframes

df_merged = pd.concat([gender_cpy, df], axis=1)
df_merged.head()


# In[99]:


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


# In[100]:


# Group by both gender and food security and count
# Select males (1) and worried (1) plus Females (2) and worried (1)
# N/B pass in the gender to obtain both names (male female)
curated_dict = food_sec_binary_yeses(df, gender_cpy, df_merged, gender)
#curated_dict = curated_dict[1]
curated_dict


# In[101]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[102]:


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


# In[103]:


### Generate plot


# In[104]:


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

# In[105]:


hh_numbers.head()


# In[106]:


# Inspect data
hh_numbers.describe()


# ##### Generate groups/class from the above data

# #### CLASSES
# ##### 0-5 people
# ##### 5-10 people
# 

# In[107]:


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


# In[108]:


# Drop the people_per_HH column and only retain Group identifies ror grouping purposes
del hh_numbers_cpy['people_per_HH'] 


# In[109]:


# Join with the food security data

df_merged = pd.concat([hh_numbers_cpy['Group_identifiers'], df], axis=1)
df_merged.head()


# In[110]:


#Generate an empty dataframe with the following column names
#0-5 people =< 5
#5-10 people = 5 - 10

columns = ["0-5 people", "5-10 people"]
index = [0]
hh_numbers_groupnames = pd.DataFrame(index=index, columns=columns)
hh_numbers_groupnames


# In[111]:


# Group by both food security and groups
# Extract foodsecure (1) for BOTH groups: 0-5 people (1)  5-10 people (2)

curated_dict = food_sec_binary_yeses(df, hh_numbers_cpy, df_merged, hh_numbers_groupnames)


# In[112]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[113]:


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


# In[114]:


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

# In[115]:


yrs_of_schlng.head()


# In[116]:


#Replace Nan with 2, and any value with 1

yrs_of_schlng_cpy = replacer(yrs_of_schlng)

yrs_of_schlng_cpy = yrs_of_schlng_cpy.drop(columns=['index']).astype(int)   #drop the newly added index column

yrs_of_schlng_cpy.head()


# In[117]:


# Our merged data looks like

df_merged = pd.concat([yrs_of_schlng_cpy, df], axis=1)
df_merged.head()


# In[118]:


curated_dict = food_sec_yeses(df, yrs_of_schlng_cpy, df_merged)
curated_dict = curated_dict[1]
curated_dict


# In[119]:


# Convert the dictionary into a dataframe
# Side_join all the dataframes
                    
curated_df = pd.concat(curated_dict.values(), axis=1)
curated_df


# In[120]:


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


# In[121]:


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
# # PART D
# 
# #==============  
# 

# ### D.(i) receipt of other farm services due to membership to cooperative society
# 

# In[122]:


print(other_farm_services_in_contract_D.shape)
other_farm_services_in_contract_D.head()


# In[123]:


other_farm_services_in_contract_D_cpy = replacer(other_farm_services_in_contract_D)

other_farm_services_in_contract_D_cpy = other_farm_services_in_contract_D_cpy.drop(columns=['index']).astype(int)   #drop the newly added index column


# In[124]:


# Coded age data looks like
other_farm_services_in_contract_D_cpy.head()


# In[125]:


# Our merged data looks like

df_merged = pd.concat([other_farm_services_in_contract_D_cpy, df], axis=1)
df_merged.head()


# In[126]:


#Group by
curated_dict = food_sec_yeses(df, other_farm_services_in_contract_D_cpy, df_merged)
curated_dict = curated_dict[1]


# In[127]:


# Convert the dictionary into a dataframe
#Side_join all the dataframes
                    
curated_df = pd.concat(curated_dict.values(), axis=1)
curated_df


# In[128]:


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


# In[51]:



curated_transposed.to_excel("other_farm_serv_in_cont.xlsx", sheet_name='other_farm_serv_vs_food_sec')


# In[53]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from bokeh.palettes import Magma, Inferno, Plasma, Viridis, Cividis


#plt.rcParams.update({'axes.facecolor':'white'})

plt.figure(facecolor='white') 

ax = curated_transposed.plot(x='FoodSecurity', 
                             y=list(curated_transposed.columns[1:]), 
                             #height=100,
                             ylim=(0,15),
                             kind="bar",
                             width=0.9,
                             color=Cividis[8],
                             rot = 0)



ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Other farms services\n in contract", bbox_to_anchor=(1., 1.03), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

plt.savefig('other_farm_services_in_contract_D.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#== N/B facecolor=ax.get_facecolor() sets the backgrund as white


# #### D (ii) cooperative_in_place

# In[129]:


print(coop_in_place_D.shape)
coop_in_place_D.head()


# In[130]:


# Merger the two columns into a single column representation 
temp = col_merger(coop_in_place_D)

#Select the merged data column, rename, and put in a new dataframe df
temp = temp.iloc[:,-1:]
coop_in_place_D_cpy = temp.rename(columns={"Yes_No": "Coop_in_place"}) 
coop_in_place_D_cpy.head()


# In[131]:


#Write out Coop in Place (for regr)
# coop_in_place_D_cpy['Coop_in_place'].to_csv("../regr/Coop_in_place_r.csv", sep=',', encoding='utf-8')


# In[132]:


# Join with the food security data

df_merged = pd.concat([coop_in_place_D_cpy, df], axis=1)
df_merged.head()


# In[133]:


#Generate an empty dataframe with the following column names

columns = ["Yes", "No"]
index = [0]
coop_in_place_D_groupnames = pd.DataFrame(index=index, columns=columns)
coop_in_place_D_groupnames


# In[134]:


# Group by both food security and groups
# Extract foodsecure (1) for BOTH groups: 0-5 people (1)  5-10 people (2)

curated_dict = food_sec_binary_yeses(df, coop_in_place_D_cpy, df_merged, coop_in_place_D_groupnames)


# In[135]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[136]:


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


# In[137]:



curated_transposed.to_excel("cooper_in_place.xlsx", sheet_name='coop_in_place_vs_food_sec')


# In[138]:


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
                             color=Cividis[3],
                             rot = 0)


ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Cooperative in place", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

plt.savefig('Coop_in_place_C.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#N/B facecolor=ax.get_facecolor() sets the backgrund as white


# #### C (iii). Cooperative member

# In[139]:


print(coop_member_D.shape)
coop_member_D.head()


# In[140]:


# Merger the two columns into a single column representation 
temp = col_merger(coop_member_D)

#Select the merged data column, rename, and put in a new dataframe df
temp = temp.iloc[:,-1:]
coop_member_D_cpy = temp.rename(columns={"Yes_No": "coop_membership"}) 
coop_member_D_cpy.head()


# In[63]:


#Write out Contracted (for regr)
coop_member_D_cpy['coop_membership'].to_csv("../regr/coop_member_r.csv", sep=',', encoding='utf-8')


# In[64]:


# Join with the food security data

df_merged = pd.concat([coop_member_D_cpy, df], axis=1)
df_merged.head()


# In[65]:


#Generate an empty dataframe with the following column names

columns = ["Yes", "No"]
index = [0]
coop_member_D_groupnames = pd.DataFrame(index=index, columns=columns)
coop_member_D_groupnames


# In[66]:


# Group by both food security and groups
# Extract foodsecure (1) for BOTH groups: 0-5 people (1)  5-10 people (2)

curated_dict = food_sec_binary_yeses(df, coop_member_D_cpy, df_merged, coop_member_D_groupnames)


# In[67]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[68]:


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


# In[ ]:



curated_transposed.to_excel("coop_member.xlsx", sheet_name='coop_member_vs_food_sec')


# In[ ]:


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
                             color=Cividis[3],
                             rot = 0)

ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "cooperative member", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

plt.savefig('coop_member_D.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#N/B facecolor=ax.get_facecolor() sets the backgrund as white


# #### D. (iV) Open market_participation_D
# 

# In[69]:


print(market_participation_D.shape)
market_participation_D.head()


# In[70]:


# Merger the two columns into a single column representation 
temp = col_merger(market_participation_D)

#Select the merged data column, rename, and put in a new dataframe df
temp = temp.iloc[:,-1:]
market_participation_D_cpy = temp.rename(columns={"Yes_No": "market_participation"}) 
market_participation_D_cpy.head()


# In[71]:


#Write out Market Participation (for regr)
market_participation_D_cpy['market_participation'].to_csv("../regr/D_M_Participation_r.csv", sep=',', encoding='utf-8')


# In[72]:


# Join with the food security data

df_merged = pd.concat([market_participation_D_cpy, df], axis=1)
df_merged.head()


# In[73]:


#Generate an empty dataframe with the following column names

columns = ["Yes", "No"]
index = [0]
market_participation_D_groupnames = pd.DataFrame(index=index, columns=columns)
market_participation_D_groupnames


# In[74]:


# Group by both food security and groups
# Extract foodsecure (1) for BOTH groups: 0-5 people (1)  5-10 people (2)

curated_dict = food_sec_binary_yeses(df, market_participation_D_cpy, df_merged, market_participation_D_groupnames)


# In[75]:


# Convert the dictionary into a dataframe
                    
curated_df = pd.concat(curated_dict[1].values(), axis=1)
curated_df


# In[76]:


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


# In[ ]:



curated_transposed.to_excel("market_participation.xlsx", sheet_name='market_part_vd_food_sec')


# In[ ]:


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
                             color=Cividis[3],
                             rot = 0)

ax.set_ylabel("Share of population living in a\n household reporting each element\n of food insecurity (percentage)", fontsize = 12)
ax.set_xlabel("FIES questions", fontsize = 12)

plt.rcParams.update({'font.size': 14})

ax.legend(ncol=1, title = "Open market\n participation", bbox_to_anchor=(1.01, 1.02), fontsize = 12)

plt.rcParams.update({'font.size': 8.8})

plt.savefig('market_participation_D.png',bbox_inches='tight', format='png', dpi=300, edgecolor='none', facecolor=ax.get_facecolor())

#N/B facecolor=ax.get_facecolor() sets the backgrund as white


# In[ ]:




