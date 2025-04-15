#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[2]:


df = pd.read_csv("C:\\Users\\91721\\Downloads\\COVID clinical trials.csv")
df.info()


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.describe()


# In[7]:


df.size


# In[8]:


df.columns


# In[9]:


df.isnull().sum()


# In[10]:


df.info()


# In[11]:


df.info()


# In[12]:


df.isnull().sum()


# In[13]:


df.isnull()


# In[14]:


# Data Cleaning

# 1. Drop columns with more than 95% missing values
df_cleaned = df.drop(columns=['Results First Posted', 'Study Documents'])

# 2. Fill missing text fields with 'Unknown'
text_cols = df_cleaned.select_dtypes(include='object').columns
df_cleaned[text_cols] = df_cleaned[text_cols].fillna('Unknown')

# 3. Fill missing numeric columns with median
df_cleaned['Enrollment'] = df_cleaned['Enrollment'].fillna(df_cleaned['Enrollment'].median())

# 4. Convert date columns to datetime
date_cols = ['Start Date', 'Primary Completion Date', 'Completion Date', 
             'First Posted', 'Last Update Posted']
for col in date_cols:
    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')

# 5. Standardize text formatting: strip whitespace and title case
for col in text_cols:
    df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.title()


# In[15]:


print(df_cleaned.columns)


# In[16]:


print(df_cleaned.isnull().sum())   # here missing values are cleaned 


# In[17]:


print(df_cleaned.shape)    # 2 unneccesary coloumns removed


# In[18]:


print(df_cleaned.head())


# In[19]:


print(df_cleaned[['Status', 'Conditions']].sample(5))


# In[20]:


print(df_cleaned[['Start Date', 'Completion Date']].head())


# In[21]:


print(df_cleaned[['Start Date', 'Completion Date']].dtypes)


# In[22]:


#EDA

print("🔍 Dataset Info:")
print(df.info())


# In[23]:


# Summary statistics
print("\n📈 Descriptive Statistics:")
print(df.describe(include='all'))


# In[24]:


# Check unique values for categorical columns
print("\n🧩 Unique values per categorical column:")
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")


# In[25]:


# Correlation heatmap (for numerical features)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("📌 Correlation Heatmap")
plt.show()


# In[26]:


# Distribution of target variable if applicable (e.g., Status)
if 'Status' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x='Status', order=df['Status'].value_counts().index)
    plt.title("✅ Clinical Trial Status Distribution")
    plt.xticks(rotation=45)
    plt.show()


# In[27]:


print(df['Age'].value_counts())
df['Age'].value_counts().plot(kind='bar', title='Age Group Distribution')


# In[28]:


status_phase = pd.crosstab(df['Status'], df['Phases'])
print(status_phase)
status_phase.plot(kind='bar', stacked=True, title='Status vs.Phases')


# In[29]:


# Time Series analysis

# --- Convert date column ---
# Replace 'Start Date' with your actual date column name
if 'Start Date' in df.columns:
    df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')

    # Drop rows where date conversion failed
    df = df.dropna(subset=['Start Date'])

    # Create a 'Year-Month' column
    df['YearMonth'] = df['Start Date'].dt.to_period('M')

    # Count trials per month
    time_series = df.groupby('YearMonth').size().reset_index(name='Trial Count')

    # Convert 'YearMonth' back to datetime for plotting
    time_series['YearMonth'] = time_series['YearMonth'].dt.to_timestamp()

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=time_series, x='YearMonth', y='Trial Count', marker='o')
    plt.title("📈 COVID Clinical Trials Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Trials Started")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ 'Start Date' column not found. Please check the column name.")


# In[30]:


# outlier detection using boxplot
# Replace 'Enrollment' with a relevant numeric column from your dataset
if 'Enrollment' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Enrollment')
    plt.title("📦 Boxplot of Enrollment Numbers")
    plt.xlabel("Enrollment")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Column 'Enrollment' not found. Replace with a numeric column name.")


# In[31]:


# outlier detection using Z score

if 'Enrollment' in df.columns:
    df['zscore'] = zscore(df['Enrollment'].dropna())
    outliers = df[np.abs(df['zscore']) > 3]  # threshold for outliers
    print(f"🚨 Found {len(outliers)} outliers in 'Enrollment' using Z-score > 3")
    print(outliers[['Enrollment', 'zscore']].head())
else:
    print("⚠️ Column 'Enrollment' not found.")


# In[32]:


# outlier detection using IQR 

if 'Enrollment' in df.columns:
    Q1 = df['Enrollment'].quantile(0.25)
    Q3 = df['Enrollment'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr = df[(df['Enrollment'] < lower_bound) | (df['Enrollment'] > upper_bound)]
    print(f"📉 Found {len(outliers_iqr)} outliers in 'Enrollment' using IQR method.")
else:
    print("⚠️ Column 'Enrollment' not found.")


# In[33]:


# --- Replace 'Enrollment' with your numeric column of interest ---
col = 'Enrollment'

if col in df.columns:
    # Drop missing values in that column
    df = df.dropna(subset=[col])

    # Calculate IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter data to remove outliers
    df_no_outliers = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(f"✅ Removed outliers from '{col}'.")
    print(f"Original shape: {df.shape}")
    print(f"After removing outliers: {df_no_outliers.shape}")
else:
    print(f"⚠️ Column '{col}' not found. Please check the column name.")


# In[34]:


# Print column names
print("Columns:\n", df.columns.tolist())

# Print top 5 values in each column
for col in df.columns:
    print(f"\n🔹 Column: {col}")
    print(df[col].value_counts().head())


# In[35]:


for col in df.columns:
    print(f"'{col}'")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




