# 🧪 COVID-19 Clinical Trials Analysis

## 📌 Overview

This data science project analyzes real-world clinical trial data related to COVID-19 to extract meaningful insights into the global research response to the pandemic. It covers exploratory data analysis (EDA), data preprocessing, and basic data transformation techniques to understand how clinical trials were structured and categorized.

---

## 🗃️ Dataset

- **Source:** A CSV file titled `COVID clinical trials.csv`
- **Key Features:**
  - `Study Title`
  - `Sponsor`
  - `Conditions`
  - `Phase`
  - `Interventions`
  - `Status`
  - `Start and End Dates`

---

## 🔍 Key Steps Performed

- ✅ Data loading and inspection (`df.info()`, `.shape`)
- ✅ Exploratory Data Analysis (EDA)
  - Study phases distribution
  - Trial status analysis
  - Sponsor and intervention breakdown
- ✅ Data preprocessing:
  - Z-score normalization
  - Label Encoding
  - Train-test split preparation

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:**
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `scipy.stats`

---

## 📊 Sample Code

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("COVID clinical trials.csv")
df.info()
df.shape
