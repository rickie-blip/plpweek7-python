# plpweek7-python
# 📊 Analyzing Data with Pandas and Visualizing Results with Matplotlib

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style using Seaborn for better aesthetics
sns.set(style="whitegrid")

# Step 2: Load and Explore the Dataset

# Load your dataset (adjust the path as needed)
# Example: Iris dataset from sklearn
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['species'] = pd.Categorical.from_codes(data.target, data.target_names)

# Display the first few rows of the dataset
df.head()

# Explore the structure of the dataset
print("Dataset Information:")
df.info()

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Data Cleaning (If needed)
# In this case, no missing values in the Iris dataset, but if your dataset has missing values:
# df = df.dropna()  # Drop missing values
# or
# df = df.fillna(df.mean())  # Fill missing values with the column mean

# Step 4: Basic Data Analysis

# Descriptive statistics of the numerical columns
print("\nDescriptive Statistics:")
print(df.describe())

# Group data by a categorical column (e.g., species) and calculate the mean
grouped = df.groupby('species').mean()
print("\nGrouped Data by Species:")
print(grouped)

# Step 5: Data Visualization

# 1. Line Chart (Example: Trends over time, modify with your own time-based data)
plt.figure(figsize=(10, 6))
sns.lineplot(x='species', y='sepal length (cm)', data=df)
plt.title('Sepal Length per Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.show()

# 2. Bar Chart (Comparison of numerical values across categories)
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='sepal width (cm)', data=df)
plt.title('Sepal Width per Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width (cm)')
plt.show()

# 3. Histogram (Distribution of a numerical column)
plt.figure(figsize=(10, 6))
sns.histplot(df['petal length (cm)'], kde=True, bins=20)
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter Plot (Relationship between two numerical columns)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()


