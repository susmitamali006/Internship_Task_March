import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv("states_all.csv")

print(df.head())

print(df.tail())

print(df.info())

print(df.describe())

# Count of total number of rows and columns in dataset
print(df.shape)

# Column names 
print("Column names :", df.columns)

# Standardise column names to avoid errors 
df.columns = (
    df.columns 
    .str.lower()
    .str.strip()
    .str.replace(" ","_")
)
print(df.columns)

# Count missing values per column
print(df.isnull().sum())

# Fill missing funding values with the average of the column
df['total_expenditure'] = df['total_expenditure'].fillna(df['total_expenditure'].mean())

# same for total revenue
df['total_revenue'] = df['total_revenue'].fillna(df['total_revenue'].mean())

#numeric columns with their own mean 
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

#object columns with 'Unknown'
object_cols = df.select_dtypes(include=['object']).columns
df[object_cols] = df[object_cols].fillna('Unknown')

# Count after handleling null values 
print(df.isnull().sum())

# Duplicate count
print("Duplicate Count:", df.duplicated().sum())

# check data types of columns 
print("Data Types:", df.dtypes)

# Calculate correlation
correlation = df['total_expenditure'].corr(df['avg_reading_4_score'])
print(f"The Correlation Coefficient (r) is: {correlation:.4f}")

# plot for National Education funding over time 
yearly_data = df.groupby('year').mean(numeric_only=True)
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(yearly_data.index, yearly_data['total_expenditure'], color='purple', marker='o')
plt.title('National Education Funding Over Time')
plt.ylabel('Total Expenditure ($)')
plt.grid(True)
plt.show()

# National Literacy score over time
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 2)
plt.plot(yearly_data.index, yearly_data['avg_reading_4_score'], color='orange', marker='s')
plt.title('National Literacy (Reading) Scores Over Time')
plt.xlabel('Year')
plt.ylabel('Avg Reading Score')
plt.grid(True)
plt.tight_layout()
plt.show()


# Group by State 
state_data = df.groupby('state')[['total_expenditure', 'avg_reading_4_score']].mean(numeric_only=True)
top_five_states = state_data.sort_values('total_expenditure', ascending=False).head(5)
top_five_literacy = state_data.sort_values('avg_reading_4_score', ascending=False).head(5)
print("Top 5 States:")
print(top_five_states)
print("\nTop 5 Literacy Performance States:")
print(top_five_literacy)

