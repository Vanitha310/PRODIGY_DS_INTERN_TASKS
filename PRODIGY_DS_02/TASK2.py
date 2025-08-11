
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("train.csv")
print("First 5 rows: \n", df.head())
print("\n Shape of Dataset:", df.shape)
print("\n Missing values:\n", df.isnull().sum())
print("\n Data types:\n", df.dtypes)

df['Age']= df['Age'].fillna(df['Age'].mean())

df['Embarked']= df['Embarked'].fillna(df['Embarked'].mode()[0])

df.drop(columns=['Cabin'], inplace=True)

print("\n Summary Statistics:\n", df.describe())

plt.figure(figsize=(6,4))
sns.countplot(x='Survived', hue='Survived', data=df, palette='pastel', legend=False)
plt.title('Survival Count')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df,hue='Sex', palette='Set2', legend=False)
plt.title('Survival Rate by Gender')
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df, hue='Pclass', palette='Set1', legend=False)
plt.title('Survival Rate by Passenger Class')
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Age', data=df, hue='Survived', palette='cool', legend=False)
plt.title('Age vs Survival')
plt.show()

plt.figure(figsize=(8,6))
numeric_df= df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

print("\n Insights:")
print("- Female had a higher survival rate than males.")
print("- 1st class passengers had better survival chances.")
print("- Younger passengers had slightly higher survival rate.")




