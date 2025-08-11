import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv", skiprows=4)

print(df.head())

year = '2020'
plt.figure(figsize=(10, 6))
sns.histplot(df[year].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title(f'Population Distribution Across Countries ({year})', fontsize=16)
plt.xlabel('Population')
plt.ylabel('Number of Countries')
plt.show()

country = 'India'
country_data = df[df['Country Name'] == country].iloc[0, 4:-1]  
years = country_data.index.astype(int)
pop_values = country_data.values

plt.figure(figsize=(12, 6))
sns.barplot(x=years, y=pop_values,hue=years, palette='pastel',legend=False)
plt.xticks(rotation=45)
plt.title(f'Population of {country} Over Time', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Population')
plt.show()
