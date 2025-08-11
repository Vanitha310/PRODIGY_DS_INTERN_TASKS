import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

df = pd.read_csv("US_Accidents_March23.csv")  

print(df.shape)
print(df.columns)
print(df.head())

df = df.dropna(subset=['Start_Lat', 'Start_Lng', 'Weather_Condition', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Start_Time'])

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')

df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.day_name()
df['Month'] = df['Start_Time'].dt.month_name()

plt.figure(figsize=(8,5))
sns.countplot(x='Hour', hue='Hour', data=df, palette='coolwarm', legend=False)
plt.title("Accidents by Hour of the Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.show()

top_weather = df['Weather_Condition'].value_counts().nlargest(10)
plt.figure(figsize=(10,6))
sns.barplot(y=top_weather.index, x=top_weather.values, hue=top_weather.index, palette='viridis', legend=False)
plt.title("Top 10 Weather Conditions During Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("Weather Condition")
plt.show()

road_features = ['Amenity', 'Bump', 'Crossing', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
road_counts = {feat: df[feat].sum() for feat in road_features if feat in df.columns}
plt.figure(figsize=(10,5))
sns.barplot(x=list(road_counts.keys()), y=list(road_counts.values()), hue=list(road_counts.keys()), palette='magma', legend=False)
plt.title("Accidents Involving Specific Road Features")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

sample_df = df.sample(1000)  
map_accidents = folium.Map(location=[sample_df['Start_Lat'].mean(), sample_df['Start_Lng'].mean()], zoom_start=4)
for idx, row in sample_df.iterrows():
    folium.CircleMarker(location=[row['Start_Lat'], row['Start_Lng']],
                        radius=2,
                        color='red',
                        fill=True,
                        fill_opacity=0.4).add_to(map_accidents)

map_accidents.save("accident_hotspots_map.html")
print("Map saved as accident_hotspots_map.html")

plt.figure(figsize=(8,5))
sns.countplot(x='DayOfWeek', hue='DayOfWeek', data=df, order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], palette='cubehelix', legend=False)
plt.title("Accidents by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Number of Accidents")
plt.show()
