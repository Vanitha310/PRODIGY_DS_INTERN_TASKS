import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("twitter_training.csv", header=None, names=["Tweet_ID", "Entity", "Sentiment", "Tweet"])
val_df = pd.read_csv("twitter_validation.csv", header=None, names=["Tweet_ID", "Entity", "Sentiment", "Tweet"])

df = pd.concat([train_df, val_df], ignore_index=True)

print(df.head())
print(f"Total records: {len(df)}")

plt.figure(figsize=(6,4))
sns.countplot(x='Sentiment', data=df,hue='Sentiment', palette='pastel', legend=False)
plt.title("Overall Sentiment Distribution", fontsize=14)
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.show()

top_entities = df['Entity'].value_counts().head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=top_entities.values, y=top_entities.index,hue=top_entities.index, palette='viridis', legend=False)
plt.title("Top 10 Most Mentioned Entities", fontsize=14)
plt.xlabel("Number of Mentions")
plt.ylabel("Entity")
plt.show()

top_entities_list = top_entities.index.tolist()
df_top = df[df['Entity'].isin(top_entities_list)]
plt.figure(figsize=(10,6))
sns.countplot(x='Entity', hue='Sentiment', data=df_top, palette='pastel')
plt.title("Sentiment Distribution for Top Entities", fontsize=14)
plt.xticks(rotation=45)
plt.show()
