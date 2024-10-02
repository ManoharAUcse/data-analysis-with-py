import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = sns.load_dataset('tips')


print(df.head())


plt.figure(figsize=(12, 6))


for i, column in enumerate(df.select_dtypes(include=['float64', 'int64']).columns, 1):
    plt.subplot(2, len(df.select_dtypes(include=['float64', 'int64']).columns) // 2 + 1, i)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))

for i, column in enumerate(df.select_dtypes(include=['float64', 'int64']).columns, 1):
    plt.subplot(2, len(df.select_dtypes(include=['float64', 'int64']).columns) // 2 + 1, i)
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
Corr_matrix = df.corr(numeric_only=True)
# Generate a heatmap
sns.heatmap(Corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Corr Matrix')
plt.show()
