import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = 'output.csv'

df = pd.read_csv(filename)

expected_columns = [
    "Accuracy",
    "Neurons in hidden layer 1",
    "Neurons in hidden layer 2",
    "Learning rate",
    "Batch size"
]
assert all(col in df.columns for col in expected_columns), "csv file contains unexpected column names"

sns.set(style="whitegrid")

for column in expected_columns[1:]:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=column, y="Accuracy", hue="Accuracy", palette="viridis", s=60)
    plt.title(f"Accuracy vs {column}")
    plt.tight_layout()
    plt.show()

best_row = df.loc[df['Accuracy'].idxmax()]

print("\nbest parameters combination for highest accuracy:")
print(best_row.to_string())
