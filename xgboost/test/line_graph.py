import matplotlib.pyplot as plt
import pandas as pd

# Data from the table
data = {
    'Dataset': [
        'Iris', 'Pima Indians', 'Heart Attack', 'Wine', 'AIDS Study 175',
        'Breast Cancer', 'Ionosphere', 'Biodegradation', 'Sonar'
    ],
    'FA Proportion': [0.500, 0.500, 0.538, 0.385, 0.478, 0.200, 0.206, 0.220, 0.167],
    'GA Proportion': [0.500, 0.625, 0.231, 0.385, 0.391, 0.133, 0.176, 0.171, 0.100]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['Dataset'], df['FA Proportion'], marker='o', label='FA Proportion')
plt.plot(df['Dataset'], df['GA Proportion'], marker='o', label='GA Proportion')

# Add titles and labels
plt.title('Proportion of Retained Features for FA and GA Relative to the Naive Control')
plt.xlabel('Dataset')
plt.ylabel('Proportion')
plt.xticks(rotation=45)
plt.ylim(0, 0.7)
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()