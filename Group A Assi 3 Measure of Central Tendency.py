


# ---------------------------- Iris dataset ( Question 2 ) -------------------------------

import pandas as pd

# Load the iris dataset
iris_data = pd.read_csv('/content/Iris.csv')

# Filter data for each species
setosa_data = iris_data[iris_data['Species'] == 'Iris-setosa']
versicolor_data = iris_data[iris_data['Species'] == 'Iris-versicolor']
virginica_data = iris_data[iris_data['Species'] == 'Iris-virginica']

# Function to display basic statistical details
def display_statistics(data, species_name):
    print("Statistics for", species_name, ":\n")
    print("Count:")
    print(data.count())
    print("\nMean:")
    print(data.mean())
    print("\nStandard Deviation:")
    print(data.std())
    print("\nPercentiles:")
    print(data.quantile([0.25, 0.5, 0.75]))

# Display statistics for each species
display_statistics(setosa_data.drop(columns=['Id', 'Species']), 'Iris-setosa')
print("\n")
display_statistics(versicolor_data.drop(columns=['Id', 'Species']), 'Iris-versicolor')
print("\n")
display_statistics(virginica_data.drop(columns=['Id', 'Species']), 'Iris-virginica')
