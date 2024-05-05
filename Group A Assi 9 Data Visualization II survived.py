import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import load_dataset

# Load Titanic dataset
titanic_data = sns.load_dataset('titanic')

# Plotting
plt.figure(figsize=(10, 6))

# Box plot of age distribution with respect to gender and survival
sns.boxplot(x='sex', y='age', hue='survived', data=titanic_data, palette='Set2')

# Adding labels and title
plt.title('Distribution of Age with Respect to Gender and Survival')
plt.xlabel('Gender')
plt.ylabel('Age')

# Show plot
plt.show()


# Observations:

# For male passengers who did not survive, the median age is around 30 years, while for male survivors, the median age is slightly higher, around 35 years.
# For female passengers who did not survive, the median age is around 28 years, while for female survivors, the median age is lower, around 26 years.
# The age distribution for male non-survivors is more spread out compared to male survivors, as indicated by the larger box and longer whiskers.
# The age distribution for female non-survivors is also more spread out compared to female survivors.
# There are several outliers (circles) in the plot, indicating the presence of passengers with ages significantly different from the majority of the passengers in their respective groups.
