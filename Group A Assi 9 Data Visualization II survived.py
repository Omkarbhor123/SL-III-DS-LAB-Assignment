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

# In both genders, the median age of those who survived appears to be slightly lower than those who did not survive.
# The age distribution for males who survived is slightly narrower compared to males who did not survive.
# Females who survived seem to have a wider range of ages compared to females who did not survive.
# There are outliers present in the age distribution for both genders and survival categories, suggesting the presence of individuals whose ages significantly deviate from the majority within their respective groups.
