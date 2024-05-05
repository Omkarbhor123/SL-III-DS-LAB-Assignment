import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=sns.load_dataset("titanic")
data

count=sns.countplot(x="sex",data=data)
for i in count.containers:
    count.bar_label(i)

bins=[10,20,30,40,50,60,70,80,90,100]
count=sns.histplot(x="age",bins=bins,data=data)
for i in count.containers:
    count.bar_label(i)
plt.plot()

count=sns.countplot(x="class",hue="sex",data=data)
for i in count.containers:
    count.bar_label(i)

sns.boxplot(x='sex',y="age",hue="survived",data=data,color='g',showmeans=True)
plt.show()

count=sns.countplot(x="sex",hue="alone",data=data)
for i in count.containers:
    count.bar_label(i)

count=sns.countplot(x="sex",hue="survived",data=data,color='y')
for i in count.containers:
    count.bar_label(i)

count=sns.violinplot(x="sex",y="age",hue="survived",data=data,color="b")

sns.displot(x="fare",kde=True,data=data)
plt.show()

sns.scatterplot(x="fare",y="sex",data=data)
