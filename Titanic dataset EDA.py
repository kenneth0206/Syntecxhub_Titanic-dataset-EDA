import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Loading dataset

titanic = sns.load_dataset('titanic')

print("First five rows")
print(titanic.head())


print("\nMissing values")
print(titanic.isnull().sum())

print("\nData Types")
print(titanic.dtypes)

#Age groups

titanic['age_group'] = pd.cut(
    titanic['age'],
    bins=[0,14,18,33,40,100],
    labels=['child', 'Teen', 'Young Adult', 'Adult', 'Senior']
)


#Survival Rates
survival_by_sex = titanic.groupby('sex')['survived'].mean() * 100
survival_by_class = titanic.groupby('pclass')['survived'].mean() * 100
survival_by_age = titanic.groupby('age_group')['survived'].mean() * 100

print("\nSurvival by Gender")
print(survival_by_sex)

print("\nSurvival by Class")
print(survival_by_class)

print("\nSurvival by Age Group")
print(survival_by_age)
# Bar Plot 1
plt.figure(figsize=(6,4))
sns.barplot(x=survival_by_sex.index, y=survival_by_sex.values)
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Rate (%)")
plt.show()

# Bar plot 2
plt.figure(figsize=(6,4))
sns.barplot(x=survival_by_class.index, y=survival_by_class.values)
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.show()

# Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x='survived', y='age', data=titanic)
plt.title("Age Distribution by Survival")
plt.show()

# Violinplot
plt.figure(figsize=(8,5))
sns.violinplot(x='survived', y='age', data=titanic)
plt.title("Age Distribution by Survival")
plt.show()