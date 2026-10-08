import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("Iris.csv")
iris = iris.drop(columns="Id")

print(iris.shape)
print(iris.head(3))
print(iris.dtypes)
print(iris.isnull().sum())
print("duplicates:", iris.duplicated().sum())
iris = iris.drop_duplicates()

print(iris.describe())
print(iris["Species"].value_counts())
print(iris.groupby("Species").mean())
print(iris.corr(numeric_only=True))

fig, ax = plt.subplots(1, 3, figsize=(14, 4))

ax[0].hist(iris["PetalLengthCm"], bins=12, color="green")
ax[0].set_title("Petal length")
ax[0].set_xlabel("cm")
ax[0].set_ylabel("count")

for name, group in iris.groupby("Species"):
    ax[1].scatter(group["PetalLengthCm"], group["PetalWidthCm"], label=name)
ax[1].set_title("Petal length vs width")
ax[1].set_xlabel("length (cm)")
ax[1].set_ylabel("width (cm)")
ax[1].legend()

iris.boxplot(column="SepalLengthCm", by="Species", ax=ax[2])
ax[2].set_title("Sepal length by species")
ax[2].set_ylabel("cm")

plt.tight_layout()
plt.show()
