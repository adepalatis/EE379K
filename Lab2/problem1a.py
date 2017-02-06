from pandas import DataFrame as df
from pandas.tools.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Using pandas
data = df.from_csv(path="C:\Users\Tony\Downloads\DF1")
scatter_matrix(data)
plt.show()

# Using Seaborn
sns.pairplot(data)
sns.plt.show()