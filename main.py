# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("./AXISBANK.csv")
data.head()

# %%
data.info()

# %%
data.isnull().sum()

# %%
data.drop(columns=["Trades",'Deliverable Volume',"%Deliverble"],inplace=True)

# %%
data.isnull().sum()

# %%


# %%
sns.boxplot(x=data['Prev Close'])

# %%
sns.boxplot(x=data['Close'])


# %%
q1 = data['Close'].quantile(0.25)
q2 = data['Close'].quantile(0.75)
iqr = q2 - q1
min_range = q1 - (1.5*iqr)
max_range = q2 + (1.5*iqr)
data = data[data['Close']<max_range]
data.head()

# %%
sns.boxplot(x=data['Close'])

# %%
data['Date'] = pd.to_datetime(data['Date'])


# %%
data.head()

# %%
# sns.heatmap(data.corr(),annot=True,cmap="Blues")
numeric_data = data.select_dtypes(include=['number'])
sns.heatmap(numeric_data.corr(), annot=True, cmap="Blues", linewidths=0.5)
# %%


# %%
sns.pairplot(data[['Close', 'Volume', 'Open', 'High']])
plt.show()

# %%

