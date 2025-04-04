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

