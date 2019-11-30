import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_one_cost = pd.read_csv("One_Control_Cost.csv")
data_two_cost = pd.read_csv("Two_Control_Cost.csv")
data_three_cost = pd.read_csv("Three_Control_Cost.csv")
plt.scatter(data_one_cost.time, data_one_cost.Int_Cost_value,s=80,
            marker='o',
            alpha=0.7)

plt.show()