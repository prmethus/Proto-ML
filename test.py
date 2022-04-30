import pandas as pd
from visualization import visualization
import matplotlib.pyplot as plt

data = pd.read_csv("/home/sarzil/Programming/Projects/AutoML/FuelConsumption.csv")

dt = data["CYLINDERS"].value_counts().sort_index()
print(dt.name)
visualization.configure_plot(style="darkgrid", palette="mako")
visualization.barplot(x=dt,y=dt.index)
plt.show()