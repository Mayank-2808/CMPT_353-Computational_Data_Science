import pandas as pd

totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])

lowest_city = totals.sum(axis=1).idxmin()
print(f"City with lowest total precipitation: {lowest_city}")

average_monthly_precipitation = totals.sum(axis=0) / counts.sum(axis=0)
print("Average precipitation in each month:")
print(average_monthly_precipitation)

average_city_precipitation = totals.sum(axis=1) / counts.sum(axis=1)
print("Average precipitation in each city:")
print(average_city_precipitation)
