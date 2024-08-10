import numpy as np

data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']

lowest_city = np.argmin(np.sum(totals, axis=1))
print(f"City with the lowest total precipitation: {lowest_city}")

average_monthly_precipitation = np.sum(totals, axis=0) / np.sum(counts, axis=0)
print("Average precipitation in each month:")
print(average_monthly_precipitation)

average_city_precipitation = np.sum(totals, axis=1) / np.sum(counts, axis=1)
print("Average precipitation in each city:")
print(average_city_precipitation)

quarters_totals = np.sum(totals.reshape(-1, 4, 3), axis=2)
print("Quarterly precipitation totals:")
print(quarters_totals)
