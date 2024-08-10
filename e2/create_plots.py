import sys
import pandas as pd
import matplotlib.pyplot as plt

filename1 = sys.argv[1]
filename2 = sys.argv[2]

data1 = pd.read_csv(filename1, sep=' ', header=None, index_col=1,
                    names=['lang', 'page', 'views', 'bytes'])

sorted_data1 = data1.sort_values(by='views', ascending=False)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(sorted_data1['views'].values)
plt.title('Popularity Distribution')
plt.xlabel('Rank')
plt.ylabel('Views')

data2 = pd.read_csv(filename2, sep=' ', header=None, index_col=1,
                    names=['lang', 'page', 'views', 'bytes'])

merged_data = pd.DataFrame({'Day 1': sorted_data1['views'], 'Day 2': data2['views']})

plt.subplot(1, 2, 2)
plt.scatter(merged_data['Day 1'], merged_data['Day 2'])
plt.xscale('log')
plt.yscale('log')
plt.title('Days Correlation')
plt.xlabel('Day 1 views')
plt.ylabel('Day 2 views')

plt.savefig('wikipedia.png')
