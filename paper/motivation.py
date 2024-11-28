import matplotlib.pyplot as plt
plt.figure(figsize=(3, 2))

queries = ['Q1', 'Q2', 'Q3']
x = [0, 0.1, 0.2]
y = [1.159, 0.829, 0.552]
colors = ['#F5F5F5', '#DAE8FC', '#FFE6CC']
plt.grid(zorder=0)
plt.ylabel('Execution Time (sec)')
plt.bar(x, y, color=colors, width=0.06, zorder=10, edgecolor='black')
plt.xticks(x, queries)
plt.subplots_adjust(left=0.21, right=0.99, top=0.99, bottom=0.12)
plt.savefig('motivation.pdf', format='pdf')