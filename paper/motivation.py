import matplotlib.pyplot as plt
plt.figure(figsize=(2, 2.2))

queries = ['$Q_{m_1}$', '$Q_{m_2}$', '$Q_{m_3}$']
x = [0, 0.1, 0.2]
y = [1.759, 1.429, 1.152]
colors = ['#F5F5F5', '#DAE8FC', '#FFE6CC']
plt.grid(zorder=0, axis='y',linestyle='--' )
plt.ylabel('Execution Time (sec)')
plt.bar(x, y, color=colors, width=0.06, zorder=10, edgecolor='black')
plt.xticks(x, queries)
plt.subplots_adjust(left=0.26, right=0.99, top=0.99, bottom=0.11)
plt.savefig('motivation.pdf', format='pdf')
