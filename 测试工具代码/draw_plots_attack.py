import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import make_interp_spline

# 读取Excel文件
file_path = 'D:/173/test_data_RNN.xlsx'
sheets = ['PGD', 'tsmiFGSM', 'mpdsm']

# 创建图表 - 增加高度以适应新子图
plt.figure(figsize=(15, 16))  # 高度从12增加到16

# 读取CNN测试数据
cnn_data_path = 'D:/173/攻击测试结果.xlsx'
cnn_df = pd.read_excel(cnn_data_path, sheet_name='RNN', header=0)
cnn_change_rates = cnn_df.iloc[:, 0].values*100
cnn_radio1 = cnn_df.iloc[:, 1].values
cnn_radio2 = cnn_df.iloc[:, 2].values
cnn_radio3 = cnn_df.iloc[:, 3].values
cnn_random = cnn_df.iloc[:, 4].values

# 存储平均值数据用于新图表
mean_comparison_data = []

for i, sheet in enumerate(sheets, 1):
    try:
        # 读取指定sheet的数据
        df = pd.read_excel(file_path, sheet_name=sheet, header=0)
        
        # 预处理数据
        # 1. 移除最后一行（测试用时）
        df = df.iloc[:-1]
        
        # 2. 通过位置索引选择列（包括平均值列）
        df = df.iloc[:, :9]  # 现在选择前9列（包含平均值列）

        # 3. 重命名列
        new_columns = ['avg_change'] + [100, 300, 500, 1000, 2000, 3000, 5000, 'mean']
        df.columns = new_columns
        
        # 保存平均值数据
        mean_comparison_data.append({
            'sheet': sheet,
            'x': df['avg_change'].values,
            'y': df['mean'].values
        })
        
        # 创建子图
        plt.subplot(4, 1, i)  # 改为4行1列
        
        # 为每个数据量绘制平滑曲线
        markers = ['o', 's', '^', 'D', 'v'] 
        line_styles = ['-', '--', '-.', ':', '-', '-', '-']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', "#f808e4", "#EBC91E"]
        
        for j, size in enumerate([100, 300, 500, 1000, 2000, 3000, 5000]):
            x = df['avg_change'].values
            y = df[size].values
            
            sorted_idx = np.argsort(x)
            x_sorted = x[sorted_idx]
            y_sorted = y[sorted_idx]
            
            try:
                x_new = np.linspace(x_sorted.min(), x_sorted.max(), 300)
                if len(x_sorted) > 3:
                    spl = make_interp_spline(x_sorted, y_sorted, k=3)
                    y_smooth = spl(x_new)
                else:
                    y_smooth = np.interp(x_new, x_sorted, y_sorted)
                
                plt.plot(x_new, y_smooth, 
                         color=colors[j],
                         linestyle=line_styles[j],
                         linewidth=1.5,
                         alpha=0.9,
                         label=f'Data Size: {size}')
                
            except Exception as e:
                print(f"插值失败 ({sheet}, size={size}): {str(e)}")
                plt.plot(x, y, 
                         marker=markers[j],
                         linestyle=line_styles[j],
                         linewidth=2,
                         markersize=8,
                         label=f'Data Size: {size}')
        
        # 添加CNN测试数据散点
        plt.scatter(cnn_change_rates, cnn_radio1, marker='.', s=80, color='red', 
                   edgecolor='black', zorder=10)
        plt.scatter(cnn_change_rates, cnn_radio2, marker='.', s=80, color='blue', 
                   edgecolor='black', zorder=10)
        plt.scatter(cnn_change_rates, cnn_radio3, marker='.', s=80, color='green', 
                   edgecolor='black', zorder=10)
        plt.scatter(cnn_change_rates, cnn_random, marker='.', s=80, color='purple', 
                   edgecolor='black', zorder=10)
        
        # 坐标轴设置
        plt.xlim(min(df['avg_change'])-0.2, max(df['avg_change'])+0.2)
        plt.ylim(0.5, 1.0)
        y_ticks = [round(y, 2) for y in np.linspace(0.5, 1.0, 11)]
        plt.yticks(y_ticks, fontsize=10)
        plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        plt.xlabel('Average Change Rate (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Attack Success Rate', fontsize=12, fontweight='bold')
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        plt.title(f'{sheet} Attack Performance', fontsize=12, fontweight='bold', pad=15)
        plt.legend(fontsize=10, frameon=True, shadow=True, 
                   facecolor='white', framealpha=0.9,
                   loc='lower right' if sheet == 'mpdsm' else 'best')
        
    except Exception as e:
        print(f"Error processing sheet {sheet}: {str(e)}")
        plt.subplot(4, 1, i)
        plt.text(0.5, 0.5, f"Error loading {sheet} data", 
                 ha='center', va='center', fontsize=12)
        plt.axis('off')

# ============= 添加第四个图：三种算法在平均值列上的对比 =============
plt.subplot(4, 1, 4)

# 为三种算法定义不同的颜色和线型
algorithm_colors = {
    'PGD': '#1f77b4',
    'tsmiFGSM': '#ff7f0e',
    'mpdsm': '#2ca02c'
}
algorithm_line_styles = {
    'PGD': '-',
    'tsmiFGSM': '--',
    'mpdsm': '-.'
}

# 绘制每种算法的平均值曲线
for data in mean_comparison_data:
    sheet = data['sheet']
    x = data['x']
    y = data['y']
    
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    
    try:
        x_new = np.linspace(x_sorted.min(), x_sorted.max(), 300)
        if len(x_sorted) > 3:
            spl = make_interp_spline(x_sorted, y_sorted, k=3)
            y_smooth = spl(x_new)
        else:
            y_smooth = np.interp(x_new, x_sorted, y_sorted)
        
        plt.plot(x_new, y_smooth, 
                 color=algorithm_colors[sheet],
                 linestyle=algorithm_line_styles[sheet],
                 linewidth=1.5,
                 alpha=0.9,
                 label=f'{sheet} (Mean)')
        
    except Exception as e:
        print(f"插值失败 ({sheet}, mean): {str(e)}")
        plt.plot(x, y, 
                 color=algorithm_colors[sheet],
                 linestyle=algorithm_line_styles[sheet],
                 linewidth=1.5,
                 marker='o',
                 markersize=7,
                 label=f'{sheet} (Mean)')

# 添加CNN测试数据散点（与前面相同）
plt.scatter(cnn_change_rates, cnn_radio1, marker='.', s=80, color='red', 
           edgecolor='black', zorder=10, label='Radio1')
plt.scatter(cnn_change_rates, cnn_radio2, marker='.', s=80, color='blue', 
           edgecolor='black', zorder=10, label='Radio2')
plt.scatter(cnn_change_rates, cnn_radio3, marker='.', s=80, color='green', 
           edgecolor='black', zorder=10, label='Radio3')
plt.scatter(cnn_change_rates, cnn_random, marker='.', s=80, color='purple', 
           edgecolor='black', zorder=10, label='Random')

# 设置坐标轴范围
all_x = [point for data in mean_comparison_data for point in data['x']]
min_x = min(all_x) - 0.2
max_x = max(all_x) + 0.2
plt.xlim(min_x, max_x)
plt.ylim(0.5, 1.0)

# 设置网格和刻度
y_ticks = [round(y, 2) for y in np.linspace(0.5, 1.0, 11)]
plt.yticks(y_ticks, fontsize=10)
plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

# 设置坐标轴标签
plt.xlabel('Average Change Rate (%)', fontsize=12, fontweight='bold')
plt.ylabel('Attack Success Rate (Mean)', fontsize=12, fontweight='bold')

# 设置边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 设置标题和图例
plt.title('Comparison of Attack Methods (Mean Values)', fontsize=12, fontweight='bold', pad=15)
plt.legend(fontsize=10, frameon=True, shadow=True, 
           facecolor='white', framealpha=0.9,
           loc='best')

# ============= 第四个图结束 =============

# 添加大标题
plt.suptitle('Adversarial Attack Success Rate Analysis(RNN)', 
             fontsize=16, fontweight='bold', y=0.98)

# 调整布局并保存
plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])
# output_path = 'D:/173/attack_success_rates_comparison.png'
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f"图表已保存至: {os.path.abspath(output_path)}")
plt.show()