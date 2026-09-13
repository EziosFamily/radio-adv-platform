import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 读取Excel文件
file_path = 'D:/173/test_data_defend.xlsx'
sheets = ['RNN', 'CNN']

# 创建图表 - 增加高度以适应新子图
plt.figure(figsize=(15, 10))  # 高度从12增加到16

for i, sheet in enumerate(sheets, 1):
    try:
        # 读取指定sheet的数据
        df = pd.read_excel(file_path, sheet_name=sheet, header=0)
        
        # 预处理数据
        # 1. 移除最后一行（测试用时）
        # df = df.iloc[:-1]
        
        # 2. 通过位置索引选择列（包括平均值列）
        df = df.iloc[:, :8]  # 现在选择前9列（包含平均值列）

        # 3. 重命名列
        new_columns = ['avg_change'] + [100, 300, 500, 1000, 2000, 3000, 5000]
        df.columns = new_columns
        print(df)
        # 创建子图
        plt.subplot(2, 1, i)  # 改为4行1列
        
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
            print(x_sorted,y_sorted)
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
        
        # 坐标轴设置
        plt.xlim(0, max(df['avg_change'])+0.0001)
        plt.ylim(0.5, 1.0)
        y_ticks = [round(y, 2) for y in np.linspace(0.5, 1.0, 11)]
        plt.yticks(y_ticks, fontsize=10)
        plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        plt.xlabel('Average Change Rate (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Defend Success Rate', fontsize=12, fontweight='bold')
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        plt.title(f'{sheet} Defend Performance', fontsize=12, fontweight='bold', pad=15)
        plt.legend(fontsize=10, frameon=True, shadow=True, 
                   facecolor='white', framealpha=0.9,
                   loc='lower right' if sheet == 'CNN' else 'best')
        
    except Exception as e:
        print(f"Error processing sheet {sheet}: {str(e)}")
        plt.subplot(4, 1, i)
        plt.text(0.5, 0.5, f"Error loading {sheet} data", 
                 ha='center', va='center', fontsize=12)
        plt.axis('off')

# 添加大标题
plt.suptitle('Adversarial Attack Detection Rate Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

# 调整布局并保存
plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])
# output_path = 'D:/173/attack_success_rates_comparison.png'
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f"图表已保存至: {os.path.abspath(output_path)}")
plt.show()