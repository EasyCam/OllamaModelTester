import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
import re

# 设置中文字体支持和SVG输出配置
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['svg.fonttype'] = 'none'  # 保持SVG文本为可编辑格式
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

def extract_model_size_mb(model_name):
    """提取模型规模并转换为MB等效单位"""
    size_match = re.search(r':(\d+\.?\d*)([bm])', model_name.lower())
    if size_match:
        size_value = float(size_match.group(1))
        unit = size_match.group(2)
        if unit == 'b':  # billion parameters
            return size_value * 1000  # 转换为MB等效
        elif unit == 'm':  # million parameters
            return size_value
    return 1000  # 默认值

def load_and_preprocess_data(csv_file):
    """加载和预处理数据"""
    df = pd.read_csv(csv_file)
    
    # 只保留成功的测试结果
    df_success = df[df['status'] == 'success'].copy()
    
    # 提取模型信息
    df_success['model_family'] = df_success['model'].str.extract(r'([^:]+)')[0]
    df_success['model_size'] = df_success['model'].str.extract(r':([\d.]+b)')[0]
    df_success['model_size_mb'] = df_success['model'].apply(extract_model_size_mb)
    
    # 计算效率指标
    df_success['efficiency'] = df_success['eval_rate_tps'] / df_success['model_size_mb']
    
    return df_success

def create_self_score_stacked_chart(df_success):
    """创建自评分堆叠柱状图"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    scenarios = df_success['scenario'].unique()
    models = df_success['model'].unique()
    
    # 计算每个模型在每个场景下的平均自评分
    score_data = df_success.groupby(['model', 'scenario'])['self_score'].mean().unstack(fill_value=0)
    
    # 确保所有场景都存在
    for scenario in scenarios:
        if scenario not in score_data.columns:
            score_data[scenario] = 0
    
    score_data = score_data[scenarios]
    
    # 创建堆叠柱状图
    bar_width = 0.6
    x_pos = np.arange(len(models))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    scenario_colors = {scenario: colors[i % len(colors)] for i, scenario in enumerate(scenarios)}
    
    bottom = np.zeros(len(models))
    for i, scenario in enumerate(scenarios):
        values = [score_data.loc[model, scenario] if model in score_data.index else 0 for model in models]
        bars = ax.bar(x_pos, values, bar_width, bottom=bottom, 
                     label=scenario, color=scenario_colors[scenario], alpha=0.8)
        
        # 在每个子柱上添加数值标签
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., bottom[j] + height/2.,
                       f'{value:.1f}', ha='center', va='center', fontsize=9, fontweight='bold')
        
        bottom += values
    
    # 在每个柱子顶部显示总分
    for i, model in enumerate(models):
        total_score = sum([score_data.loc[model, scenario] if model in score_data.index else 0 for scenario in scenarios])
        ax.text(i, total_score + 0.2, f'总分: {total_score:.1f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('自评分总和', fontsize=12)
    ax.set_title('各模型在四个场景下的自评分堆叠图', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='测试场景', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig('ollama_self_score_stacked.svg', format='svg', bbox_inches='tight', dpi=300)
    print("自评分堆叠图已保存为: ollama_self_score_stacked.svg")
    plt.show()
    plt.close()

def create_performance_ranking_chart(df_success):
    """创建模型性能排名图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_avg = df_success.groupby('model')['eval_rate_tps'].agg(['mean', 'std']).sort_values('mean', ascending=True)
    
    y_pos = np.arange(len(model_avg))
    bars = ax.barh(y_pos, model_avg['mean'], xerr=model_avg['std'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(model_avg))))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_avg.index, fontsize=10)
    ax.set_xlabel('平均 eval_rate_tps (tokens/s)', fontsize=12)
    ax.set_title('模型整体性能排名（带标准差）', fontsize=14, fontweight='bold')
    
    # 去掉右侧和上方框线
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # 在柱状图上添加数值标签
    for i, (mean_val, std_val) in enumerate(zip(model_avg['mean'], model_avg['std'])):
        ax.text(mean_val + std_val + 100, i, f'{mean_val:.1f}±{std_val:.1f}', 
               va='center', fontsize=9)
    
    plt.tight_layout()
    fig.savefig('ollama_performance_ranking.svg', format='svg', bbox_inches='tight', dpi=300)
    print("模型性能排名图已保存为: ollama_performance_ranking.svg")
    plt.show()
    plt.close()

def create_heatmap_chart(df_success):
    """创建热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pivot_data = df_success.groupby(['model', 'scenario'])['eval_rate_tps'].mean().unstack(fill_value=0)
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': 'eval_rate_tps (tokens/s)'})
    
    ax.set_title('各模型在不同场景下的平均eval_rate_tps热力图', fontsize=14, fontweight='bold')
    ax.set_xlabel('测试场景', fontsize=12)
    ax.set_ylabel('模型', fontsize=12)
    ax.tick_params(axis='x', rotation=30, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    plt.tight_layout()
    fig.savefig('ollama_performance_heatmap.svg', format='svg', bbox_inches='tight', dpi=300)
    print("热力图已保存为: ollama_performance_heatmap.svg")
    plt.show()
    plt.close()

def create_efficiency_comparison_chart(df_success):
    """创建效率对比图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    scenarios = df_success['scenario'].unique()
    models = df_success['model'].unique()
    
    # 计算每个模型的平均效率
    efficiency_data = df_success.groupby(['model', 'scenario'])['efficiency'].mean().unstack(fill_value=0)
    
    # 确保所有场景都存在
    for scenario in scenarios:
        if scenario not in efficiency_data.columns:
            efficiency_data[scenario] = 0
    
    efficiency_data = efficiency_data[scenarios]
    
    # 创建效率堆叠柱状图
    bar_width = 0.6
    x_pos = np.arange(len(models))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    scenario_colors = {scenario: colors[i % len(colors)] for i, scenario in enumerate(scenarios)}
    
    bottom = np.zeros(len(models))
    for i, scenario in enumerate(scenarios):
        values = [efficiency_data.loc[model, scenario] if model in efficiency_data.index else 0 for model in models]
        bars = ax.bar(x_pos, values, bar_width, bottom=bottom, 
                     label=scenario, color=scenario_colors[scenario], alpha=0.8)
        
        # 在每个子柱上添加数值标签
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 0.1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., bottom[j] + height/2.,
                       f'{value:.1f}', ha='center', va='center', fontsize=8, fontweight='bold')
        
        bottom += values
    
    # 在每个柱子顶部显示总效率
    for i, model in enumerate(models):
        total_efficiency = sum([efficiency_data.loc[model, scenario] if model in efficiency_data.index else 0 for scenario in scenarios])
        ax.text(i, total_efficiency + 0.1, f'{total_efficiency:.1f}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel('效率 (tokens/s per MB)', fontsize=12)
    ax.set_title('模型效率对比图 (eval_rate_tps / 模型规模)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='测试场景', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig('ollama_efficiency_comparison.svg', format='svg', bbox_inches='tight', dpi=300)
    print("效率对比图已保存为: ollama_efficiency_comparison.svg")
    plt.show()
    plt.close()

def generate_detailed_report(df_success):
    """生成详细统计报告"""
    print("=" * 60)
    print("Ollama模型性能分析报告")
    print("=" * 60)
    
    # 1. 模型整体性能排名
    print("\n1. 模型整体性能排名（按平均eval_rate_tps）:")
    print("-" * 40)
    model_avg = df_success.groupby('model')['eval_rate_tps'].agg(['mean', 'std'])
    for model, stats in model_avg.sort_values('mean', ascending=False).iterrows():
        print(f"{model:20s}: {stats['mean']:8.1f} ± {stats['std']:6.1f} tokens/s")
    
    # 2. 模型自评分统计
    print("\n2. 模型自评分统计:")
    print("-" * 40)
    model_scores = df_success.groupby('model')['self_score'].agg(['mean', 'std', 'count'])
    for model, stats in model_scores.sort_values('mean', ascending=False).iterrows():
        print(f"{model:20s}: {stats['mean']:6.1f} ± {stats['std']:4.1f} (样本数: {stats['count']})")
    
    # 3. 场景性能统计
    print("\n3. 场景性能统计:")
    print("-" * 40)
    scenario_stats = df_success.groupby('scenario')['eval_rate_tps'].agg(['mean', 'std', 'count'])
    for scenario, stats in scenario_stats.sort_values('mean', ascending=False).iterrows():
        print(f"{scenario:15s}: {stats['mean']:8.1f} ± {stats['std']:6.1f} tokens/s (样本数: {stats['count']})")
    
    # 4. 模型效率统计
    print("\n4. 模型效率统计（tokens/s per MB）:")
    print("-" * 40)
    model_efficiency = df_success.groupby('model')['efficiency'].agg(['mean', 'std', 'count'])
    for model, stats in model_efficiency.sort_values('mean', ascending=False).iterrows():
        print(f"{model:20s}: {stats['mean']:6.2f} ± {stats['std']:4.2f} (样本数: {stats['count']})")
    
    # 5. 最佳性能组合
    print("\n5. 最佳性能组合:")
    print("-" * 40)
    best_combinations = df_success.groupby(['model', 'scenario'])['eval_rate_tps'].mean().sort_values(ascending=False).head(10)
    for (model, scenario), rate in best_combinations.items():
        print(f"{model:20s} + {scenario:15s}: {rate:8.1f} tokens/s")
    
    # 保存详细数据到CSV
    detailed_stats = df_success.groupby(['model', 'scenario']).agg({
        'eval_rate_tps': ['mean', 'std', 'min', 'max', 'count'],
        'self_score': ['mean', 'std'],
        'efficiency': ['mean', 'std'],
        'total_duration_raw': 'mean',
        'load_duration_sec': 'mean'
    }).round(2)
    
    detailed_stats.columns = ['_'.join(col).strip() for col in detailed_stats.columns]
    detailed_stats.to_csv('ollama_performance_analysis.csv')
    print(f"\n详细分析结果已保存到: ollama_performance_analysis.csv")

def main():
    """主函数"""
    try:
        # 加载和预处理数据
        df_success = load_and_preprocess_data('ollama_test_results_20250818_044636.csv')
        
        print(f"成功加载数据，共有 {len(df_success)} 条成功测试记录")
        print(f"涉及 {df_success['model'].nunique()} 个模型和 {df_success['scenario'].nunique()} 个场景")
        
        # 创建四个独立的图表
        print("\n正在生成图表...")
        create_self_score_stacked_chart(df_success)
        create_performance_ranking_chart(df_success)
        create_heatmap_chart(df_success)
        create_efficiency_comparison_chart(df_success)
        
        # 生成详细报告
        generate_detailed_report(df_success)
        
        print("\n分析完成！所有图表已生成并保存为SVG格式，文本保持为可编辑格式。")
        
    except FileNotFoundError:
        print("错误：找不到CSV文件 'ollama_test_results_20250816_112335.csv'")
        print("请确保文件存在于当前目录中。")
    except Exception as e:
        print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    main()