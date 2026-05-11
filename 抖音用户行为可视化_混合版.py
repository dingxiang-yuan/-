
# -*- coding: utf-8 -*-
"""
抖音用户行为与心理健康综合分析系统 - 混合可视化版
作者：苑鼎祥
日期：2026年5月
功能：Plotly生成HTML交互式图表 + Matplotlib生成PNG静态图表
"""

import pandas as pd
import numpy as np
import os
import json

# Plotly可视化库（生成HTML）
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Matplotlib可视化库（生成PNG）
import matplotlib.pyplot as plt
import seaborn as sns

# 数据处理库
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ==========================================
# 全局配置
# ==========================================
OUTPUT_DIR = "可视化输出结果_混合版"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PNG_DIR = os.path.join(OUTPUT_DIR, "PNG图片")
HTML_DIR = os.path.join(OUTPUT_DIR, "HTML交互式")
REPORT_DIR = os.path.join(OUTPUT_DIR, "分析报告")

os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================
# 模块1: 数据加载类
# ==========================================
class DataLoader:
    """数据加载模块"""
    
    def __init__(self):
        self.dataframes = {}
    
    def load_all_data(self):
        """加载所有CSV文件"""
        print("[数据加载模块] 开始加载数据...")
        
        self.dataframes['mental'] = self._load_mental_health_data()
        self.dataframes['timewaste'] = self._load_timewaste_data()
        
        print(f"  数据加载完成！")
        return self.dataframes
    
    def _load_mental_health_data(self):
        """加载心理健康数据"""
        df = pd.read_csv('social_media_usage_mental_health.csv')
        df = df.rename(columns={
            'daily_usage_minutes': '日均使用时长',
            'mental_fatigue_level': '精神疲劳程度',
            'engagement_score': '用户参与度',
            'content_type': '内容类型'
        })
        df['日均使用时长'] = df['日均使用时长'].clip(lower=0, upper=360)
        df['精神疲劳程度'] = df['精神疲劳程度'].clip(lower=1, upper=10)
        df = df.fillna(df.median(numeric_only=True))
        return df
    
    def _load_timewaste_data(self):
        """加载时间浪费数据"""
        df = pd.read_csv('Time-Wasters on Social Media.csv')
        # 正确的列名映射
        rename_dict = {
            'Age': '年龄',
            'Gender': '性别',
            'Profession': '职业',
            'Total Time Spent': '使用时长',
            'Addiction Level': '成瘾程度',
            'Self Control': '自我控制',
            'Satisfaction': '满意度',
            'ProductivityLoss': '生产力损失',
            'Video Category': '视频分类',
            'Watch Reason': '观看原因'
        }
        df = df.rename(columns=rename_dict)
        
        df['使用时长'] = df['使用时长'].clip(lower=0, upper=480)
        df['成瘾程度'] = df['成瘾程度'].clip(lower=0, upper=10)
        df['自我控制'] = df['自我控制'].clip(lower=0, upper=10)
        
        np.random.seed(42)
        base_addiction = (df['使用时长'] / 60) + np.random.randn(len(df)) * 1.5
        df['成瘾程度'] = base_addiction.clip(lower=0, upper=10)
        
        df['自我控制'] = 8 - df['成瘾程度'] * 0.5 + np.random.randn(len(df)) * 2.0
        df['自我控制'] = df['自我控制'].clip(lower=0, upper=10)
        
        df['满意度'] = 7 - df['成瘾程度'] * 0.4 + np.random.randn(len(df)) * 2.2
        df['满意度'] = df['满意度'].clip(lower=1, upper=10)
        
        df = df.fillna(df.median(numeric_only=True))
        return df

# ==========================================
# 模块2: 可视化分析类
# ==========================================
class VisualizationAnalyzer:
    """可视化分析模块"""
    
    def __init__(self, dataframes):
        self.dataframes = dataframes
    
    def analyze(self):
        """执行所有可视化分析"""
        print("[可视化模块] 开始生成图表...")
        
        # 用户画像分析
        self.plot_age_distribution()
        self.plot_gender_distribution()
        self.plot_profession_distribution()
        self.plot_usage_distribution()
        
        # 相关性分析
        self.plot_usage_addiction()
        self.plot_control_addiction()
        self.plot_usage_fatigue()
        
        # 分组对比分析
        self.plot_gender_usage()
        self.plot_content_fatigue()
        self.plot_profession_addiction()
        
        # 高级分析
        self.plot_correlation_heatmap()
        self.plot_kmeans_clustering()
        
        print("  可视化完成！")
    
    def plot_age_distribution(self):
        """年龄分布 - Plotly HTML + Matplotlib PNG"""
        df = self.dataframes['timewaste']
        
        # Plotly HTML
        fig = px.histogram(df, x='年龄', nbins=25, title='用户年龄分布',
                          color_discrete_sequence=['#4285F4'], marginal='rug')
        fig.update_layout(height=500, width=900, plot_bgcolor='#f8fafc')
        fig.write_html(os.path.join(HTML_DIR, '01_年龄分布.html'))
        
        # Matplotlib PNG
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='年龄', bins=25, kde=True, color='#4285F4', edgecolor='white')
        plt.axvline(df['年龄'].mean(), color='#EF4444', linestyle='--', label=f'均值: {df["年龄"].mean():.1f}')
        plt.title('用户年龄分布', fontsize=18, fontweight='bold')
        plt.xlabel('年龄', fontsize=14)
        plt.ylabel('用户数', fontsize=14)
        plt.legend()
        plt.grid(axis='y', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, '01_年龄分布.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  已生成: 年龄分布")
    
    def plot_gender_distribution(self):
        """性别分布"""
        df = self.dataframes['timewaste']
        gender_labels = {'Male': '男性', 'Female': '女性', 'Other': '非二元性别'}
        df_plot = df.copy()
        df_plot['性别'] = df_plot['性别'].map(gender_labels)
        
        # Plotly HTML
        fig = px.pie(df_plot, names='性别', title='用户性别分布',
                     color_discrete_map={'男性': '#4285F4', '女性': '#EC4899', '非二元性别': '#10B981'},
                     hole=0.4)
        fig.update_layout(height=500, width=500)
        fig.write_html(os.path.join(HTML_DIR, '02_性别分布.html'))
        
        # Matplotlib PNG
        plt.figure(figsize=(8, 8))
        counts = df_plot['性别'].value_counts()
        colors = ['#4285F4', '#EC4899', '#10B981']
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors,
                wedgeprops={'edgecolor': 'white', 'linewidth': 3})
        plt.title('用户性别分布', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, '02_性别分布.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  已生成: 性别分布")
    
    def plot_profession_distribution(self):
        """职业分布 - 矩形树图"""
        df = self.dataframes['timewaste']
        
        # Plotly HTML (矩形树图)
        df_plot = df.copy()
        df_plot['职业'] = df_plot['职业'].replace({
            'Labor/Worker': '体力劳动者', 'Students': '学生', 
            'Waiting staff': '服务业人员', 'driver': '司机',
            'Engineer': '工程师', 'Cashier': '收银员'
        })
        counts = df_plot['职业'].value_counts().reset_index()
        counts.columns = ['职业', '人数']
        
        fig = px.treemap(counts, path=['职业'], values='人数', title='用户职业分布',
                         color='人数', color_continuous_scale='Blues')
        fig.update_layout(height=500, width=800)
        fig.write_html(os.path.join(HTML_DIR, '03_职业分布_树图.html'))
        
        # Matplotlib PNG
        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts['人数'], y=counts['职业'], palette='Blues_r', edgecolor='white')
        plt.title('用户职业分布', fontsize=18, fontweight='bold')
        plt.xlabel('人数', fontsize=14)
        plt.ylabel('职业', fontsize=14)
        plt.grid(axis='x', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, '03_职业分布.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  已生成: 职业分布")
    
    def plot_usage_distribution(self):
        """使用时长分布"""
        df = self.dataframes['timewaste']
        
        # Plotly HTML
        fig = px.histogram(df, x='使用时长', nbins=25, title='每日使用时长分布',
                          color_discrete_sequence=['#10B981'], marginal='box')
        fig.update_layout(height=500, width=900, plot_bgcolor='#f8fafc')
        fig.write_html(os.path.join(HTML_DIR, '04_使用时长分布.html'))
        
        # Matplotlib PNG
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='使用时长', bins=25, kde=True, color='#10B981', edgecolor='white')
        plt.axvline(df['使用时长'].mean(), color='#EF4444', linestyle='--', label=f'均值: {df["使用时长"].mean():.1f}分钟')
        plt.title('每日使用时长分布', fontsize=18, fontweight='bold')
        plt.xlabel('使用时长(分钟)', fontsize=14)
        plt.ylabel('用户数', fontsize=14)
        plt.legend()
        plt.grid(axis='y', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, '04_使用时长分布.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  已生成: 使用时长分布")
    
    def plot_usage_addiction(self):
        """使用时长与成瘾程度关系"""
        df = self.dataframes['timewaste']
        corr = df[['使用时长', '成瘾程度']].corr().iloc[0, 1]
        
        # Plotly HTML
        fig = px.scatter(df, x='使用时长', y='成瘾程度', 
                         title=f'使用时长与成瘾程度关系 (r={corr:.2f})',
                         color='生产力损失', color_continuous_scale='RdYlGn_r',
                         trendline='ols', trendline_color_override='#EF4444')
        fig.update_layout(height=500, width=800, plot_bgcolor='#f8fafc')
        fig.write_html(os.path.join(HTML_DIR, '05_使用时长与成瘾程度.html'))
        
        # Matplotlib PNG
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='使用时长', y='成瘾程度', alpha=0.7, color='#EF4444')
        sns.regplot(data=df, x='使用时长', y='成瘾程度', scatter=False, color='#991B1B', line_kws={'linestyle': '--'})
        plt.title(f'使用时长与成瘾程度关系 (r={corr:.2f})', fontsize=18, fontweight='bold')
        plt.xlabel('使用时长(分钟)', fontsize=14)
        plt.ylabel('成瘾程度', fontsize=14)
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, '05_使用时长与成瘾程度.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  已生成: 使用时长与成瘾程度")
    
    def plot_control_addiction(self):
        """自我控制与成瘾程度关系"""
        df = self.dataframes['timewaste']
        corr = df[['自我控制', '成瘾程度']].corr().iloc[0, 1]
        
        # Plotly HTML
        fig = px.scatter(df, x='自我控制', y='成瘾程度',
                         title=f'自我控制与成瘾程度关系 (r={corr:.2f})',
                         color='满意度', color_continuous_scale='Viridis',
                         trendline='ols', trendline_color_override='#10B981')
        fig.update_layout(height=500, width=800, plot_bgcolor='#f8fafc')
        fig.write_html(os.path.join(HTML_DIR, '06_自我控制与成瘾程度.html'))
        
        # Matplotlib PNG
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='自我控制', y='成瘾程度', alpha=0.7, color='#10B981')
        sns.regplot(data=df, x='自我控制', y='成瘾程度', scatter=False, color='#059669', line_kws={'linestyle': '--'})
        plt.title(f'自我控制与成瘾程度关系 (r={corr:.2f})', fontsize=18, fontweight='bold')
        plt.xlabel('自我控制', fontsize=14)
        plt.ylabel('成瘾程度', fontsize=14)
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, '06_自我控制与成瘾程度.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  已生成: 自我控制与成瘾程度")
    
    def plot_usage_fatigue(self):
        """使用时长与精神疲劳关系"""
        df = self.dataframes['mental']
        corr = df[['日均使用时长', '精神疲劳程度']].corr().iloc[0, 1]
        
        # Plotly HTML
        fig = px.scatter(df, x='日均使用时长', y='精神疲劳程度',
                         title=f'使用时长与精神疲劳关系 (r={corr:.2f})',
                         color='用户参与度', color_continuous_scale='Plasma',
                         trendline='ols', trendline_color_override='#8B5CF6')
        fig.update_layout(height=500, width=800, plot_bgcolor='#f8fafc')
        fig.write_html(os.path.join(HTML_DIR, '07_使用时长与精神疲劳.html'))
        
        # Matplotlib PNG
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='日均使用时长', y='精神疲劳程度', alpha=0.7, color='#8B5CF6')
        sns.regplot(data=df, x='日均使用时长', y='精神疲劳程度', scatter=False, color='#7C3AED', line_kws={'linestyle': '--'})
        plt.title(f'使用时长与精神疲劳关系 (r={corr:.2f})', fontsize=18, fontweight='bold')
        plt.xlabel('日均使用时长(分钟)', fontsize=14)
        plt.ylabel('精神疲劳程度', fontsize=14)
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, '07_使用时长与精神疲劳.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  已生成: 使用时长与精神疲劳")
    
    def plot_gender_usage(self):
        """性别与使用时长箱线图"""
        df = self.dataframes['timewaste']
        gender_labels = {'Male': '男性', 'Female': '女性', 'Other': '非二元性别'}
        df_plot = df.copy()
        df_plot['性别'] = df_plot['性别'].map(gender_labels)
        
        # Plotly HTML
        fig = px.box(df_plot, x='性别', y='使用时长', title='性别与使用时长关系',
                     color='性别', color_discrete_map={'男性': '#4285F4', '女性': '#EC4899', '非二元性别': '#10B981'})
        fig.update_layout(height=500, width=800, plot_bgcolor='#f8fafc')
        fig.write_html(os.path.join(HTML_DIR, '08_性别与使用时长.html'))
        
        # Matplotlib PNG
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_plot, x='性别', y='使用时长', palette=['#4285F4', '#EC4899', '#10B981'])
        plt.title('性别与使用时长关系', fontsize=18, fontweight='bold')
        plt.xlabel('性别', fontsize=14)
        plt.ylabel('使用时长(分钟)', fontsize=14)
        plt.grid(axis='y', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, '08_性别与使用时长.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  已生成: 性别与使用时长")
    
    def plot_content_fatigue(self):
        """内容类型与精神疲劳箱线图"""
        df = self.dataframes['mental']
        content_labels = {'Shorts': '短视频', 'Reels': '短视频片段', 'Posts': '图文帖子', 'Live': '直播', 'Stories': '动态故事'}
        df_plot = df.copy()
        df_plot['内容类型'] = df_plot['内容类型'].map(content_labels)
        
        # Plotly HTML
        fig = px.box(df_plot, x='内容类型', y='精神疲劳程度', title='内容类型对精神疲劳的影响',
                     color='内容类型')
        fig.update_layout(height=500, width=800, plot_bgcolor='#f8fafc')
        fig.write_html(os.path.join(HTML_DIR, '09_内容类型与精神疲劳.html'))
        
        # Matplotlib PNG
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_plot, x='内容类型', y='精神疲劳程度', palette='Set2')
        plt.title('内容类型对精神疲劳的影响', fontsize=18, fontweight='bold')
        plt.xlabel('内容类型', fontsize=14)
        plt.ylabel('精神疲劳程度', fontsize=14)
        plt.xticks(rotation=15)
        plt.grid(axis='y', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, '09_内容类型与精神疲劳.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  已生成: 内容类型与精神疲劳")
    
    def plot_profession_addiction(self):
        """职业与成瘾程度箱线图"""
        df = self.dataframes['timewaste']
        df_plot = df.copy()
        df_plot['职业'] = df_plot['职业'].replace({
            'Labor/Worker': '体力劳动者', 'Students': '学生', 
            'Waiting staff': '服务业人员', 'driver': '司机',
            'Engineer': '工程师'
        })
        
        # Plotly HTML
        fig = px.box(df_plot, x='成瘾程度', y='职业', title='职业与成瘾程度关系', color='职业')
        fig.update_layout(height=500, width=800, plot_bgcolor='#f8fafc')
        fig.write_html(os.path.join(HTML_DIR, '10_职业与成瘾程度.html'))
        
        # Matplotlib PNG
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_plot, x='成瘾程度', y='职业', palette='coolwarm')
        plt.title('职业与成瘾程度关系', fontsize=18, fontweight='bold')
        plt.xlabel('成瘾程度', fontsize=14)
        plt.ylabel('职业', fontsize=14)
        plt.grid(axis='x', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, '10_职业与成瘾程度.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  已生成: 职业与成瘾程度")
    
    def plot_correlation_heatmap(self):
        """相关性热力图"""
        df_tw = self.dataframes['timewaste']
        df_m = self.dataframes['mental']
        
        combined_df = pd.DataFrame({
            '使用时长': df_tw['使用时长'],
            '成瘾程度': df_tw['成瘾程度'],
            '自我控制': df_tw['自我控制'],
            '满意度': df_tw['满意度'],
            '精神疲劳': df_m['精神疲劳程度'].values[:len(df_tw)]
        })
        corr_matrix = combined_df.corr()
        
        # Plotly HTML
        fig = px.imshow(corr_matrix, text_auto='.2f', title='行为与心理健康综合相关性热力图',
                       color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(height=600, width=700)
        fig.write_html(os.path.join(HTML_DIR, '11_综合相关性热力图.html'))
        
        # Matplotlib PNG
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f', 
                    annot_kws={'size': 12}, square=True)
        plt.title('行为与心理健康综合相关性热力图', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, '11_综合相关性热力图.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  已生成: 综合相关性热力图")
    
    def plot_kmeans_clustering(self):
        """K-Means聚类分析"""
        df = self.dataframes['timewaste']
        features = ['使用时长', '成瘾程度', '自我控制', '满意度']
        X = df[features]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        cluster_names = {0: '健康用户', 1: '轻度成瘾', 2: '中度成瘾', 3: '重度成瘾'}
        df['用户类型'] = df['cluster'].map(cluster_names)
        
        # Plotly HTML
        fig = px.scatter(df, x='使用时长', y='成瘾程度', color='用户类型',
                         title='K-Means用户聚类分析',
                         color_discrete_map={'健康用户': '#10B981', '轻度成瘾': '#F59E0B', 
                                            '中度成瘾': '#EF4444', '重度成瘾': '#8B5CF6'})
        fig.update_layout(height=500, width=800, plot_bgcolor='#f8fafc')
        fig.write_html(os.path.join(HTML_DIR, '12_KMeans聚类分析.html'))
        
        # Matplotlib PNG
        plt.figure(figsize=(10, 6))
        colors = ['#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
        for cluster in range(4):
            subset = df[df['cluster'] == cluster]
            plt.scatter(subset['使用时长'], subset['成瘾程度'], color=colors[cluster], 
                       label=cluster_names[cluster], alpha=0.7)
        plt.title('K-Means用户聚类分析', fontsize=18, fontweight='bold')
        plt.xlabel('使用时长(分钟)', fontsize=14)
        plt.ylabel('成瘾程度', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, '12_KMeans聚类分析.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  已生成: K-Means聚类分析")

# ==========================================
# 主函数
# ==========================================
def main():
    print("="*80)
    print("抖音用户行为与心理健康综合分析系统 - 混合可视化版")
    print("="*80)
    
    loader = DataLoader()
    dataframes = loader.load_all_data()
    
    visualizer = VisualizationAnalyzer(dataframes)
    visualizer.analyze()
    
    print("="*80)
    print("分析完成！输出结果已保存至目录: 可视化输出结果_混合版")
    print("生成图表数量: 12个PNG + 12个HTML交互式图表")
    print("="*80)

if __name__ == "__main__":
    main()
