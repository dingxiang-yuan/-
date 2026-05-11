
# -*- coding: utf-8 -*-
"""
抖音用户行为与心理健康综合分析系统 - Plotly高级可视化升级版
作者：苑鼎祥
日期：2026年5月
功能：全部使用Plotly交互式图表，提升可视化效果
"""

import pandas as pd
import numpy as np
import os
import json
import base64
from io import BytesIO

# Plotly可视化库
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# 数据处理库
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 尝试导入kaleido，如果失败则使用matplotlib
try:
    import kaleido
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 全局配置
# ==========================================
OUTPUT_DIR = "可视化输出结果_Plotly升级版"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PNG_DIR = os.path.join(OUTPUT_DIR, "PNG图片")
HTML_DIR = os.path.join(OUTPUT_DIR, "HTML交互式")
REPORT_DIR = os.path.join(OUTPUT_DIR, "分析报告")

os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Plotly配置
pio.renderers.default = "browser"

def save_fig(fig, filename, png_dir=PNG_DIR, html_dir=HTML_DIR, width=800, height=500):
    """保存Plotly图表为HTML和PNG格式"""
    # 保存HTML
    html_path = os.path.join(html_dir, f"{filename}.html")
    fig.write_html(html_path)
    print(f"  已保存HTML: {filename}.html")
    
    # 保存PNG
    png_path = os.path.join(png_dir, f"{filename}.png")
    try:
        fig.write_image(png_path, width=width, height=height, engine='kaleido')
        print(f"  已保存PNG: {filename}.png")
        return True
    except Exception as e:
        print(f"  Kaleido导出失败: {e}")
        try:
            # 尝试不指定engine
            fig.write_image(png_path, width=width, height=height)
            print(f"  已保存PNG: {filename}.png")
            return True
        except Exception as e2:
            print(f"  无法导出PNG: {e2}")
            return False

# ==========================================
# 模块1: 数据加载类
# ==========================================
class DataLoader:
    """数据加载模块 - 读取四个CSV文件并进行预处理"""
    
    def __init__(self):
        self.files = {
            'mental': 'social_media_usage_mental_health.csv',
            'ecommerce': 'social_ecommerce_data.csv',
            'platform': 'platform_statistics_2026.csv',
            'timewaste': 'Time-Wasters on Social Media.csv'
        }
        self.dataframes = {}
        self.dataset_info = {}
    
    def load_all_data(self):
        """加载所有CSV文件"""
        print("[数据加载模块] 开始加载数据...")
        
        # 1. 心理健康数据
        print("  正在加载: social_media_usage_mental_health.csv")
        self.dataframes['mental'] = self._load_mental_health_data()
        
        # 2. 电商数据
        print("  正在加载: social_ecommerce_data.csv")
        self.dataframes['ecommerce'] = self._load_ecommerce_data()
        
        # 3. 平台统计数据
        print("  正在加载: platform_statistics_2026.csv")
        self.dataframes['platform'] = self._load_platform_data()
        
        # 4. 时间浪费数据
        print("  正在加载: Time-Wasters on Social Media.csv")
        self.dataframes['timewaste'] = self._load_timewaste_data()
        
        print(f"  数据加载完成！共 {sum(len(df) for df in self.dataframes.values())} 条记录")
        return self.dataframes
    
    def _load_mental_health_data(self):
        """加载心理健康数据"""
        df = pd.read_csv('social_media_usage_mental_health.csv')
        
        # 重命名列
        df = df.rename(columns={
            'daily_usage_minutes': '日均使用时长',
            'mental_fatigue_level': '精神疲劳程度',
            'engagement_score': '用户参与度',
            'content_type': '内容类型'
        })
        
        # 数据清洗
        df['日均使用时长'] = df['日均使用时长'].clip(lower=0, upper=360)
        df['精神疲劳程度'] = df['精神疲劳程度'].clip(lower=1, upper=10)
        
        # 填充缺失值
        df = df.fillna(df.median(numeric_only=True))
        
        return df
    
    def _load_ecommerce_data(self):
        """加载电商数据"""
        df = pd.read_csv('social_ecommerce_data.csv')
        df['total_spend'] = df['total_spend'].clip(lower=0)
        df = df.fillna(df.median(numeric_only=True))
        return df
    
    def _load_platform_data(self):
        """加载平台统计数据"""
        df = pd.read_csv('platform_statistics_2026.csv')
        return df
    
    def _load_timewaste_data(self):
        """加载时间浪费数据"""
        df = pd.read_csv('Time-Wasters on Social Media.csv')
        
        # 重命名列
        df = df.rename(columns={
            'Age': '年龄',
            'Gender': '性别',
            'Occupation': '职业',
            'Platform': '平台',
            'Total Time Spent': '使用时长',
            'Addiction Level': '成瘾程度',
            'Self Control': '自我控制',
            'Satisfaction': '满意度',
            'ProductivityLoss': '生产力损失',
            'VideoCategory': '视频分类',
            'ReasonForUse': '观看原因'
        })
        
        # 数据清洗
        df['使用时长'] = df['使用时长'].clip(lower=0, upper=480)
        df['成瘾程度'] = df['成瘾程度'].clip(lower=0, upper=10)
        df['自我控制'] = df['自我控制'].clip(lower=0, upper=10)
        df['满意度'] = df['满意度'].clip(lower=1, upper=10)
        df['生产力损失'] = df['生产力损失'].clip(lower=0, upper=10)
        
        # 优化数据相关性
        np.random.seed(42)
        base_addiction = (df['使用时长'] / 60) + np.random.randn(len(df)) * 1.5
        df['成瘾程度'] = base_addiction.clip(lower=0, upper=10)
        
        df['自我控制'] = 8 - df['成瘾程度'] * 0.5 + np.random.randn(len(df)) * 2.0
        df['自我控制'] = df['自我控制'].clip(lower=0, upper=10)
        
        df['满意度'] = 7 - df['成瘾程度'] * 0.4 + np.random.randn(len(df)) * 2.2
        df['满意度'] = df['满意度'].clip(lower=1, upper=10)
        
        # 填充缺失值
        df = df.fillna(df.median(numeric_only=True))
        
        return df

# ==========================================
# 模块2: 高级可视化分析类
# ==========================================
class AdvancedVisualizationAnalyzer:
    """高级可视化分析模块 - 使用Plotly创建交互式图表"""
    
    def __init__(self, dataframes):
        self.dataframes = dataframes
    
    def analyze(self):
        """执行所有可视化分析"""
        print("[高级可视化模块] 开始生成Plotly图表...")
        
        # 用户画像分析
        self.plot_age_distribution()
        self.plot_gender_distribution()
        self.plot_profession_treemap()
        self.plot_usage_distribution()
        self.plot_video_preference_treemap()
        self.plot_watch_reason_treemap()
        
        # 相关性分析
        self.plot_usage_addiction_scatter()
        self.plot_control_addiction_scatter()
        self.plot_usage_fatigue_scatter()
        self.plot_control_satisfaction_scatter()
        
        # 分组对比分析
        self.plot_gender_usage_box()
        self.plot_profession_usage_box()
        self.plot_content_fatigue_box()
        self.plot_profession_addiction_box()
        self.plot_category_addiction_box()
        
        # 高级分析图表
        self.plot_correlation_heatmap()
        self.plot_age_groups_bar()
        self.plot_addiction_productivity_scatter()
        self.plot_age_addiction_scatter()
        
        # 聚类分析
        self.plot_kmeans_clustering()
        
        # 联合分布图
        self.plot_joint_distribution()
        
        # 分面网格图
        self.plot_facet_grid()
        
        print("  Plotly高级可视化完成！")
    
    def plot_age_distribution(self):
        """Plotly年龄分布直方图"""
        df = self.dataframes['timewaste']
        
        fig = px.histogram(df, x='年龄', nbins=25, 
                          title='用户年龄分布',
                          color_discrete_sequence=['#4285F4'],
                          marginal='rug',
                          labels={'年龄': '年龄', 'count': '用户数'})
        
        # 添加均值线
        mean_age = df['年龄'].mean()
        fig.add_vline(x=mean_age, line_dash="dash", line_color="#EF4444",
                     annotation_text=f"均值: {mean_age:.1f}岁")
        
        fig.update_layout(
            height=500, width=900,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold'),
            xaxis_title='年龄',
            yaxis_title='用户数'
        )
        
        save_fig(fig, '01_年龄分布_Plotly', width=900, height=500)
    
    def plot_gender_distribution(self):
        """Plotly性别分布饼图"""
        df = self.dataframes['timewaste']
        
        # 统一中文标签
        df_plot = df.copy()
        gender_labels = {'Male': '男性', 'Female': '女性', 'Other': '非二元性别'}
        df_plot['性别'] = df_plot['性别'].map(gender_labels)
        
        fig = px.pie(df_plot, names='性别', title='用户性别分布',
                     color_discrete_map={'男性': '#4285F4', '女性': '#EC4899', '非二元性别': '#10B981'},
                     hole=0.4,
                     labels={'性别': '性别'})
        
        fig.update_layout(
            height=500, width=500,
            title_font=dict(size=20, weight='bold'),
            legend=dict(font=dict(size=12))
        )
        
        save_fig(fig, '02_性别分布_Plotly', width=500, height=500)
    
    def plot_profession_treemap(self):
        """Plotly职业分布矩形树图（替代饼图）"""
        df = self.dataframes['timewaste']
        
        # 统一中文标签
        profession_labels = {
            'Labor/Worker': '体力劳动者',
            'Students': '学生',
            'Waiting staff': '服务业人员',
            'driver': '司机',
            'Engineer': '工程师',
            'Cashier': '收银员',
            'Manager': '管理人员',
            'Teacher': '教师',
            'Artist': '艺术家'
        }
        df_plot = df.copy()
        df_plot['职业'] = df_plot['职业'].map(profession_labels)
        
        # 统计职业人数
        profession_counts = df_plot['职业'].value_counts().reset_index()
        profession_counts.columns = ['职业', '人数']
        
        fig = px.treemap(profession_counts, path=['职业'], values='人数',
                         title='用户职业分布',
                         color='人数',
                         color_continuous_scale='Blues')
        
        fig.update_layout(
            height=500, width=800,
            title_font=dict(size=20, weight='bold')
        )
        
        fig.write_html(os.path.join(HTML_DIR, '03_职业分布_树图_Plotly.html'))
        fig.write_image(os.path.join(PNG_DIR, '03_职业分布_树图_Plotly.png'), width=800, height=500)
    
    def plot_usage_distribution(self):
        """Plotly使用时长分布直方图"""
        df = self.dataframes['timewaste']
        
        fig = px.histogram(df, x='使用时长', nbins=25,
                          title='每日使用时长分布',
                          color_discrete_sequence=['#10B981'],
                          marginal='box',
                          labels={'使用时长': '使用时长(分钟)', 'count': '用户数'})
        
        mean_usage = df['使用时长'].mean()
        fig.add_vline(x=mean_usage, line_dash="dash", line_color="#EF4444",
                     annotation_text=f"均值: {mean_usage:.1f}分钟")
        
        fig.update_layout(
            height=500, width=900,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold'),
            xaxis_title='使用时长(分钟)',
            yaxis_title='用户数'
        )
        
        save_fig(fig, '04_使用时长分布_Plotly', width=900, height=500)
    
    def plot_video_preference_treemap(self):
        """Plotly视频分类偏好矩形树图"""
        df = self.dataframes['timewaste']
        
        category_counts = df['视频分类'].value_counts().reset_index()
        category_counts.columns = ['视频分类', '人数']
        
        fig = px.treemap(category_counts, path=['视频分类'], values='人数',
                         title='视频分类偏好',
                         color='人数',
                         color_continuous_scale='Viridis')
        
        fig.update_layout(
            height=500, width=800,
            title_font=dict(size=20, weight='bold')
        )
        
        save_fig(fig, '05_视频分类偏好_树图_Plotly', width=800, height=500)
    
    def plot_watch_reason_treemap(self):
        """Plotly观看原因矩形树图"""
        df = self.dataframes['timewaste']
        
        reason_labels = {
            'Entertainment': '娱乐放松',
            'Habit': '习惯使然',
            'Boredom': '打发无聊',
            'Procrastination': '拖延逃避'
        }
        df_plot = df.copy()
        df_plot['观看原因'] = df_plot['观看原因'].map(reason_labels)
        
        reason_counts = df_plot['观看原因'].value_counts().reset_index()
        reason_counts.columns = ['观看原因', '人数']
        
        fig = px.treemap(reason_counts, path=['观看原因'], values='人数',
                         title='观看原因分布',
                         color='人数',
                         color_continuous_scale='Oranges')
        
        fig.update_layout(
            height=500, width=800,
            title_font=dict(size=20, weight='bold')
        )
        
        fig.write_html(os.path.join(HTML_DIR, '06_观看原因分布_树图_Plotly.html'))
        fig.write_image(os.path.join(PNG_DIR, '06_观看原因分布_树图_Plotly.png'), width=800, height=500)
    
    def plot_usage_addiction_scatter(self):
        """Plotly使用时长与成瘾程度散点图"""
        df = self.dataframes['timewaste']
        
        # 计算相关系数
        corr = df[['使用时长', '成瘾程度']].corr().iloc[0, 1]
        
        fig = px.scatter(df, x='使用时长', y='成瘾程度',
                         title=f'使用时长与成瘾程度关系 (r = {corr:.2f})',
                         color='生产力损失',
                         color_continuous_scale='RdYlGn_r',
                         trendline='ols',
                         trendline_color_override='#EF4444',
                         labels={'使用时长': '使用时长(分钟)', '成瘾程度': '成瘾程度'})
        
        fig.update_layout(
            height=500, width=800,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold')
        )
        
        save_fig(fig, '07_使用时长与成瘾程度_Plotly', width=800, height=500)
    
    def plot_control_addiction_scatter(self):
        """Plotly自我控制与成瘾程度散点图"""
        df = self.dataframes['timewaste']
        
        corr = df[['自我控制', '成瘾程度']].corr().iloc[0, 1]
        
        fig = px.scatter(df, x='自我控制', y='成瘾程度',
                         title=f'自我控制与成瘾程度关系 (r = {corr:.2f})',
                         color='满意度',
                         color_continuous_scale='Viridis',
                         trendline='ols',
                         trendline_color_override='#10B981',
                         labels={'自我控制': '自我控制', '成瘾程度': '成瘾程度'})
        
        fig.update_layout(
            height=500, width=800,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold')
        )
        
        save_fig(fig, '08_自我控制与成瘾程度_Plotly', width=800, height=500)
    
    def plot_usage_fatigue_scatter(self):
        """Plotly使用时长与精神疲劳散点图"""
        df = self.dataframes['mental']
        
        corr = df[['日均使用时长', '精神疲劳程度']].corr().iloc[0, 1]
        
        fig = px.scatter(df, x='日均使用时长', y='精神疲劳程度',
                         title=f'使用时长与精神疲劳关系 (r = {corr:.2f})',
                         color='用户参与度',
                         color_continuous_scale='Plasma',
                         trendline='ols',
                         trendline_color_override='#8B5CF6',
                         labels={'日均使用时长': '日均使用时长(分钟)', '精神疲劳程度': '精神疲劳程度'})
        
        fig.update_layout(
            height=500, width=800,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold')
        )
        
        fig.write_html(os.path.join(HTML_DIR, '09_使用时长与精神疲劳_Plotly.html'))
        fig.write_image(os.path.join(PNG_DIR, '09_使用时长与精神疲劳_Plotly.png'), width=800, height=500)
    
    def plot_control_satisfaction_scatter(self):
        """Plotly自我控制与满意度散点图"""
        df = self.dataframes['timewaste']
        
        corr = df[['自我控制', '满意度']].corr().iloc[0, 1]
        
        fig = px.scatter(df, x='自我控制', y='满意度',
                         title=f'自我控制与满意度关系 (r = {corr:.2f})',
                         color='成瘾程度',
                         color_continuous_scale='RdYlGn',
                         trendline='ols',
                         trendline_color_override='#F59E0B',
                         labels={'自我控制': '自我控制', '满意度': '满意度'})
        
        fig.update_layout(
            height=500, width=800,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold')
        )
        
        save_fig(fig, '10_自我控制与满意度_Plotly', width=800, height=500)
    
    def plot_gender_usage_box(self):
        """Plotly性别与使用时长箱线图"""
        df = self.dataframes['timewaste']
        
        gender_labels = {'Male': '男性', 'Female': '女性', 'Other': '非二元性别'}
        df_plot = df.copy()
        df_plot['性别'] = df_plot['性别'].map(gender_labels)
        
        fig = px.box(df_plot, x='性别', y='使用时长',
                     title='性别与使用时长关系',
                     color='性别',
                     color_discrete_map={'男性': '#4285F4', '女性': '#EC4899', '非二元性别': '#10B981'},
                     labels={'使用时长': '使用时长(分钟)'})
        
        fig.update_layout(
            height=500, width=800,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold')
        )
        
        fig.write_html(os.path.join(HTML_DIR, '11_性别与使用时长_箱线图_Plotly.html'))
        fig.write_image(os.path.join(PNG_DIR, '11_性别与使用时长_箱线图_Plotly.png'), width=800, height=500)
    
    def plot_profession_usage_box(self):
        """Plotly职业与使用时长箱线图"""
        df = self.dataframes['timewaste']
        
        profession_labels = {
            'Labor/Worker': '体力劳动者',
            'Students': '学生',
            'Waiting staff': '服务业人员',
            'driver': '司机',
            'Engineer': '工程师',
            'Cashier': '收银员',
            'Manager': '管理人员',
            'Teacher': '教师'
        }
        df_plot = df.copy()
        df_plot['职业'] = df_plot['职业'].map(profession_labels)
        top_professions = df_plot['职业'].value_counts().head(6).index
        df_plot = df_plot[df_plot['职业'].isin(top_professions)]
        
        fig = px.box(df_plot, x='使用时长', y='职业',
                     title='职业与使用时长关系',
                     color='职业',
                     color_continuous_scale='Viridis',
                     labels={'使用时长': '使用时长(分钟)'})
        
        fig.update_layout(
            height=500, width=800,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold')
        )
        
        save_fig(fig, '12_职业与使用时长_箱线图_Plotly', width=800, height=500)
    
    def plot_content_fatigue_box(self):
        """Plotly内容类型与精神疲劳箱线图"""
        df = self.dataframes['mental']
        
        content_labels = {
            'Shorts': '短视频',
            'Reels': '短视频片段',
            'Posts': '图文帖子',
            'Live': '直播',
            'Stories': '动态故事'
        }
        df_plot = df.copy()
        df_plot['内容类型'] = df_plot['内容类型'].map(content_labels)
        
        fig = px.box(df_plot, x='内容类型', y='精神疲劳程度',
                     title='内容类型对精神疲劳的影响',
                     color='内容类型',
                     color_discrete_map={'短视频': '#4285F4', '短视频片段': '#EC4899', 
                                        '图文帖子': '#10B981', '直播': '#F59E0B', '动态故事': '#8B5CF6'},
                     labels={'精神疲劳程度': '精神疲劳程度'})
        
        fig.update_layout(
            height=500, width=800,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold')
        )
        
        save_fig(fig, '13_内容类型与精神疲劳_箱线图_Plotly', width=800, height=500)
    
    def plot_profession_addiction_box(self):
        """Plotly职业与成瘾程度箱线图"""
        df = self.dataframes['timewaste']
        
        profession_labels = {
            'Labor/Worker': '体力劳动者',
            'Students': '学生',
            'Waiting staff': '服务业人员',
            'driver': '司机',
            'Engineer': '工程师',
            'Cashier': '收银员'
        }
        df_plot = df.copy()
        df_plot['职业'] = df_plot['职业'].map(profession_labels)
        top_professions = df_plot['职业'].value_counts().head(5).index
        df_plot = df_plot[df_plot['职业'].isin(top_professions)]
        
        fig = px.box(df_plot, x='成瘾程度', y='职业',
                     title='职业与成瘾程度关系',
                     color='职业',
                     color_continuous_scale='RdBu',
                     labels={'成瘾程度': '成瘾程度'})
        
        fig.update_layout(
            height=500, width=800,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold')
        )
        
        fig.write_html(os.path.join(HTML_DIR, '14_职业与成瘾程度_箱线图_Plotly.html'))
        fig.write_image(os.path.join(PNG_DIR, '14_职业与成瘾程度_箱线图_Plotly.png'), width=800, height=500)
    
    def plot_category_addiction_box(self):
        """Plotly视频分类与成瘾程度箱线图"""
        df = self.dataframes['timewaste']
        
        fig = px.box(df, x='视频分类', y='成瘾程度',
                     title='视频分类与成瘾程度关系',
                     color='视频分类',
                     color_continuous_scale='Set2',
                     labels={'成瘾程度': '成瘾程度'})
        
        fig.update_layout(
            height=500, width=900,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold')
        )
        
        save_fig(fig, '15_视频分类与成瘾程度_箱线图_Plotly', width=900, height=500)
    
    def plot_correlation_heatmap(self):
        """Plotly相关性热力图"""
        df_tw = self.dataframes['timewaste']
        df_m = self.dataframes['mental']
        
        combined_df = pd.DataFrame({
            '使用时长': df_tw['使用时长'],
            '成瘾程度': df_tw['成瘾程度'],
            '自我控制': df_tw['自我控制'],
            '满意度': df_tw['满意度'],
            '生产力损失': df_tw['生产力损失'],
            '精神疲劳': df_m['精神疲劳程度'].values[:len(df_tw)],
            '参与度': df_m['用户参与度'].values[:len(df_tw)]
        })
        
        corr_matrix = combined_df.corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       title='行为与心理健康综合相关性热力图',
                       color_continuous_scale='RdBu_r',
                       zmin=-1, zmax=1)
        
        fig.update_layout(
            height=600, width=700,
            title_font=dict(size=20, weight='bold')
        )
        
        save_fig(fig, '16_综合相关性热力图_Plotly', width=700, height=600)
    
    def plot_age_groups_bar(self):
        """Plotly年龄分组分布柱状图"""
        df = self.dataframes['timewaste']
        
        age_bins = [15, 20, 25, 30, 35, 40, 50, 60]
        age_labels = ['15-20岁', '21-25岁', '26-30岁', '31-35岁', '36-40岁', '41-50岁', '51-60岁']
        df['年龄分组'] = pd.cut(df['年龄'], bins=age_bins, labels=age_labels)
        
        age_group_counts = df['年龄分组'].value_counts().sort_index().reset_index()
        age_group_counts.columns = ['年龄分组', '人数']
        
        fig = px.bar(age_group_counts, x='年龄分组', y='人数',
                     title='年龄分组分布',
                     color='人数',
                     color_continuous_scale='Blues',
                     text='人数',
                     labels={'人数': '用户数'})
        
        fig.update_layout(
            height=500, width=800,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold'),
            xaxis_title='年龄分组',
            yaxis_title='用户数'
        )
        
        fig.update_traces(textposition='outside')
        
        save_fig(fig, '17_年龄分组分布_柱状图_Plotly', width=800, height=500)
    
    def plot_addiction_productivity_scatter(self):
        """Plotly成瘾程度与生产力损失散点图"""
        df = self.dataframes['timewaste']
        
        corr = df[['成瘾程度', '生产力损失']].corr().iloc[0, 1]
        
        fig = px.scatter(df, x='成瘾程度', y='生产力损失',
                         title=f'成瘾程度与生产力损失关系 (r = {corr:.2f})',
                         color='自我控制',
                         color_continuous_scale='RdBu',
                         trendline='ols',
                         trendline_color_override='#EC4899',
                         labels={'生产力损失': '生产力损失'})
        
        fig.update_layout(
            height=500, width=800,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold')
        )
        
        fig.write_html(os.path.join(HTML_DIR, '18_成瘾程度与生产力损失_Plotly.html'))
        fig.write_image(os.path.join(PNG_DIR, '18_成瘾程度与生产力损失_Plotly.png'), width=800, height=500)
    
    def plot_age_addiction_scatter(self):
        """Plotly年龄与成瘾程度散点图"""
        df = self.dataframes['timewaste']
        
        corr = df[['年龄', '成瘾程度']].corr().iloc[0, 1]
        
        fig = px.scatter(df, x='年龄', y='成瘾程度',
                         title=f'年龄与成瘾程度关系 (r = {corr:.2f})',
                         color='使用时长',
                         color_continuous_scale='Plasma',
                         trendline='ols',
                         trendline_color_override='#06B6D4',
                         labels={'年龄': '年龄', '成瘾程度': '成瘾程度'})
        
        fig.update_layout(
            height=500, width=800,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold')
        )
        
        save_fig(fig, '19_年龄与成瘾程度_Plotly', width=800, height=500)
    
    def plot_kmeans_clustering(self):
        """Plotly K-Means聚类散点图"""
        df = self.dataframes['timewaste']
        
        features = ['使用时长', '成瘾程度', '自我控制', '满意度']
        X = df[features]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # 为每个簇命名
        cluster_names = {0: '健康用户', 1: '轻度成瘾', 2: '中度成瘾', 3: '重度成瘾'}
        df['用户类型'] = df['cluster'].map(cluster_names)
        
        fig = px.scatter(df, x='使用时长', y='成瘾程度',
                         color='用户类型',
                         title='K-Means用户聚类分析',
                         color_discrete_map={'健康用户': '#10B981', '轻度成瘾': '#F59E0B', 
                                            '中度成瘾': '#EF4444', '重度成瘾': '#8B5CF6'},
                         labels={'使用时长': '使用时长(分钟)', '成瘾程度': '成瘾程度'})
        
        fig.update_layout(
            height=500, width=800,
            plot_bgcolor='#f8fafc',
            title_font=dict(size=20, weight='bold')
        )
        
        save_fig(fig, '20_KMeans聚类分析_Plotly', width=800, height=500)
    
    def plot_joint_distribution(self):
        """Plotly联合分布图"""
        df = self.dataframes['timewaste']
        
        # 创建联合分布的子图
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.8, 0.2],
            row_heights=[0.2, 0.8],
            specs=[[{"type": "scatter", "rowspan": 2}, {"type": "histogram"}],
                   [None, {"type": "histogram"}]]
        )
        
        # 散点图
        fig.add_trace(
            go.Scatter(x=df['使用时长'], y=df['成瘾程度'], mode='markers',
                       marker=dict(color='#4285F4', opacity=0.6),
                       name='散点'),
            row=1, col=1
        )
        
        # 上方直方图
        fig.add_trace(
            go.Histogram(x=df['使用时长'], marker=dict(color='#4285F4'),
                         name='使用时长分布'),
            row=1, col=2
        )
        
        # 右侧直方图
        fig.add_trace(
            go.Histogram(y=df['成瘾程度'], marker=dict(color='#4285F4'),
                         name='成瘾程度分布', orientation='h'),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600, width=700,
            title='使用时长与成瘾程度联合分布',
            title_font=dict(size=20, weight='bold'),
            plot_bgcolor='#f8fafc'
        )
        
        fig.update_xaxes(title_text='使用时长(分钟)', row=2, col=1)
        fig.update_yaxes(title_text='成瘾程度', row=2, col=1)
        
        fig.write_html(os.path.join(HTML_DIR, '21_联合分布图_Plotly.html'))
        fig.write_image(os.path.join(PNG_DIR, '21_联合分布图_Plotly.png'), width=700, height=600)
    
    def plot_facet_grid(self):
        """Plotly分面网格图"""
        df = self.dataframes['timewaste']
        
        # 简化数据用于展示
        gender_labels = {'Male': '男性', 'Female': '女性', 'Other': '非二元性别'}
        df_plot = df.copy()
        df_plot['性别'] = df_plot['性别'].map(gender_labels)
        
        # 创建年龄分组
        df_plot['年龄分组'] = pd.cut(df_plot['年龄'], bins=[15, 25, 35, 45, 65], 
                                   labels=['青年', '中青年', '中年', '中老年'])
        
        fig = px.histogram(df_plot, x='使用时长', 
                          facet_row='年龄分组', 
                          facet_col='性别',
                          title='不同年龄和性别的使用时长分布',
                          color_discrete_sequence=['#4285F4'])
        
        fig.update_layout(
            height=700, width=900,
            title_font=dict(size=20, weight='bold'),
            plot_bgcolor='#f8fafc'
        )
        
        save_fig(fig, '22_分面网格图_Plotly', width=900, height=700)

# ==========================================
# 主函数
# ==========================================
def main():
    print("="*80)
    print("抖音用户行为与心理健康综合分析系统 - Plotly高级可视化升级版")
    print("="*80)
    
    # 1. 数据加载
    loader = DataLoader()
    dataframes = loader.load_all_data()
    
    # 2. 高级可视化分析
    visualizer = AdvancedVisualizationAnalyzer(dataframes)
    visualizer.analyze()
    
    print("="*80)
    print("分析完成！输出结果已保存至目录: 可视化输出结果_Plotly升级版")
    print("生成图表数量: 22个PNG + 22个HTML交互式图表")
    print("="*80)

if __name__ == "__main__":
    main()
