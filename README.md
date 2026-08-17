<div align="center">

# 📊 Douyin User Behavior & Mental Health Visualization

### 抖音用户行为模式与心理健康可视化研究

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)]()
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)]()
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)]()
[![Nanchang University](https://img.shields.io/badge/Nanchang_University-际銮书院-2563EB?style=for-the-badge&logo=bookstack&logoColor=white)](https://www.ncu.edu.cn)

</div>

---

## 📖 Project Overview

This project conducts a **systematic multi-dimensional analysis** of Douyin (TikTok China) users, integrating **user profiling, usage behavior, mental health, e-commerce consumption, and platform operations**. Using publicly available datasets, web-scraped data, and platform statistics, we uncover the intrinsic relationships between **short-video usage patterns, addiction risk, mental well-being, and commercial value**.

> 🎯 **Core Goal:** Reveal how usage duration, content type, and user demographics correlate with addiction levels and mental fatigue — providing data-driven insights for platform health operation, addiction intervention, content optimization, and e-commerce marketing.

---

## 🔍 Research Background

With **1.16 billion short-video users** in China (90.6% of netizens) and Douyin's global MAU exceeding **1.5 billion**, short-video platforms have become the most concentrated traffic ecosystem. However, the high-immersion, high-stimulation nature of short videos brings concerns:

- ⏱️ Average daily usage exceeds **90 minutes**; adolescents exceed **120 minutes**
- 🧠 ~35% of adolescents report varying degrees of phone dependency
- 😰 Overuse linked to attention fragmentation, mental fatigue, anxiety, and social isolation

**Research Gap:** Existing studies are mostly single-dimensional; this project builds an **integrated framework** connecting user behavior, addiction mechanisms, mental health, and business value.

---

## 📊 Data Sources

| Dataset | Description | Source |
|:---|:---|:---|
| `social_media_usage_mental_health.csv` | User demographics, usage duration, addiction scales, mental health indicators | Public dataset |
| `Time-Wasters on Social Media.csv` | Cross-platform social media usage & time-wasting patterns | Public dataset |
| `social_ecommerce_data.csv` | Douyin e-commerce consumption, category preferences, spending tiers | Scraped & compiled |
| `platform_statistics_2026.csv` | Platform MAU, growth rates, industry rankings (QuestMobile TRUTH) | Industry reports |

**Data Processing:** Complete pipeline including cleaning, outlier handling, missing value imputation, feature engineering, and normalization — documented in `data-preprocessing.md`.

---

## 🧪 Analysis Methodology

### 1. Descriptive Statistics
User profile distribution, usage patterns, consumption tiers, and mental health score distributions.

### 2. Correlation Analysis (Pearson)
Heatmaps revealing relationships between usage duration, addiction scores, self-control, mental fatigue, and anxiety.

### 3. Regression Analysis
Linear & multiple regression modeling the impact of demographic and behavioral factors on addiction risk.

### 4. Comparative Analysis
Cross-group comparisons by age, gender, occupation, and content preference.

### 5. Multi-dimensional Association
Joint analysis of behavior × mental health × consumption patterns.

### 6. K-Means Clustering
User segmentation into **four distinct groups** based on behavioral and psychological features.

---

## 🏆 Key Findings

### 📌 Core Correlations

| Relationship | Direction | Strength |
|:---|:---:|:---:|
| Usage duration ↔ Addiction level | ⬆️ Positive | Significant |
| Usage duration ↔ Mental fatigue | ⬆️ Positive | Significant |
| Self-control ↔ Addiction level | ⬇️ Negative | Strong |
| Age ↔ Usage habits | — | Significant |

### 📌 User Clusters (K-Means)

| Cluster | Profile | Characteristics |
|:---|:---|:---|
| 🔴 **High Addiction Risk** | Heavy users, low self-control, high fatigue | Long duration, compulsive checking |
| 🟢 **Healthy Users** | Moderate usage, high self-control | Purpose-driven viewing, good mental state |
| 🟡 **Moderate Users** | Average duration, mixed patterns | Occasional binge-watching |
| 🔵 **Light Users** | Minimal usage, low engagement | Infrequent opening, low addiction risk |

### 📌 E-commerce Insights
- Consumption shows clear **hierarchical stratification**
- Category preferences are **highly concentrated**
- Addiction risk correlates with **impulse buying** behavior

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

</div>

| Layer | Tools |
|:---|:---|
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly (interactive dashboards) |
| **Machine Learning** | Scikit-learn (K-Means, Linear Regression) |
| **Data Collection** | Web scraping, public APIs |

---

## 📁 Project Structure

```
douyin-mental-health-viz/
├── README.md                          # Project documentation
├── data/                              # Datasets (CSV)
│   ├── social_media_usage_mental_health.csv
│   ├── Time-Wasters on Social Media.csv
│   ├── social_ecommerce_data.csv
│   └── platform_statistics_2026.csv
├── src/                               # Visualization scripts
│   ├── visualization-integrated.py    # Full integrated dashboard
│   ├── visualization-plotly.py        # Plotly interactive version
│   └── visualization-hybrid.py        # Mixed static + interactive
├── assets/                            # Charts & figures
├── docs/                              # Reports & presentations
│   ├── full-report.md                 # Complete lab report
│   └── data-preprocessing.md          # Data processing log
└── research-framework.png             # Analysis workflow
```

---

## 📈 Visualization Highlights

The project produces a rich set of visualizations:

- 🔥 **Correlation Heatmaps** — multi-variable relationship matrices
- 📈 **Scatter Plots** — usage duration vs. addiction / mental health
- 📦 **Box Plots** — cross-group comparisons by demographics
- 🎯 **K-Means Cluster Visualization** — user segmentation in 2D space
- 📊 **Comprehensive Dashboards** — multi-panel interactive Plotly dashboards

---

## 📄 Documents

<div align="center">

| 📘 Full Report | 📊 Presentation Slides |
|:---:|:---:|
| [**Download Report (PDF)**](./report.pdf) | [**Download Slides (PDF)**](./presentation.pdf) |

</div>

---

## 👤 Author

| Field | Details |
|:---|:---|
| **Name** | Yuan Dingxiang (苑鼎祥) |
| **Student ID** | 6118123011 |
| **Class** | AI Experimental Class, Grade 2023 |
| **College** | JiLuan College, Nanchang University |
| **Group** | Group 9 |
| **Date** | May 2026 |

---

<div align="center">

*Data Visualization Course Project · Nanchang University · 2026*

</div>
