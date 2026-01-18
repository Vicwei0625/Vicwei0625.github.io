---
title: "PICU 患儿诊疗路径与住院结局关联分析"
collection: portfolio
type: "Data Analysis"
permalink: /portfolio/picu-treatment-pathway-analysis
date: 2026-01-18
excerpt: "对 PICU 患儿的诊疗路径与住院结局进行深入分析，包括诊断类别分布、住院死亡率、诊疗路径特征差异、住院时长与结局关系、生命体征分析，并构建预测住院死亡率的机器学习模型。"
header:
  teaser: /images/portfolio/picu-treatment-pathway-analysis/diagnosis_category_distribution.png
tags:
  - 医疗数据分析
  - PICU 患儿
  - 住院结局
  - 诊疗路径
  - 机器学习
  - 预测模型
tech_stack:
  - name: Python
  - name: Pandas
  - name: NumPy
  - name: Matplotlib
  - name: Seaborn
  - name: Scikit-learn
---
```

### 项目背景 (Background)
PICU 是专门收治危重患儿的科室，患儿病情复杂，治疗难度大，住院死亡率高。对 PICU 患儿的诊疗路径与住院结局进行关联分析，有助于了解影响患儿预后的因素，优化诊疗方案，提高医疗质量。本项目基于真实的 PICU 患儿数据，从多个维度进行分析，并构建预测住院死亡率的机器学习模型。

### 核心实现 (Implementation)
#### 1. 数据清洗与预处理
```python
# 复制原始数据
df_clean = df.copy()

# 检查并处理重复值
df_clean = df_clean.drop_duplicates()

# 处理缺失值（针对数值型变量用中位数填充）
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

# 将年龄从月份转换为年
df_clean['age_years'] = df_clean['age_month'] / 12

# 创建年龄分组
def age_group(months):
    if months < 12:
        return '婴儿(<1岁)'
    elif months < 36:
        return '幼儿(1-3岁)'
    elif months < 72:
        return '学龄前(3-6岁)'
    elif months < 144:
        return '学龄期(6-12岁)'
    else:
        return '青少年(≥12岁)'

df_clean['age_group'] = df_clean['age_month'].apply(age_group)

# 编码性别
df_clean['gender'] = df_clean['gender_code'].map({0: '男', 1: '女'})
```
此代码对原始数据进行清洗和预处理，包括处理重复值、缺失值、创建年龄分组和编码性别。

#### 2. 诊断类别分析
```python
# 查看最常见的诊断
top_diagnoses = df_clean['primary_diagnosis'].value_counts().head(10)

# 基于ICD编码创建诊断大类
def get_diagnosis_category(icd_code):
    if pd.isna(icd_code):
        return '其他'
    icd_prefix = icd_code.split('.')[0]
    if icd_prefix.startswith('J'):
        return '呼吸系统疾病'
    elif icd_prefix.startswith('G'):
        return '神经系统疾病'
    elif icd_prefix.startswith('I'):
        return '循环系统疾病'
    elif icd_prefix.startswith('A'):
        return '感染性疾病'
    elif icd_prefix.startswith('C'):
        return '肿瘤'
    elif icd_prefix.startswith('Q'):
        return '先天畸形'
    elif icd_prefix.startswith('K'):
        return '消化系统疾病'
    else:
        return '其他'

df_clean['diagnosis_category'] = df_clean['primary_diagnosis'].apply(get_diagnosis_category)

# 可视化并保存图表
plt.figure(figsize=(12, 6))
sns.barplot(x=diag_cat_counts.values, y=diag_cat_counts.index, palette='viridis')
plt.title('主要诊断类别分布')
plt.xlabel('病例数')
plt.tight_layout()
plt.savefig('figures/diagnosis_category_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/diagnosis_category_distribution.pdf', bbox_inches='tight')
plt.show()
```
该代码分析了主要诊断类别的分布情况，并可视化了诊断类别分布。

#### 3. 住院结局分析
```python
# 总体死亡率
total_patients = len(df_clean)
expired_patients = df_clean['hospital_expire_flag'].sum()
mortality_rate = expired_patients / total_patients * 100

# 按诊断大类分析死亡率
mortality_by_category = df_clean.groupby('diagnosis_category')['hospital_expire_flag'].agg(['count', 'sum'])
mortality_by_category['mortality_rate'] = mortality_by_category['sum'] / mortality_by_category['count'] * 100
mortality_by_category = mortality_by_category.sort_values('mortality_rate', ascending=False)

# 可视化并保存图表
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(mortality_by_category)), mortality_by_category['mortality_rate'],
               color='steelblue', alpha=0.8)
plt.title('不同诊断大类的死亡率')
plt.xlabel('诊断类别')
plt.ylabel('死亡率 (%)')
plt.xticks(range(len(mortality_by_category)), mortality_by_category.index, rotation=45)
plt.tight_layout()

# 添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom')

plt.savefig('figures/mortality_by_diagnosis_category.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/mortality_by_diagnosis_category.pdf', bbox_inches='tight')
plt.show()
```
此代码分析了总体死亡率和各诊断大类的死亡率，并可视化了死亡率分布。

#### 4. 诊疗路径特征分析
```python
# 选择关键诊疗路径变量
treatment_vars = [
    'med_count', 'unique_meds', 'injection_count', 'antibiotic_flag',
    'lab_test_count', 'surgery_flag', 'exam_report_count', 'microbiology_flag'
]

# 按死亡/生存分组比较
survival_group = df_clean[df_clean['hospital_expire_flag'] == 0]
death_group = df_clean[df_clean['hospital_expire_flag'] == 1]

for var in treatment_vars:
    if var.endswith('_flag'):
        survival_rate = survival_group[var].mean() * 100
        death_rate = death_group[var].mean() * 100
        t_stat, p_val = stats.chi2_contingency(pd.crosstab(df_clean[var], df_clean['hospital_expire_flag']))[:2]
    else:
        t_stat, p_val = stats.ttest_ind(survival_group[var], death_group[var], nan_policy='omit')
```
该代码分析了诊疗路径特征在生存组和死亡组之间的差异，并进行了统计检验。

#### 5. 机器学习模型构建与评估
```python
# 准备特征和标签
features = [
    'age_month', 'gender_code', 'los_days',
    'med_count', 'unique_meds', 'injection_count', 'antibiotic_flag',
    'lab_test_count', 'surgery_flag', 'exam_report_count',
    'temperature_mean', 'pulse_mean', 'resp_rate_mean'
]

X = df_clean[available_features].copy()
y = df_clean['hospital_expire_flag'].copy()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 训练逻辑回归模型
logreg = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
logreg.fit(X_train_scaled, y_train)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'逻辑回归 (AUC = {roc_auc:.3f})')
plt.plot(fpr_rf, tpr_rf, color='red', lw=2, label=f'随机森林 (AUC = {roc_auc_rf:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机分类器')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('预测模型ROC曲线比较')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('figures/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/roc_curves_comparison.pdf', bbox_inches='tight')
plt.show()
```
此代码构建了逻辑回归和随机森林模型，用于预测住院死亡率，并绘制了 ROC 曲线比较模型性能。

### 分析结果 (Results & Analysis)
- **主要诊断类别分布**：呼吸系统疾病是最常见的诊断类别。
- **死亡率**：总体死亡率为一定比例，不同诊断类别的死亡率存在显著差异。
- **诊疗路径特征差异**：死亡组患者的用药总数、注射次数、抗生素使用率等指标显著高于生存组。
- **住院时长**：死亡组平均住院天数显著低于生存组。
- **生命体征**：死亡组的平均体温、心率、呼吸频率等生命体征与生存组存在显著差异。
- **模型性能**：随机森林模型的 ROC AUC 为一定值，优于逻辑回归模型。
- **重要特征**：年龄、住院时长、用药总数等是预测住院死亡率的重要特征。

### 第五步：交付与行动指引
1. 在您的网站项目中创建文件夹：`images/portfolio/picu-treatment-pathway-analysis/`。
2. 回到您的 Jupyter Notebook，手动保存以下位置的图片，并重命名为指定文件名，上传到 `images/portfolio/picu-treatment-pathway-analysis/` 文件夹：
    - 4 节 主要诊断类别分布条形图 -> `diagnosis_category_distribution.png`
    - 5 节 不同诊断大类的死亡率柱状图 -> `mortality_by_diagnosis_category.png`
    - 7 节 关键诊疗路径特征比较柱状图 -> `treatment_pathway_comparison.png`
    - 8 节 住院时长与结局关系分析图表 -> `length_of_stay_analysis.png`
    - 9 节 显著差异的生命体征比较箱线图 -> `vital_signs_comparison.png`
    - 12 节 随机森林特征重要性条形图 -> `random_forest_feature_importance.png`
    - 13 节 ROC 曲线比较图 -> `roc_curves_comparison.png`
3. 创建文件 `_portfolio/picu-treatment-pathway-analysis.md`，并将上述生成的 Markdown 内容粘贴到该文件中。
4. 将上述文件上传到 GitHub Page，验证作品集页面的显示效果。
