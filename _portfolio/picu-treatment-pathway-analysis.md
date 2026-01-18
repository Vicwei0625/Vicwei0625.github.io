---
title: "PICU患儿诊疗路径分析与死亡率预测模型"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/picu-treatment-outcome-prediction
date: 2024-01-01
excerpt: "基于真实临床数据构建机器学习模型，分析儿童重症监护室患儿的诊疗路径特征与住院结局关联，预测住院死亡率"
header:
  teaser: /images/portfolio/picu-treatment-outcome-prediction/diagnosis_category_distribution.png
tags:
  - 医疗数据分析
  - 机器学习
  - 临床预测模型
  - 儿科重症监护
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: Pandas
  - name: Matplotlib
  - name: Seaborn
---
项目背景
本项目基于某儿童重症监护室（PICU）24小时内的临床诊疗数据，旨在分析患儿的诊疗路径特征与住院结局（生存/死亡）之间的关联。通过数据清洗、特征工程、统计分析以及机器学习建模，识别影响患儿预后的关键因素，构建死亡率预测模型，为临床决策提供数据支持。

核心实现
1. 数据预处理与特征工程

'''
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
'''

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
2. 诊疗路径特征分析
python
# 选择关键诊疗路径变量
treatment_vars = [
    'med_count', 'unique_meds', 'injection_count', 'antibiotic_flag',
    'lab_test_count', 'surgery_flag', 'exam_report_count', 'microbiology_flag'
]

# 按死亡/生存分组比较
survival_group = df_clean[df_clean['hospital_expire_flag'] == 0]
death_group = df_clean[df_clean['hospital_expire_flag'] == 1]

# 统计分析
for var in treatment_vars:
    if var.endswith('_flag'):
        survival_rate = survival_group[var].mean() * 100
        death_rate = death_group[var].mean() * 100
        t_stat, p_val = stats.chi2_contingency(pd.crosstab(df_clean[var], df_clean['hospital_expire_flag']))[:2]
    else:
        t_stat, p_val = stats.ttest_ind(survival_group[var], death_group[var], nan_policy='omit')
3. 逻辑回归预测模型
python
# 准备特征和标签
features = [
    'age_month', 'gender_code', 'los_days',
    'med_count', 'unique_meds', 'injection_count', 'antibiotic_flag',
    'lab_test_count', 'surgery_flag', 'exam_report_count',
    'temperature_mean', 'pulse_mean', 'resp_rate_mean'
]

available_features = [f for f in features if f in df_clean.columns]
X = df_clean[available_features].copy()
y = df_clean['hospital_expire_flag'].copy()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 训练逻辑回归模型
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
logreg.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
4. 随机森林模型
python
# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# 预测与评估
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

# 特征重要性分析
rf_feature_importance = pd.DataFrame({
    '特征': available_features,
    '重要性': rf.feature_importances_
}).sort_values('重要性', ascending=False)
分析结果
1. 诊断分布与死亡率
![alt text](https://github.com/username/reponame/blob/branch/image.jpg?raw=true)
![Project Logo]((https://images/portfolio/picu-treatment-outcome-prediction/diagnosis_category_distribution.png)?raw=true)
数据分析显示，呼吸系统疾病是PICU最常见的诊断类别（占45.2%），其次是神经系统疾病（28.7%）和感染性疾病（12.3%）。
![alt text](https://github.com/username/reponame/blob/branch/image.jpg?raw=true)
![Project Logo]([https://github.com/johnsmith/myproject/blob/main/images/logo.png](https:///images/portfolio/picu-treatment-outcome-prediction/mortality_by_diagnosis_category.png)?raw=true)
循环系统疾病的死亡率最高（15.6%），其次是神经系统疾病（9.8%）和先天畸形（8.2%）。总体死亡率为7.3%。

2. 诊疗路径特征差异
![alt text](https://github.com/username/reponame/blob/branch/image.jpg?raw=true)
https:///images/portfolio/picu-treatment-outcome-prediction/treatment_pathway_comparison.png

死亡组相比生存组显示：

用药总数更高：12.4 vs 8.2（p<0.001）

注射次数更多：5.8 vs 3.1（p<0.001）

抗生素使用率更高：78.3% vs 62.1%（p<0.05）

3. 住院时长分析
![alt text](https://github.com/username/reponame/blob/branch/image.jpg?raw=true)
https:///images/portfolio/picu-treatment-outcome-prediction/length_of_stay_analysis.png

生存组平均住院时长为6.2天，死亡组为4.8天（p<0.05）。死亡组住院时长分布更集中，多数在7天内。

4. 生命体征差异
![alt text](https://github.com/username/reponame/blob/branch/image.jpg?raw=true)
https:///images/portfolio/picu-treatment-outcome-prediction/vital_signs_comparison.png

死亡组平均心率（124.3次/分）显著高于生存组（112.7次/分，p<0.01），平均呼吸频率（32.1次/分）也显著高于生存组（26.4次/分，p<0.05）。

5. 模型性能与特征重要性
![alt text](https://github.com/username/reponame/blob/branch/image.jpg?raw=true)
https:///images/portfolio/picu-treatment-outcome-prediction/random_forest_feature_importance.png

随机森林模型识别的最重要预测特征包括：

平均呼吸频率（重要性：0.186）

平均心率（重要性：0.152）

用药总数（重要性：0.134）

住院天数（重要性：0.121）

注射次数（重要性：0.098）
![alt text](https://github.com/Vicwei0625/images/portfolio/picu-treatment-outcome-prediction/roc_curves_comparison?raw=true)

随机森林模型的AUC为0.832，优于逻辑回归模型（AUC=0.786）。随机森林在召回率（死亡）方面表现更好（0.741 vs 0.685），能更有效地识别高危患者。
