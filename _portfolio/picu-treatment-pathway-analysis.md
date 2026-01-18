---
title: "PICU患儿诊疗路径分析与死亡率预测模型"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/picu-treatment-outcome-prediction
date: 2026-01-18
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
# 项目背景
本项目基于某儿童重症监护室（PICU）24小时内的临床诊疗数据，旨在分析患儿的诊疗路径特征与住院结局（生存/死亡）之间的关联。通过数据清洗、特征工程、统计分析以及机器学习建模，识别影响患儿预后的关键因素，构建死亡率预测模型，为临床决策提供数据支持。

# 核心实现
# 1. 诊断类别分类
```
# 查看最常见的诊断
#print("最常见的10个主要诊断:")
top_diagnoses = df_clean['primary_diagnosis'].value_counts().head(10)
display(top_diagnoses)

# 保存诊断分布表格
top_diagnoses_df = pd.DataFrame(top_diagnoses).reset_index()
top_diagnoses_df.columns = ['诊断编码', '病例数']
top_diagnoses_df.to_csv('tables/top_diagnoses.csv', index=False, encoding='utf-8-sig')
#print("已保存: tables/top_diagnoses.csv")

# 基于ICD编码创建诊断大类
def get_diagnosis_category(icd_code):
    if pd.isna(icd_code):
        return '其他'

    icd_prefix = icd_code.split('.')[0]

    # 根据ICD-10分类
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

#print("\n诊断大类分布:")
diag_cat_counts = df_clean['diagnosis_category'].value_counts()
display(diag_cat_counts)

# 保存诊断大类分布表格
diag_cat_counts_df = pd.DataFrame(diag_cat_counts).reset_index()
diag_cat_counts_df.columns = ['诊断类别', '病例数']
diag_cat_counts_df['百分比'] = (diag_cat_counts_df['病例数'] / diag_cat_counts_df['病例数'].sum() * 100).round(1)
diag_cat_counts_df.to_csv('tables/diagnosis_category_distribution.csv', index=False, encoding='utf-8-sig')
#print("已保存: tables/diagnosis_category_distribution.csv")

# 可视化并保存图表
plt.figure(figsize=(12, 6))
sns.barplot(x=diag_cat_counts.values, y=diag_cat_counts.index, palette='viridis')
plt.title('主要诊断类别分布')
plt.xlabel('病例数')
plt.tight_layout()
plt.savefig('figures/diagnosis_category_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/diagnosis_category_distribution.pdf', bbox_inches='tight')
plt.show()
#print("已保存: figures/diagnosis_category_distribution.png/.pdf")
```
# 2. 诊疗路径特征分析
```
# Cell 6: 诊疗路径特征分析（修改版）
plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用黑体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 选择关键诊疗路径变量
treatment_vars = [
    'med_count', 'unique_meds', 'injection_count', 'antibiotic_flag',
    'lab_test_count', 'surgery_flag', 'exam_report_count', 'microbiology_flag'
]

# 按死亡/生存分组比较
survival_group = df_clean[df_clean['hospital_expire_flag'] == 0]
death_group = df_clean[df_clean['hospital_expire_flag'] == 1]

#print("诊疗路径特征比较 (生存组 vs 死亡组):")
treatment_comparison_data = []

for var in treatment_vars:
    if var.endswith('_flag'):
        # 对于标志变量，计算比例
        survival_rate = survival_group[var].mean() * 100
        death_rate = death_group[var].mean() * 100
        t_stat, p_val = stats.chi2_contingency(pd.crosstab(df_clean[var], df_clean['hospital_expire_flag']))[:2]

        #print(f"\n{var}:")
        #print(f"  生存组: {survival_rate:.1f}%")
        #print(f"  死亡组: {death_rate:.1f}%")
        #print(f"  χ²检验p值: {p_val:.4f}")

        treatment_comparison_data.append({
            '变量': var,
            '类型': '分类变量',
            '生存组': f"{survival_rate:.1f}%",
            '死亡组': f"{death_rate:.1f}%",
            'p值': p_val,
            '差异显著': '是' if p_val < 0.05 else '否'
        })
    else:
        # 对于连续变量，计算均值和统计检验
        t_stat, p_val = stats.ttest_ind(survival_group[var], death_group[var], nan_policy='omit')

        #print(f"\n{var}:")
        #print(f"  生存组均值: {survival_group[var].mean():.2f}")
        #print(f"  死亡组均值: {death_group[var].mean():.2f}")
        #print(f"  t检验p值: {p_val:.4f}")
       # if p_val < 0.05:
          #  print(f"  *差异显著 (p<0.05)")

        treatment_comparison_data.append({
            '变量': var,
            '类型': '连续变量',
            '生存组': f"{survival_group[var].mean():.2f}",
            '死亡组': f"{death_group[var].mean():.2f}",
            'p值': p_val,
            '差异显著': '是' if p_val < 0.05 else '否'
        })

# 保存诊疗路径特征比较表格
treatment_comparison_df = pd.DataFrame(treatment_comparison_data)
treatment_comparison_df.to_csv('tables/treatment_pathway_comparison.csv', index=False, encoding='utf-8-sig')
#print("\n已保存: tables/treatment_pathway_comparison.csv")

```

# 3. 逻辑回归预测模型及训练
```
# 处理缺失值
X = X.fillna(X.median())

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#print(f"训练集大小: {X_train.shape}")
#print(f"测试集大小: {X_test.shape}")
#print(f"训练集死亡率: {y_train.mean():.3f}")
#print(f"测试集死亡率: {y_test.mean():.3f}")

# Cell 11: 训练逻辑回归模型（修改版）
# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练逻辑回归模型
logreg = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
logreg.fit(X_train_scaled, y_train)

# 预测
y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]

# 评估模型
#print("逻辑回归模型评估:")
#print("\n分类报告:")
#print(classification_report(y_test, y_pred))

#print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['实际生存', '实际死亡'], columns=['预测生存', '预测死亡'])
display(cm_df)

# 计算AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
#print(f"\nROC AUC: {roc_auc:.3f}")

# 特征重要性
feature_importance = pd.DataFrame({
    '特征': available_features,
    '系数': logreg.coef_[0],
    '绝对值系数': np.abs(logreg.coef_[0])
}).sort_values('绝对值系数', ascending=False)

#print("\n特征重要性 (系数绝对值排序):")
display(feature_importance.head(10))

# 保存逻辑回归模型结果
# 混淆矩阵
confusion_matrix_df = pd.DataFrame({
    '实际\\预测': ['实际生存', '实际死亡'],
    '预测生存': [cm[0, 0], cm[1, 0]],
    '预测死亡': [cm[0, 1], cm[1, 1]]
})
confusion_matrix_df.to_csv('tables/logistic_regression_confusion_matrix.csv', index=False, encoding='utf-8-sig')

# 特征重要性
feature_importance.to_csv('tables/logistic_regression_feature_importance.csv', index=True, encoding='utf-8-sig')

# 保存模型性能指标
lr_performance = pd.DataFrame({
    '指标': ['ROC AUC', '准确率', '精确率(死亡)', '召回率(死亡)', 'F1分数(死亡)'],
    '数值': [
        roc_auc,
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, pos_label=1),
        recall_score(y_test, y_pred, pos_label=1),
        f1_score(y_test, y_pred, pos_label=1)
    ]
})
lr_performance.to_csv('tables/logistic_regression_performance.csv', index=False, encoding='utf-8-sig')

#print("已保存逻辑回归模型相关表格")
```

# 4. 随机森林模型
```
# Cell 12: 随机森林模型（修改版）
plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用黑体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# 预测
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

# 评估模型
#print("随机森林模型评估:")
#print("\n分类报告:")
#print(classification_report(y_test, y_pred_rf))

#print("\n混淆矩阵:")
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_rf_df = pd.DataFrame(cm_rf, index=['实际生存', '实际死亡'], columns=['预测生存', '预测死亡'])
display(cm_rf_df)

# 计算AUC
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
#print(f"\nROC AUC: {roc_auc_rf:.3f}")

# 特征重要性
rf_feature_importance = pd.DataFrame({
    '特征': available_features,
    '重要性': rf.feature_importances_
}).sort_values('重要性', ascending=False)

#print("\n随机森林特征重要性:")
display(rf_feature_importance.head(10))

# 保存随机森林模型结果
# 混淆矩阵
confusion_matrix_rf_df = pd.DataFrame({
    '实际\\预测': ['实际生存', '实际死亡'],
    '预测生存': [cm_rf[0, 0], cm_rf[1, 0]],
    '预测死亡': [cm_rf[0, 1], cm_rf[1, 1]]
})
confusion_matrix_rf_df.to_csv('tables/random_forest_confusion_matrix.csv', index=False, encoding='utf-8-sig')

# 特征重要性
rf_feature_importance.to_csv('tables/random_forest_feature_importance.csv', index=True, encoding='utf-8-sig')

# 保存模型性能指标
rf_performance = pd.DataFrame({
    '指标': ['ROC AUC', '准确率', '精确率(死亡)', '召回率(死亡)', 'F1分数(死亡)'],
    '数值': [
        roc_auc_rf,
        accuracy_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_rf, pos_label=1),
        recall_score(y_test, y_pred_rf, pos_label=1),
        f1_score(y_test, y_pred_rf, pos_label=1)
    ]
})
rf_performance.to_csv('tables/random_forest_performance.csv', index=False, encoding='utf-8-sig')

# 可视化特征重要性并保存
plt.figure(figsize=(10, 6))
top_features = rf_feature_importance.head(10)
sns.barplot(data=top_features, x='重要性', y='特征', palette='viridis')
plt.title('随机森林特征重要性 (Top 10)')
plt.tight_layout()
plt.savefig('figures/random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/random_forest_feature_importance.pdf', bbox_inches='tight')
plt.show()
#print("已保存: figures/random_forest_feature_importance.png/.pdf")

#print("已保存随机森林模型相关表格")

```

# 分析结果
# 1. 诊断分布与死亡率
![图1: 主要诊断类别分类](/images/portfolio/diagnosis_category_distribution.png)
数据分析显示，呼吸系统疾病是PICU最常见的诊断类别（占45.2%），其次是神经系统疾病（28.7%）和感染性疾病（12.3%）。
![图2: 不同诊断大类的死亡率](/images/portfolio/mortality_by_diagnosis_category.png)
循环系统疾病的死亡率最高（15.6%），其次是神经系统疾病（9.8%）和先天畸形（8.2%）。总体死亡率为7.3%。

# 2. 诊疗路径特征差异

![图3: 显著差异的生命体征比较](/images/portfolio/vital_signs_comparison.png)

死亡组相比生存组显示：

用药总数更高：12.4 vs 8.2（p<0.001）

注射次数更多：5.8 vs 3.1（p<0.001）

抗生素使用率更高：78.3% vs 62.1%（p<0.05）

# 3. 住院时长分析
   
![图4: 住院时长与结局的关系](/images/portfolio/length_of_stay_analysis.png)

生存组平均住院时长为6.2天，死亡组为4.8天（p<0.05）。死亡组住院时长分布更集中，多数在7天内。

# 4. 生命体征差异

![图5: 关键诊疗路径特征比较](/images/portfolio/treatment_pathway_comparison.png)

死亡组平均心率（124.3次/分）显著高于生存组（112.7次/分，p<0.01），平均呼吸频率（32.1次/分）也显著高于生存组（26.4次/分，p<0.05）。

# 5. 模型性能与特征重要性

![图6: 随机森林特征重要性](/images/portfolio/random_forest_feature_importance.png)

随机森林模型识别的最重要预测特征包括：平均呼吸频率（重要性：0.186）、平均心率（重要性：0.152）、用药总数（重要性：0.134）、住院天数（重要性：0.121）、注射次数（重要性：0.098）。

![图7: 预测模型ROC曲线比较](/images/portfolio/roc_curves_comparison.png)

随机森林模型的AUC为0.832，优于逻辑回归模型（AUC=0.786）。随机森林在召回率（死亡）方面表现更好（0.741 vs 0.685），能更有效地识别高危患者。
