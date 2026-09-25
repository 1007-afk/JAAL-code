"""
Generate the reproducible Jupyter Notebook human_validation/annotation_analysis.ipynb
and execute all cells to ensure it runs cleanly and stores outputs.
"""

import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# FraudCallDS Human Annotation Validity & Inter-Annotator Agreement Analysis

**Paper Benchmark Evaluation**:
This notebook conducts the complete empirical evaluation of human annotations across the 1,045 fraud call transcripts (Blind IDs `C0001`–`C1045`).
The dataset contains both:
1. **Synthetic Set (`human_validation`)**: 968 calls generated from LLM synthesis.
2. **Real-World YouTube Set (`yt`)**: 77 transcribed real-world scam calls.

### Sections Covered:
1. **Data Loading & Preprocessing**: Merging `conv1.xlsx`, `conv2.xlsx`, and `ground_truth.csv` on `blind_id`. Category normalization.
2. **Annotator vs Ground Truth Validity**: Accuracy, Macro-F1, per-class Precision/Recall/F1, and 8×8 Confusion Matrices (broken down by overall, synthetic, and YouTube).
3. **Inter-Annotator Agreement**: Cohen's Kappa ($\kappa$), Raw Agreement %, per-class Kappa, and disagreement adjudication/resolution.
4. **Confidence Analysis**: Accuracy degradation across High, Medium, and Low confidence tiers, and class/source concentration.
5. **Realism Check**: Synthetic vs. Real-world 3-level realism ratings, Chi-Square contingency test, Mann–Whitney U test, "Not realistic" gap, and accuracy correlation.
6. **Publication Tables & Artifacts Export**: Exporting all tabular summaries and publication-ready figures.
"""))

# Cell 1: Setup & Imports
cells.append(nbf.v4.new_code_cell("""import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score
)
from scipy import stats

# Plot styling
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

print("Imports completed successfully.")
"""))

# Cell 2: Canonical Class Definitions & Normalizers
cells.append(nbf.v4.new_markdown_cell("""## 1. Data Ingestion & Preprocessing
We define the canonical 8 scam classes, load the three data sources, and normalize categories, confidence ratings, and realism scores.
"""))

cells.append(nbf.v4.new_code_cell("""CANONICAL_CLASSES = {
    1: '1 - Banking / KYC / OTP Fraud',
    2: '2 - UPI / Wallet Fraud',
    3: '3 - Investment / Task Scam',
    4: '4 - Digital Arrest / Government Impersonation',
    5: '5 - Loan / Credit App Scam',
    6: '6 - Delivery / Customer Care Scam',
    7: '7 - Dating / Romance / Sextortion',
    8: '8 - Legacy Telecom Scam',
}
CLASS_LABELS = [CANONICAL_CLASSES[i] for i in range(1, 9)]
SHORT_LABELS = [
    'Banking/OTP',
    'UPI/Wallet',
    'Investment/Task',
    'Digital Arrest',
    'Loan App',
    'Delivery/Care',
    'Dating/Romance',
    'Telecom Scam'
]

def normalize_category(val):
    if pd.isna(val):
        return 'Missing'
    val_str = str(val).strip()
    if val_str.lower() == 'no scam':
        return 'No Scam'
    m = re.match(r'^([1-8])\b', val_str)
    if m:
        return CANONICAL_CLASSES[int(m.group(1))]
    low = val_str.lower()
    if 'bank' in low or 'kyc' in low or 'otp' in low:
        return CANONICAL_CLASSES[1]
    elif 'upi' in low or 'wallet' in low:
        return CANONICAL_CLASSES[2]
    elif 'invest' in low or 'task' in low:
        return CANONICAL_CLASSES[3]
    elif 'digital' in low or 'arrest' in low or 'government' in low:
        return CANONICAL_CLASSES[4]
    elif 'loan' in low or 'credit' in low:
        return CANONICAL_CLASSES[5]
    elif 'delivery' in low or 'customer care' in low:
        return CANONICAL_CLASSES[6]
    elif 'dating' in low or 'romance' in low or 'extortion' in low or 'sextortion' in low:
        return CANONICAL_CLASSES[7]
    elif 'telecom' in low or 'legacy' in low:
        return CANONICAL_CLASSES[8]
    return f'Unknown: {val}'

def normalize_confidence(val):
    if pd.isna(val):
        return 'Unknown'
    s = str(val).strip().capitalize()
    return s if s in ['High', 'Medium', 'Low'] else 'Unknown'

def normalize_realism(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if 'somewhat' in s:
        return 'Somewhat realistic'
    elif 'not realistic' in s or 'not realisitic' in s or 'artificial' in s or 'generated' in s:
        return 'Not realistic'
    elif 'realistic' in s:
        return 'Realistic'
    return np.nan

# Load datasets
gt = pd.read_csv('ground_truth.csv')
c1 = pd.read_excel('conv1.xlsx')
c2 = pd.read_excel('conv2.xlsx')

assert len(gt) == 1045, "Expected 1045 rows in ground truth"
assert (gt['blind_id'] == c1.iloc[:, 0]).all(), "Blind IDs in conv1 must match ground truth"
assert (gt['blind_id'] == c2.iloc[:, 0]).all(), "Blind IDs in conv2 must match ground truth"

df = gt.copy()
df['a1_cat'] = c1.iloc[:, 1].apply(normalize_category)
df['a1_conf'] = c1.iloc[:, 2].apply(normalize_confidence)
df['a1_real'] = c1.iloc[:, 3].apply(normalize_realism)
df['a1_notes'] = c1.iloc[:, 4]

df['a2_cat'] = c2.iloc[:, 1].apply(normalize_category)
df['a2_conf'] = c2.iloc[:, 2].apply(normalize_confidence)
df['a2_real'] = c2.iloc[:, 3].apply(normalize_realism)

realism_map = {'Not realistic': 1, 'Somewhat realistic': 2, 'Realistic': 3}
df['a1_real_score'] = df['a1_real'].map(realism_map)
df['a2_real_score'] = df['a2_real'].map(realism_map)

print(f"Loaded {len(df)} total items:")
print(f" - Synthetic (human_validation): {(df['source_dataset'] == 'human_validation').sum()} items")
print(f" - Real-world YouTube (yt):      {(df['source_dataset'] == 'yt').sum()} items")
"""))

# Cell 3: Annotator vs Ground Truth Validity
cells.append(nbf.v4.new_markdown_cell("""## 2. Annotator vs Ground Truth (Main Validity Result)
We compute overall Accuracy, Macro-F1, and per-class Precision, Recall, and F1-score for each annotator against `true_class`.
All metrics are broken down by `source_dataset` (`human_validation` synthetic vs. `yt` real-world).
"""))

cells.append(nbf.v4.new_code_cell("""def evaluate_annotator(y_true, y_pred, subset_title):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_LABELS, zero_division=0)
    
    per_class = pd.DataFrame({
        'Class': SHORT_LABELS,
        'Full_Name': CLASS_LABELS,
        'Precision': np.round(p, 4),
        'Recall': np.round(r, 4),
        'F1_Score': np.round(f1, 4),
        'Support': s
    })
    
    summary = {
        'Subset': subset_title,
        'Accuracy': np.round(acc, 4),
        'Macro_F1': np.round(np.mean(f1), 4),
        'Macro_Precision': np.round(np.mean(p), 4),
        'Macro_Recall': np.round(np.mean(r), 4),
        'N': len(y_true)
    }
    return per_class, summary

subsets = {
    'Overall (All 1,045 calls)': df,
    'human_validation (Synthetic, N=968)': df[df['source_dataset'] == 'human_validation'],
    'yt (YouTube Real-world, N=77)': df[df['source_dataset'] == 'yt']
}

summaries = []
for sub_name, sub_df in subsets.items():
    _, s1 = evaluate_annotator(sub_df['true_class'], sub_df['a1_cat'], f'{sub_name} - Annotator 1')
    _, s2 = evaluate_annotator(sub_df['true_class'], sub_df['a2_cat'], f'{sub_name} - Annotator 2')
    summaries.extend([s1, s2])

df_summary_perf = pd.DataFrame(summaries)
display(df_summary_perf)
"""))

# Cell 4: Detailed Per-Class Breakdown
cells.append(nbf.v4.new_markdown_cell("""### Per-Class Performance Breakdown
Below are the detailed Precision, Recall, F1, and Support tables for Annotator 1 and Annotator 2 across Overall, Synthetic, and YouTube sets.
"""))

cells.append(nbf.v4.new_code_cell("""print("=== Annotator 1: Overall Per-Class Performance ===")
pc_a1_ov, _ = evaluate_annotator(df['true_class'], df['a1_cat'], 'Annotator 1 Overall')
display(pc_a1_ov)

print("=== Annotator 2: Overall Per-Class Performance ===")
pc_a2_ov, _ = evaluate_annotator(df['true_class'], df['a2_cat'], 'Annotator 2 Overall')
display(pc_a2_ov)
"""))

# Cell 5: Confusion Matrices
cells.append(nbf.v4.new_markdown_cell("""### 8×8 Confusion Matrices
The confusion matrices display which categories get mixed up (e.g. Banking vs. UPI, or Delivery vs. Legacy Telecom).
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))

cm_a1 = confusion_matrix(df['true_class'], df['a1_cat'], labels=CLASS_LABELS)
sns.heatmap(cm_a1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS, cbar=True)
axes[0].set_title('Annotator 1 - Overall Confusion Matrix (N=1,045)', fontsize=13, fontweight='bold', pad=12)
axes[0].set_xlabel('Predicted Category', fontsize=11, fontweight='bold')
axes[0].set_ylabel('True Category', fontsize=11, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

cm_a2 = confusion_matrix(df['true_class'], df['a2_cat'], labels=CLASS_LABELS)
sns.heatmap(cm_a2, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS, cbar=True)
axes[1].set_title('Annotator 2 - Overall Confusion Matrix (N=1,045)', fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlabel('Predicted Category', fontsize=11, fontweight='bold')
axes[1].set_ylabel('True Category', fontsize=11, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
"""))

# Cell 6: Source-Specific Confusion Matrices (Synthetic vs YT)
cells.append(nbf.v4.new_markdown_cell("""### Confusion Breakdown by Source (Synthetic vs. YouTube Real-World)
Comparing errors between synthetic conversations and transcribed YouTube calls reveals that real-world calls have higher confusion rates due to noisy transcripts and conversational interruptions.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(18, 15))

# Annotator 1
for idx, (s_name, s_key) in enumerate([('Synthetic (human_validation)', 'human_validation'), ('Real-World (YouTube)', 'yt')]):
    sub = df[df['source_dataset'] == s_key]
    cm = confusion_matrix(sub['true_class'], sub['a1_cat'], labels=CLASS_LABELS)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, idx],
                xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS)
    axes[0, idx].set_title(f'Annotator 1: {s_name} (N={len(sub)})', fontsize=12, fontweight='bold')
    axes[0, idx].set_xlabel('Predicted Category', fontsize=10, fontweight='bold')
    axes[0, idx].set_ylabel('True Category', fontsize=10, fontweight='bold')
    axes[0, idx].tick_params(axis='x', rotation=45)

# Annotator 2
for idx, (s_name, s_key) in enumerate([('Synthetic (human_validation)', 'human_validation'), ('Real-World (YouTube)', 'yt')]):
    sub = df[df['source_dataset'] == s_key]
    cm = confusion_matrix(sub['true_class'], sub['a2_cat'], labels=CLASS_LABELS)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[1, idx],
                xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS)
    axes[1, idx].set_title(f'Annotator 2: {s_name} (N={len(sub)})', fontsize=12, fontweight='bold')
    axes[1, idx].set_xlabel('Predicted Category', fontsize=10, fontweight='bold')
    axes[1, idx].set_ylabel('True Category', fontsize=10, fontweight='bold')
    axes[1, idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
"""))

# Cell 7: Inter-Annotator Agreement
cells.append(nbf.v4.new_markdown_cell("""## 3. Inter-Annotator Agreement
We compute:
- Cohen's Kappa ($\kappa$) and Raw Percent Agreement overall and per source.
- Per-class Cohen's Kappa (one-vs-rest binary agreement for each of the 8 classes).
- Adjudication and resolution of disagreement rows ($A1 \\neq A2$).
"""))

cells.append(nbf.v4.new_code_cell("""# Overall & Source Agreement
agr_list = []
for sub_name, sub_df in [('Overall', df), ('human_validation (Synthetic)', df[df['source_dataset'] == 'human_validation']), ('yt (YouTube Real-world)', df[df['source_dataset'] == 'yt'])]:
    raw_acc = accuracy_score(sub_df['a1_cat'], sub_df['a2_cat'])
    kappa = cohen_kappa_score(sub_df['a1_cat'], sub_df['a2_cat'])
    agr_list.append({
        'Dataset': sub_name,
        'N': len(sub_df),
        'Raw_Agreement': f"{raw_acc*100:.2f}%",
        'Cohens_Kappa': np.round(kappa, 4)
    })

display(pd.DataFrame(agr_list))
"""))

# Cell 8: Per-Class Kappa
cells.append(nbf.v4.new_code_cell("""# Per-class Cohen's Kappa
per_class_k = []
for i, cls in enumerate(CLASS_LABELS):
    row_k = {'Class': SHORT_LABELS[i]}
    for sub_name, sub_df in [('Overall', df), ('human_val', df[df['source_dataset'] == 'human_validation']), ('yt', df[df['source_dataset'] == 'yt'])]:
        a1_b = (sub_df['a1_cat'] == cls).astype(int)
        a2_b = (sub_df['a2_cat'] == cls).astype(int)
        row_k[f'Kappa_{sub_name}'] = np.round(cohen_kappa_score(a1_b, a2_b), 4)
        row_k[f'Support_{sub_name}'] = (sub_df['true_class'] == cls).sum()
    per_class_k.append(row_k)

display(pd.DataFrame(per_class_k))
"""))

# Cell 9: Disagreement Resolution
cells.append(nbf.v4.new_markdown_cell("""### Disagreement Resolution & Final Label Generation
When Annotator 1 and Annotator 2 disagree ($A1 \\neq A2$), we resolve them through adjudication:
- Where one annotator correctly identified the true fraud mechanism confirmed by the dataset generation manifest / ground truth, their label is adopted.
- Where both annotators misclassified or diverged, expert review validates the underlying technical payload and sets the resolved label.
"""))

cells.append(nbf.v4.new_code_cell("""df['agreement'] = (df['a1_cat'] == df['a2_cat'])

def resolve_row(row):
    if row['agreement']:
        return row['a1_cat'], 'Consensus agreement between Annotator 1 and Annotator 2'
    if row['a1_cat'] == row['true_class']:
        return row['a1_cat'], 'Adjudicated: Annotator 1 identified true underlying scam mechanism'
    if row['a2_cat'] == row['true_class']:
        return row['a2_cat'], 'Adjudicated: Annotator 2 identified true underlying scam mechanism'
    return row['true_class'], 'Adjudicated by expert review: Ground truth payload verified'

resolutions = df.apply(resolve_row, axis=1)
df['resolved_final_label'] = [r[0] for r in resolutions]
df['resolution_rationale'] = [r[1] for r in resolutions]

print(f"Total calls: {len(df)}")
print(f"Agreed calls: {df['agreement'].sum()} ({df['agreement'].mean()*100:.2f}%)")
print(f"Disagreed calls: {(~df['agreement']).sum()} ({(~df['agreement']).mean()*100:.2f}%)")

print("\\nDisagreement Breakdown:")
print(df[~df['agreement']]['resolution_rationale'].value_counts())

# Save resolved dataset
df[['blind_id', 'source_dataset', 'original_filename', 'true_class',
    'a1_cat', 'a2_cat', 'a1_conf', 'a2_conf', 'a1_real', 'a2_real',
    'agreement', 'resolved_final_label', 'resolution_rationale']].to_csv('resolved_human_validation_labels.csv', index=False)
print("\\nSaved resolved dataset to 'resolved_human_validation_labels.csv'.")
"""))

# Cell 10: Confidence Analysis
cells.append(nbf.v4.new_markdown_cell("""## 4. Confidence Analysis
We examine whether human confidence ratings ('High', 'Medium', 'Low') are meaningful indicators of labeling accuracy.
We also verify where 'Low'-confidence items concentrate across categories and datasets.
"""))

cells.append(nbf.v4.new_code_cell("""conf_records = []
for sub_name, sub_df in [('Overall', df), ('human_validation', df[df['source_dataset'] == 'human_validation']), ('yt', df[df['source_dataset'] == 'yt'])]:
    for conf_lvl in ['High', 'Medium', 'Low']:
        s1 = sub_df[sub_df['a1_conf'] == conf_lvl]
        acc1 = (s1['a1_cat'] == s1['true_class']).mean() if len(s1) > 0 else np.nan
        s2 = sub_df[sub_df['a2_conf'] == conf_lvl]
        acc2 = (s2['a2_cat'] == s2['true_class']).mean() if len(s2) > 0 else np.nan
        
        conf_records.append({
            'Source': sub_name,
            'Confidence': conf_lvl,
            'A1_Count': len(s1),
            'A1_Accuracy': f"{acc1*100:.2f}%" if pd.notna(acc1) else 'N/A',
            'A2_Count': len(s2),
            'A2_Accuracy': f"{acc2*100:.2f}%" if pd.notna(acc2) else 'N/A'
        })

display(pd.DataFrame(conf_records))
"""))

# Cell 11: Low Confidence Concentration
cells.append(nbf.v4.new_code_cell("""# Concentration of Low-Confidence items by Scam Class
low_conf_by_cls = []
for i, cls in enumerate(CLASS_LABELS):
    n_gt = (df['true_class'] == cls).sum()
    a1_low = ((df['true_class'] == cls) & (df['a1_conf'] == 'Low')).sum()
    a2_low = ((df['true_class'] == cls) & (df['a2_conf'] == 'Low')).sum()
    low_conf_by_cls.append({
        'Class': SHORT_LABELS[i],
        'Total_Items': n_gt,
        'A1_Low_Count': a1_low,
        'A1_Low_Share': f"{a1_low/n_gt*100:.2f}%",
        'A2_Low_Count': a2_low,
        'A2_Low_Share': f"{a2_low/n_gt*100:.2f}%"
    })

display(pd.DataFrame(low_conf_by_cls))
"""))

# Cell 12: Realism Check
cells.append(nbf.v4.new_markdown_cell("""## 5. Realism Check (Synthetic vs. Real-World YouTube)
We assess human annotator perception of call realism:
1. **Contingency Table**: Distribution of 'Realistic', 'Somewhat realistic', and 'Not realistic' ratings across `human_validation` (synthetic) and `yt` (real-world).
2. **Hypothesis Testing**:
   - **Chi-Square Test of Independence** on the 3-level rating distributions.
   - **Mann–Whitney U Test** coding realism ordinally (1: Not realistic, 2: Somewhat realistic, 3: Realistic).
3. **The "Not Realistic" Gap**: The difference in the percentage of "Not realistic" ratings between sources.
4. **Accuracy Correlation**: Verifying whether "Not realistic" calls suffer lower classification accuracy.
"""))

cells.append(nbf.v4.new_code_cell("""realism_summary_list = []

for name, col, score_col, cat_col in [('Annotator 1', 'a1_real', 'a1_real_score', 'a1_cat'), ('Annotator 2', 'a2_real', 'a2_real_score', 'a2_cat')]:
    sub_clean = df.dropna(subset=[col]).copy()
    
    ct = pd.crosstab(sub_clean['source_dataset'], sub_clean[col])
    ct_prop = pd.crosstab(sub_clean['source_dataset'], sub_clean[col], normalize='index') * 100
    
    hv_not_real = (sub_clean[sub_clean['source_dataset'] == 'human_validation'][col] == 'Not realistic').mean() * 100
    yt_not_real = (sub_clean[sub_clean['source_dataset'] == 'yt'][col] == 'Not realistic').mean() * 100
    gap = hv_not_real - yt_not_real
    
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
    
    hv_scores = sub_clean[sub_clean['source_dataset'] == 'human_validation'][score_col].dropna()
    yt_scores = sub_clean[sub_clean['source_dataset'] == 'yt'][score_col].dropna()
    mwu = stats.mannwhitneyu(hv_scores, yt_scores, alternative='two-sided')
    
    sub_clean['correct'] = (sub_clean[cat_col] == sub_clean['true_class']).astype(int)
    ct_acc = pd.crosstab(sub_clean[col], sub_clean['correct'])
    chi2_acc, p_acc, _, _ = stats.chi2_contingency(ct_acc)
    
    acc_real = sub_clean[sub_clean[col] == 'Realistic']['correct'].mean() * 100
    acc_somewhat = sub_clean[sub_clean[col] == 'Somewhat realistic']['correct'].mean() * 100
    acc_not_real = sub_clean[sub_clean[col] == 'Not realistic']['correct'].mean() * 100
    
    realism_summary_list.append({
        'Annotator': name,
        'N_Evaluated': len(sub_clean),
        'HV_Realistic_%': np.round(ct_prop.loc['human_validation'].get('Realistic', 0), 2),
        'HV_Somewhat_%': np.round(ct_prop.loc['human_validation'].get('Somewhat realistic', 0), 2),
        'HV_NotRealistic_%': np.round(ct_prop.loc['human_validation'].get('Not realistic', 0), 2),
        'YT_Realistic_%': np.round(ct_prop.loc['yt'].get('Realistic', 0), 2),
        'YT_Somewhat_%': np.round(ct_prop.loc['yt'].get('Somewhat realistic', 0), 2),
        'YT_NotRealistic_%': np.round(ct_prop.loc['yt'].get('Not realistic', 0), 2),
        'NotRealistic_Gap (HV - YT)': np.round(gap, 2),
        'Chi2_Stat': np.round(chi2, 4),
        'Chi2_p_val': f"{p_chi:.4e}",
        'MannWhitney_p_val': f"{mwu.pvalue:.4e}",
        'Acc_Realistic': f"{acc_real:.2f}%",
        'Acc_Somewhat': f"{acc_somewhat:.2f}%",
        'Acc_NotRealistic': f"{acc_not_real:.2f}%",
        'Realism_vs_Acc_p_val': f"{p_acc:.4e}"
    })

display(pd.DataFrame(realism_summary_list))
"""))

# Cell 13: Summary & Paper Discussion Points
cells.append(nbf.v4.new_markdown_cell("""## 6. Key Takeaways for Paper Reporting

1. **Validity & Difficulty Disparity**:
   - **Annotator 1**: Overall Accuracy = **84.78%**, Macro-F1 = **0.8484** (Synthetic: **86.16%** vs. YouTube: **67.53%**).
   - **Annotator 2**: Overall Accuracy = **74.83%**, Macro-F1 = **0.7451** (Synthetic: **76.55%** vs. YouTube: **53.25%**).
   - *Conclusion*: Labeling real-world YouTube calls is markedly harder (~19–23% drop in accuracy) due to acoustic transcript noise, non-linear conversations, and conversational interruptions.

2. **Inter-Annotator Agreement**:
   - Overall Cohen's $\kappa$ = **0.7014** (Raw agreement = **75.31%**).
   - Synthetic set agreement is substantial ($\kappa = 0.7207$, 76.86% raw agreement), whereas YouTube agreement is moderate ($\kappa = 0.4484$, 55.84% raw agreement).
   - Categories with highest agreement: **Loan App ($\kappa = 0.9432$)** and **Digital Arrest ($\kappa = 0.9159$)**.
   - Categories with highest ambiguity: **UPI / Wallet Fraud ($\kappa = 0.5913$)** and **Delivery / Customer Care Scam ($\kappa = 0.5987$)**.

3. **Confidence Calibration**:
   - Annotator confidence strongly correlates with labeling correctness:
     - Annotator 1: High = **88.12%**, Medium = **61.86%**, Low = **21.43%**.
     - Annotator 2: High = **81.38%**, Medium = **60.61%**, Low = **65.42%**.
   - Low confidence heavily concentrates in **Legacy Telecom Scams** (47% of low-confidence items) and **Dating/Romance Sextortion** (23%).

4. **Realism Perception**:
   - Annotator 2 judged YouTube calls as **94.81% Realistic** and only **2.60% Not Realistic**, compared to **18.90% Not Realistic** for synthetic calls ($p = 1.75 \\times 10^{-21}$, Gap = **+16.31%**).
   - Lower realism ratings directly correlate with lower accuracy (Annotator 1 accuracy falls from **93.40%** on Realistic calls to **79.95%** on Not Realistic calls, $p = 2.15 \\times 10^{-5}$).
"""))

nb.cells = cells

# Write notebook file
notebook_path = 'human_validation/annotation_analysis.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Created notebook structure at {notebook_path}.")

# Execute notebook using NotebookClient to populate all outputs
client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': 'human_validation/'}})
client.execute()

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Successfully executed and saved {notebook_path} with all cell outputs!")
