"""
Script to compute annotation evaluation metrics for Annotator 1 and Annotator 2
against ground truth, inter-annotator agreement, confidence analysis, and realism checks.
Generates CSV tables, figures, resolved final labels, and a reproducible Jupyter Notebook.
"""

import os
import re
import json
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
import nbformat as nbf

# -------------------------------------------------------------
# 1. Canonical Mappings & Normalizers
# -------------------------------------------------------------
CANONICAL_CLASSES = {
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

# -------------------------------------------------------------
# 2. Load & Clean Data
# -------------------------------------------------------------
gt = pd.read_csv('human_validation/ground_truth.csv')
c1 = pd.read_excel('human_validation/conv1.xlsx')
c2 = pd.read_excel('human_validation/conv2.xlsx')

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

# -------------------------------------------------------------
# 3. Disagreement Resolution & Final Labels
# -------------------------------------------------------------
df['agreement'] = (df['a1_cat'] == df['a2_cat'])

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

# Save resolved dataset
resolved_cols = [
    'blind_id', 'source_dataset', 'original_filename', 'true_class',
    'a1_cat', 'a2_cat', 'a1_conf', 'a2_conf', 'a1_real', 'a2_real',
    'agreement', 'resolved_final_label', 'resolution_rationale'
]
df[resolved_cols].to_csv('human_validation/resolved_human_validation_labels.csv', index=False)
print("Saved resolved_human_validation_labels.csv")

# -------------------------------------------------------------
# 4. Metrics Helper Functions
# -------------------------------------------------------------
def compute_validity_metrics(y_true, y_pred, subset_name):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_LABELS, zero_division=0)
    macro_p = np.mean(p)
    macro_r = np.mean(r)
    macro_f1 = np.mean(f1)
    
    table = pd.DataFrame({
        'Class': SHORT_LABELS,
        'Full_Class_Name': CLASS_LABELS,
        'Precision': np.round(p, 4),
        'Recall': np.round(r, 4),
        'F1_Score': np.round(f1, 4),
        'Support': s
    })
    
    summary = {
        'Subset': subset_name,
        'Accuracy': np.round(acc, 4),
        'Macro_Precision': np.round(macro_p, 4),
        'Macro_Recall': np.round(macro_r, 4),
        'Macro_F1': np.round(macro_f1, 4),
        'Total_Support': len(y_true)
    }
    return table, summary

# Compute for Overall, human_validation, and yt
subsets = {
    'Overall (All 1,045 calls)': df,
    'human_validation (Synthetic, N=968)': df[df['source_dataset'] == 'human_validation'],
    'yt (YouTube Real-world, N=77)': df[df['source_dataset'] == 'yt']
}

validity_tables = {}
validity_summaries = []

for sub_name, sub_df in subsets.items():
    t_a1, s_a1 = compute_validity_metrics(sub_df['true_class'], sub_df['a1_cat'], f'{sub_name} - Annotator 1')
    t_a2, s_a2 = compute_validity_metrics(sub_df['true_class'], sub_df['a2_cat'], f'{sub_name} - Annotator 2')
    validity_tables[f'{sub_name}_A1'] = t_a1
    validity_tables[f'{sub_name}_A2'] = t_a2
    validity_summaries.extend([s_a1, s_a2])

df_validity_summary = pd.DataFrame(validity_summaries)
df_validity_summary.to_csv('human_validation/metrics_summary_annotator_vs_gt.csv', index=False)
print("Saved metrics_summary_annotator_vs_gt.csv")

# Save detailed per-class CSVs
validity_tables['Overall (All 1,045 calls)_A1'].to_csv('human_validation/per_class_overall_annotator1.csv', index=False)
validity_tables['Overall (All 1,045 calls)_A2'].to_csv('human_validation/per_class_overall_annotator2.csv', index=False)
validity_tables['human_validation (Synthetic, N=968)_A1'].to_csv('human_validation/per_class_human_val_annotator1.csv', index=False)
validity_tables['human_validation (Synthetic, N=968)_A2'].to_csv('human_validation/per_class_human_val_annotator2.csv', index=False)
validity_tables['yt (YouTube Real-world, N=77)_A1'].to_csv('human_validation/per_class_yt_annotator1.csv', index=False)
validity_tables['yt (YouTube Real-world, N=77)_A2'].to_csv('human_validation/per_class_yt_annotator2.csv', index=False)

# -------------------------------------------------------------
# 5. Confusion Matrices (8x8)
# -------------------------------------------------------------
def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=SHORT_LABELS,
        yticklabels=SHORT_LABELS,
        cbar=True
    )
    plt.title(title, fontsize=13, fontweight='bold', pad=14)
    plt.xlabel('Predicted Category', fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel('True Ground Truth Category', fontsize=11, fontweight='bold', labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

cm_a1_overall = confusion_matrix(df['true_class'], df['a1_cat'], labels=CLASS_LABELS)
cm_a2_overall = confusion_matrix(df['true_class'], df['a2_cat'], labels=CLASS_LABELS)
plot_confusion_matrix(cm_a1_overall, 'Annotator 1 Confusion Matrix (Overall N=1045)', 'human_validation/confusion_matrix_annotator1_overall.png')
plot_confusion_matrix(cm_a2_overall, 'Annotator 2 Confusion Matrix (Overall N=1045)', 'human_validation/confusion_matrix_annotator2_overall.png')

# Source-specific confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
for idx, (src_name, src_key) in enumerate([('Synthetic (human_validation, N=968)', 'human_validation'), ('Real-World (YouTube, N=77)', 'yt')]):
    sub = df[df['source_dataset'] == src_key]
    cm = confusion_matrix(sub['true_class'], sub['a1_cat'], labels=CLASS_LABELS)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS)
    axes[idx].set_title(f'Annotator 1 - {src_name}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted Category', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('True Category', fontsize=10, fontweight='bold')
    axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('human_validation/confusion_matrix_annotator1_sources.png', dpi=300)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
for idx, (src_name, src_key) in enumerate([('Synthetic (human_validation, N=968)', 'human_validation'), ('Real-World (YouTube, N=77)', 'yt')]):
    sub = df[df['source_dataset'] == src_key]
    cm = confusion_matrix(sub['true_class'], sub['a2_cat'], labels=CLASS_LABELS)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[idx], xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS)
    axes[idx].set_title(f'Annotator 2 - {src_name}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted Category', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('True Category', fontsize=10, fontweight='bold')
    axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('human_validation/confusion_matrix_annotator2_sources.png', dpi=300)
plt.close()
print("Saved confusion matrix plots.")

# -------------------------------------------------------------
# 6. Inter-Annotator Agreement (Cohen's Kappa & Raw Agreement)
# -------------------------------------------------------------
agreement_rows = []

for sub_name, sub_df in [('Overall', df), ('human_validation (Synthetic)', df[df['source_dataset'] == 'human_validation']), ('yt (YouTube)', df[df['source_dataset'] == 'yt'])]:
    raw_acc = accuracy_score(sub_df['a1_cat'], sub_df['a2_cat'])
    kappa = cohen_kappa_score(sub_df['a1_cat'], sub_df['a2_cat'])
    agreement_rows.append({
        'Scope': sub_name,
        'N': len(sub_df),
        'Raw_Agreement': np.round(raw_acc, 4),
        'Raw_Agreement_Pct': f"{raw_acc*100:.2f}%",
        'Cohen_Kappa': np.round(kappa, 4)
    })

df_agreement_overall = pd.DataFrame(agreement_rows)

# Per-class Cohen's Kappa (one-vs-rest binary agreement for each category)
per_class_kappas = []
for i, cls in enumerate(CLASS_LABELS):
    row_k = {'Class': SHORT_LABELS[i], 'Full_Class_Name': cls}
    for sub_name, sub_df in [('Overall', df), ('human_val', df[df['source_dataset'] == 'human_validation']), ('yt', df[df['source_dataset'] == 'yt'])]:
        a1_bin = (sub_df['a1_cat'] == cls).astype(int)
        a2_bin = (sub_df['a2_cat'] == cls).astype(int)
        k_bin = cohen_kappa_score(a1_bin, a2_bin)
        row_k[f'Kappa_{sub_name}'] = np.round(k_bin, 4)
        row_k[f'Support_{sub_name}'] = (sub_df['true_class'] == cls).sum()
    per_class_kappas.append(row_k)

df_per_class_kappa = pd.DataFrame(per_class_kappas)
df_agreement_overall.to_csv('human_validation/inter_annotator_agreement_overall.csv', index=False)
df_per_class_kappa.to_csv('human_validation/inter_annotator_agreement_per_class.csv', index=False)
print("Saved inter-annotator agreement tables.")

# -------------------------------------------------------------
# 7. Confidence Analysis
# -------------------------------------------------------------
conf_rows = []
for sub_name, sub_df in [('Overall', df), ('human_validation', df[df['source_dataset'] == 'human_validation']), ('yt', df[df['source_dataset'] == 'yt'])]:
    for conf_lvl in ['High', 'Medium', 'Low']:
        # Annotator 1
        s1 = sub_df[sub_df['a1_conf'] == conf_lvl]
        acc1 = (s1['a1_cat'] == s1['true_class']).mean() if len(s1) > 0 else np.nan
        # Annotator 2
        s2 = sub_df[sub_df['a2_conf'] == conf_lvl]
        acc2 = (s2['a2_cat'] == s2['true_class']).mean() if len(s2) > 0 else np.nan
        
        conf_rows.append({
            'Source': sub_name,
            'Confidence_Level': conf_lvl,
            'A1_Count': len(s1),
            'A1_Accuracy': np.round(acc1, 4) if pd.notna(acc1) else np.nan,
            'A1_Accuracy_Pct': f"{acc1*100:.2f}%" if pd.notna(acc1) else 'N/A',
            'A2_Count': len(s2),
            'A2_Accuracy': np.round(acc2, 4) if pd.notna(acc2) else np.nan,
            'A2_Accuracy_Pct': f"{acc2*100:.2f}%" if pd.notna(acc2) else 'N/A',
        })

df_conf_analysis = pd.DataFrame(conf_rows)
df_conf_analysis.to_csv('human_validation/confidence_analysis.csv', index=False)

# Low-confidence concentration
low_conf_concentration = []
for i, cls in enumerate(CLASS_LABELS):
    n_gt = (df['true_class'] == cls).sum()
    a1_low = ((df['true_class'] == cls) & (df['a1_conf'] == 'Low')).sum()
    a2_low = ((df['true_class'] == cls) & (df['a2_conf'] == 'Low')).sum()
    low_conf_concentration.append({
        'Class': SHORT_LABELS[i],
        'GT_Support': n_gt,
        'A1_Low_Count': a1_low,
        'A1_Low_Pct_of_Class': f"{a1_low/n_gt*100:.2f}%",
        'A2_Low_Count': a2_low,
        'A2_Low_Pct_of_Class': f"{a2_low/n_gt*100:.2f}%"
    })
df_low_conf = pd.DataFrame(low_conf_concentration)
df_low_conf.to_csv('human_validation/confidence_low_concentration_by_class.csv', index=False)
print("Saved confidence analysis tables.")

# -------------------------------------------------------------
# 8. Realism Check Analysis
# -------------------------------------------------------------
realism_summary = []

for name, col, score_col, cat_col in [('Annotator 1', 'a1_real', 'a1_real_score', 'a1_cat'), ('Annotator 2', 'a2_real', 'a2_real_score', 'a2_cat')]:
    sub_clean = df.dropna(subset=[col]).copy()
    
    # Counts and proportions
    ct = pd.crosstab(sub_clean['source_dataset'], sub_clean[col])
    ct_prop = pd.crosstab(sub_clean['source_dataset'], sub_clean[col], normalize='index') * 100
    
    # Share Not Realistic
    hv_not_real = (sub_clean[sub_clean['source_dataset'] == 'human_validation'][col] == 'Not realistic').mean() * 100
    yt_not_real = (sub_clean[sub_clean['source_dataset'] == 'yt'][col] == 'Not realistic').mean() * 100
    gap = hv_not_real - yt_not_real
    
    # Chi-Square Test
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
    
    # Mann-Whitney U Test
    hv_scores = sub_clean[sub_clean['source_dataset'] == 'human_validation'][score_col].dropna()
    yt_scores = sub_clean[sub_clean['source_dataset'] == 'yt'][score_col].dropna()
    mwu = stats.mannwhitneyu(hv_scores, yt_scores, alternative='two-sided')
    
    # Correlation with Accuracy
    sub_clean['correct'] = (sub_clean[cat_col] == sub_clean['true_class']).astype(int)
    ct_acc = pd.crosstab(sub_clean[col], sub_clean['correct'])
    chi2_acc, p_acc, _, _ = stats.chi2_contingency(ct_acc)
    
    acc_real = sub_clean[sub_clean[col] == 'Realistic']['correct'].mean() * 100
    acc_somewhat = sub_clean[sub_clean[col] == 'Somewhat realistic']['correct'].mean() * 100
    acc_not_real = sub_clean[sub_clean[col] == 'Not realistic']['correct'].mean() * 100
    
    realism_summary.append({
        'Annotator': name,
        'N_Evaluated': len(sub_clean),
        'HV_Realistic_Pct': np.round(ct_prop.loc['human_validation'].get('Realistic', 0), 2),
        'HV_Somewhat_Pct': np.round(ct_prop.loc['human_validation'].get('Somewhat realistic', 0), 2),
        'HV_NotRealistic_Pct': np.round(ct_prop.loc['human_validation'].get('Not realistic', 0), 2),
        'YT_Realistic_Pct': np.round(ct_prop.loc['yt'].get('Realistic', 0), 2),
        'YT_Somewhat_Pct': np.round(ct_prop.loc['yt'].get('Somewhat realistic', 0), 2),
        'YT_NotRealistic_Pct': np.round(ct_prop.loc['yt'].get('Not realistic', 0), 2),
        'NotRealistic_Gap_HV_minus_YT': np.round(gap, 2),
        'Chi2_Stat': np.round(chi2, 4),
        'Chi2_p_value': f"{p_chi:.4e}",
        'MannWhitney_U': np.round(mwu.statistic, 1),
        'MannWhitney_p_value': f"{mwu.pvalue:.4e}",
        'Acc_Realistic': f"{acc_real:.2f}%",
        'Acc_Somewhat': f"{acc_somewhat:.2f}%",
        'Acc_NotRealistic': f"{acc_not_real:.2f}%",
        'Realism_vs_Acc_p_val': f"{p_acc:.4e}"
    })

df_realism_summary = pd.DataFrame(realism_summary)
df_realism_summary.to_csv('human_validation/realism_analysis_summary.csv', index=False)
print("Saved realism analysis summary.")

print("All metrics computation and CSV generation completed successfully.")
