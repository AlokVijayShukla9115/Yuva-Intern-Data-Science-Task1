import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("📥 Executing Week 3 Statistical Pipeline...")
np.random.seed(42)

# 1. Group Data Generation (Simulating Real-World Business Context)
group_control = np.random.normal(loc=2500, scale=350, size=120)
group_target = np.random.normal(loc=2620, scale=380, size=120)

df_control = pd.DataFrame({'Metric_Value': group_control, 'Group': 'Control_Cohort'})
df_target = pd.DataFrame({'Metric_Value': group_target, 'Group': 'Target_Cohort'})
df_pipeline = pd.concat([df_control, df_target], ignore_index=True)

# 2. Parametric Assumptions Validation
print("\n📊 Statistical Validation Profile:")
shapiro_control = stats.shapiro(group_control)
shapiro_target = stats.shapiro(group_target)
levene_test = stats.levene(group_control, group_target)
print(f"• Shapiro-Wilk (Control): p = {shapiro_control.pvalue:.4f}")
print(f"• Shapiro-Wilk (Target): p = {shapiro_target.pvalue:.4f}")
print(f"• Levene Homoscedasticity: p = {levene_test.pvalue:.4f}")

# 3. Inferential T-Test Execution
t_stat, p_val = stats.ttest_ind(group_control, group_target, equal_var=True)
print(f"\n🎯 Core T-Test Output Matrix:")
print(f"• Empirical T-Statistic Vector: {t_stat:.4f}")
print(f"• Calculated Significance Value (p-value): {p_val:.4f}")

# 4. Automated Visualizations Compilation
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.histplot(data=df_pipeline, x='Metric_Value', hue='Group', kde=True, bins=25, palette='muted', alpha=0.6)
plt.axvline(group_control.mean(), color='blue', linestyle='--', linewidth=1.5, label=f'Control Mean ({group_control.mean():.1f})')
plt.axvline(group_target.mean(), color='orange', linestyle='--', linewidth=1.5, label=f'Target Mean ({group_target.mean():.1f})')
plt.title('Figure 1: Metric Density and Statistical Distribution Profiles', fontsize=12, fontweight='bold')
plt.xlabel('Metric Magnitude Unit Scale')
plt.ylabel('Observation Frequency Count')
plt.legend()
plt.savefig('statistical_hypothesis_distribution.png', bbox_inches='tight')
plt.close()
print("\n🎉 Visualizations successfully saved to directory.")
