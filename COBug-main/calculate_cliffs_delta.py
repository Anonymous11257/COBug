import numpy as np
from scipy import stats

def cliff_delta(x, y):
    """
    Calculate Cliff's delta effect size.
    
    Cliff's delta = (# of times x_i > y_j - # of times x_i < y_j) / (n_x * n_y)
    
    Interpretation:
    |δ| < 0.147: negligible
    0.147 ≤ |δ| < 0.33: small
    0.33 ≤ |δ| < 0.474: medium
    |δ| ≥ 0.474: large
    """
    n_x = len(x)
    n_y = len(y)
    
    more = 0
    less = 0
    
    for xi in x:
        for yj in y:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1
    
    delta = (more - less) / (n_x * n_y)
    return delta

def interpret_cliff_delta(delta):
    """Interpret Cliff's delta magnitude."""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"

def wilcoxon_test(x, y):
    """Perform Wilcoxon signed-rank test for paired samples."""
    # Remove pairs where difference is zero
    diff = np.array(x) - np.array(y)
    non_zero_mask = diff != 0
    
    if np.sum(non_zero_mask) < 1:
        return np.nan, 1.0  # No non-zero differences
    
    try:
        stat, p_value = stats.wilcoxon(x, y, alternative='two-sided')
        return stat, p_value
    except ValueError:
        return np.nan, 1.0

# =============================================================================
# DATA EXTRACTION FROM LATEX TABLES
# =============================================================================

# Repository names (14 repositories)
repos = [
    "GaloisGirl_Coding",
    "IBM_example-health-apis",
    "Martinfx_Cobol",
    "abrignoli_COBSOFT",
    "bhbandam_AZ-Legacy-Engineering",
    "bmcsoftware_vscode-ispw",
    "debinix_openjensen",
    "gbeine_COBOLUnit",
    "lucasrmagalhaes_learning-COBOL",
    "neopragma_cobol-unit-test",
    "seanpm2001_SNU_2D_ProgrammingTools_IDE_COBOL",
    "thospfuller_rcoboldi",
    "ve3wwg_cobcurses",
    "z390development_z390"
]

# =============================================================================
# QWEN MAP VALUES
# =============================================================================
qwen_ir_map = [0.1612, 0.2071, 0.2123, 0.1377, 0.2296, 0.1428, 0.1502, 0.1422, 
               0.1406, 0.1950, 0.2606, 0.0000, 0.0671, 0.0953]
qwen_lr_map = [0.1573, 0.2050, 0.2122, 0.1345, 0.2190, 0.1427, 0.1506, 0.1424,
               0.1453, 0.1857, 0.2606, 0.0000, 0.0695, 0.0939]
qwen_rf_map = [0.1095, 0.1341, 0.1221, 0.1045, 0.1568, 0.0840, 0.1707, 0.1061,
               0.1038, 0.1629, 0.1207, 0.0000, 0.0576, 0.0664]

# =============================================================================
# CODELLAMA MAP VALUES
# =============================================================================
codellama_ir_map = [0.1955, 0.2204, 0.1991, 0.1187, 0.1982, 0.1674, 0.1815, 0.0488,
                    0.1540, 0.2235, 0.2241, 0.0000, 0.0710, 0.0728]
codellama_lr_map = [0.1877, 0.2282, 0.1993, 0.1194, 0.1904, 0.1566, 0.1819, 0.0476,
                    0.1537, 0.2130, 0.2241, 0.0000, 0.0699, 0.0712]
codellama_rf_map = [0.1302, 0.1628, 0.1723, 0.0852, 0.1737, 0.1875, 0.1504, 0.1380,
                    0.1385, 0.1492, 0.1206, 0.0000, 0.0480, 0.0822]

# =============================================================================
# DEEPSEEK MAP VALUES
# =============================================================================
deepseek_ir_map = [0.1783, 0.2447, 0.2028, 0.1088, 0.1712, 0.1648, 0.1573, 0.1071,
                   0.1413, 0.1785, 0.2992, 0.0000, 0.0802, 0.0916]
deepseek_lr_map = [0.1778, 0.2449, 0.2030, 0.1089, 0.1682, 0.1643, 0.1550, 0.1005,
                   0.1335, 0.1783, 0.2278, 0.0000, 0.0804, 0.0887]
deepseek_rf_map = [0.1391, 0.1253, 0.1120, 0.1197, 0.1739, 0.1445, 0.1116, 0.1193,
                   0.0890, 0.1921, 0.0819, 0.0000, 0.0395, 0.0678]

# =============================================================================
# CALCULATIONS
# =============================================================================

print("=" * 80)
print("CLIFF'S DELTA AND WILCOXON TEST CALCULATIONS")
print("=" * 80)

results = []

# -----------------------------------------------------------------------------
# WITHIN-LLM COMPARISONS (Model comparisons for each LLM setting)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SECTION 1: MODEL COMPARISONS WITHIN EACH LLM SETTING")
print("=" * 80)

for llm_name, ir_map, lr_map, rf_map in [
    ("Qwen", qwen_ir_map, qwen_lr_map, qwen_rf_map),
    ("CodeLLaMA", codellama_ir_map, codellama_lr_map, codellama_rf_map),
    ("DeepSeek", deepseek_ir_map, deepseek_lr_map, deepseek_rf_map)
]:
    print(f"\n--- {llm_name} ---")
    
    # IR vs IR+LR
    delta = cliff_delta(ir_map, lr_map)
    interp = interpret_cliff_delta(delta)
    stat, p = wilcoxon_test(ir_map, lr_map)
    print(f"IR vs IR+LR: δ = {delta:.4f} ({interp}), W = {stat}, p = {p:.4f}")
    results.append({
        'LLM': llm_name, 'Comparison': 'IR vs IR+LR',
        'Cliff_Delta': delta, 'Effect': interp,
        'Wilcoxon_Stat': stat, 'p_value': p
    })
    
    # IR vs IR+RF
    delta = cliff_delta(ir_map, rf_map)
    interp = interpret_cliff_delta(delta)
    stat, p = wilcoxon_test(ir_map, rf_map)
    print(f"IR vs IR+RF: δ = {delta:.4f} ({interp}), W = {stat}, p = {p:.4f}")
    results.append({
        'LLM': llm_name, 'Comparison': 'IR vs IR+RF',
        'Cliff_Delta': delta, 'Effect': interp,
        'Wilcoxon_Stat': stat, 'p_value': p
    })
    
    # IR+LR vs IR+RF
    delta = cliff_delta(lr_map, rf_map)
    interp = interpret_cliff_delta(delta)
    stat, p = wilcoxon_test(lr_map, rf_map)
    print(f"IR+LR vs IR+RF: δ = {delta:.4f} ({interp}), W = {stat}, p = {p:.4f}")
    results.append({
        'LLM': llm_name, 'Comparison': 'IR+LR vs IR+RF',
        'Cliff_Delta': delta, 'Effect': interp,
        'Wilcoxon_Stat': stat, 'p_value': p
    })

# -----------------------------------------------------------------------------
# CROSS-LLM COMPARISONS (Using IR MAP values)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SECTION 2: CROSS-LLM COMPARISONS (Using IR MAP values)")
print("=" * 80)

# Qwen IR vs CodeLLaMA IR
delta = cliff_delta(qwen_ir_map, codellama_ir_map)
interp = interpret_cliff_delta(delta)
stat, p = wilcoxon_test(qwen_ir_map, codellama_ir_map)
print(f"\nQwen vs CodeLLaMA (IR): δ = {delta:.4f} ({interp}), W = {stat}, p = {p:.4f}")
results.append({
    'LLM': 'Cross-LLM', 'Comparison': 'Qwen vs CodeLLaMA (IR)',
    'Cliff_Delta': delta, 'Effect': interp,
    'Wilcoxon_Stat': stat, 'p_value': p
})

# Qwen IR vs DeepSeek IR
delta = cliff_delta(qwen_ir_map, deepseek_ir_map)
interp = interpret_cliff_delta(delta)
stat, p = wilcoxon_test(qwen_ir_map, deepseek_ir_map)
print(f"Qwen vs DeepSeek (IR): δ = {delta:.4f} ({interp}), W = {stat}, p = {p:.4f}")
results.append({
    'LLM': 'Cross-LLM', 'Comparison': 'Qwen vs DeepSeek (IR)',
    'Cliff_Delta': delta, 'Effect': interp,
    'Wilcoxon_Stat': stat, 'p_value': p
})

# CodeLLaMA IR vs DeepSeek IR
delta = cliff_delta(codellama_ir_map, deepseek_ir_map)
interp = interpret_cliff_delta(delta)
stat, p = wilcoxon_test(codellama_ir_map, deepseek_ir_map)
print(f"CodeLLaMA vs DeepSeek (IR): δ = {delta:.4f} ({interp}), W = {stat}, p = {p:.4f}")
results.append({
    'LLM': 'Cross-LLM', 'Comparison': 'CodeLLaMA vs DeepSeek (IR)',
    'Cliff_Delta': delta, 'Effect': interp,
    'Wilcoxon_Stat': stat, 'p_value': p
})

# -----------------------------------------------------------------------------
# AGGREGATED COMPARISON ACROSS ALL LLMs
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SECTION 3: AGGREGATED MODEL COMPARISONS (Pooled across LLMs)")
print("=" * 80)

# Pool all IR, LR, RF values
all_ir = qwen_ir_map + codellama_ir_map + deepseek_ir_map
all_lr = qwen_lr_map + codellama_lr_map + deepseek_lr_map
all_rf = qwen_rf_map + codellama_rf_map + deepseek_rf_map

# IR vs IR+LR (Aggregated)
delta = cliff_delta(all_ir, all_lr)
interp = interpret_cliff_delta(delta)
stat, p = wilcoxon_test(all_ir, all_lr)
print(f"\nAggregated IR vs IR+LR: δ = {delta:.4f} ({interp}), W = {stat}, p = {p:.4f}")
results.append({
    'LLM': 'Aggregated', 'Comparison': 'IR vs IR+LR',
    'Cliff_Delta': delta, 'Effect': interp,
    'Wilcoxon_Stat': stat, 'p_value': p
})

# IR vs IR+RF (Aggregated)
delta = cliff_delta(all_ir, all_rf)
interp = interpret_cliff_delta(delta)
stat, p = wilcoxon_test(all_ir, all_rf)
print(f"Aggregated IR vs IR+RF: δ = {delta:.4f} ({interp}), W = {stat}, p = {p:.4f}")
results.append({
    'LLM': 'Aggregated', 'Comparison': 'IR vs IR+RF',
    'Cliff_Delta': delta, 'Effect': interp,
    'Wilcoxon_Stat': stat, 'p_value': p
})

# IR+LR vs IR+RF (Aggregated)
delta = cliff_delta(all_lr, all_rf)
interp = interpret_cliff_delta(delta)
stat, p = wilcoxon_test(all_lr, all_rf)
print(f"Aggregated IR+LR vs IR+RF: δ = {delta:.4f} ({interp}), W = {stat}, p = {p:.4f}")
results.append({
    'LLM': 'Aggregated', 'Comparison': 'IR+LR vs IR+RF',
    'Cliff_Delta': delta, 'Effect': interp,
    'Wilcoxon_Stat': stat, 'p_value': p
})

# =============================================================================
# GENERATE LATEX TABLES
# =============================================================================
print("\n" + "=" * 80)
print("LATEX TABLE OUTPUT")
print("=" * 80)

# Table 1: Per-LLM Model Comparisons
print("\n% TABLE 1: Per-LLM Model Comparisons")
print(r"""
\begin{table}[ht]
\centering
\caption{Wilcoxon signed-rank test and Cliff's delta effect size for model comparisons within each LLM setting.}
\label{tab:cliff_delta_per_llm}
\small
\begin{tabular}{llcccc}
\toprule
\textbf{LLM} & \textbf{Comparison} & \textbf{Wilcoxon $W$} & \textbf{$p$-value} & \textbf{Cliff's $\delta$} & \textbf{Effect Size} \\
\midrule""")

for llm in ["Qwen", "CodeLLaMA", "DeepSeek"]:
    llm_results = [r for r in results if r['LLM'] == llm]
    for i, r in enumerate(llm_results):
        llm_col = llm if i == 0 else ""
        w_str = f"{r['Wilcoxon_Stat']:.1f}" if not np.isnan(r['Wilcoxon_Stat']) else "N/A"
        sig = "*" if r['p_value'] < 0.05 else ""
        print(f"{llm_col} & {r['Comparison']} & {w_str} & {r['p_value']:.4f}{sig} & {r['Cliff_Delta']:.4f} & {r['Effect'].capitalize()} \\\\")
    if llm != "DeepSeek":
        print(r"\midrule")

print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item $^*$ Statistically significant at $\alpha = 0.05$. Cliff's $\delta$ interpretation: $|\delta| < 0.147$ = negligible, $0.147 \leq |\delta| < 0.33$ = small, $0.33 \leq |\delta| < 0.474$ = medium, $|\delta| \geq 0.474$ = large.
\end{tablenotes}
\end{table}
""")

# Table 2: Cross-LLM and Aggregated Comparisons
print("\n% TABLE 2: Cross-LLM and Aggregated Comparisons")
print(r"""
\begin{table}[ht]
\centering
\caption{Wilcoxon signed-rank test and Cliff's delta for cross-LLM and aggregated model comparisons.}
\label{tab:cliff_delta_cross_llm}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Comparison} & \textbf{Wilcoxon $W$} & \textbf{$p$-value} & \textbf{Cliff's $\delta$} & \textbf{Effect Size} \\
\midrule""")

# Cross-LLM comparisons
cross_llm = [r for r in results if r['LLM'] == 'Cross-LLM']
for r in cross_llm:
    w_str = f"{r['Wilcoxon_Stat']:.1f}" if not np.isnan(r['Wilcoxon_Stat']) else "N/A"
    sig = "*" if r['p_value'] < 0.05 else ""
    print(f"{r['Comparison']} & {w_str} & {r['p_value']:.4f}{sig} & {r['Cliff_Delta']:.4f} & {r['Effect'].capitalize()} \\\\")

print(r"\midrule")

# Aggregated comparisons
agg = [r for r in results if r['LLM'] == 'Aggregated']
for r in agg:
    w_str = f"{r['Wilcoxon_Stat']:.1f}" if not np.isnan(r['Wilcoxon_Stat']) else "N/A"
    sig = "*" if r['p_value'] < 0.05 else ""
    print(f"Aggregated {r['Comparison']} & {w_str} & {r['p_value']:.4f}{sig} & {r['Cliff_Delta']:.4f} & {r['Effect'].capitalize()} \\\\")

print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item $^*$ Statistically significant at $\alpha = 0.05$.
\end{tablenotes}
\end{table}
""")

# Table 3: Combined comprehensive table (similar to user's example)
print("\n% TABLE 3: Combined Comprehensive Table (Matching User's Requested Format)")
print(r"""
\begin{table*}[ht]
\centering
\caption{Wilcoxon signed-rank test results and Cliff's delta effect sizes for all comparisons.}
\label{tab:cliff_delta_comprehensive}
\small
\begin{tabular}{llcccc}
\toprule
\textbf{Category} & \textbf{Comparison} & \textbf{$p$-value} & \textbf{Cliff's $\delta$} & \textbf{Wilcoxon $W$} & \textbf{Effect Size} \\
\midrule
\multicolumn{6}{l}{\textit{Within Qwen Setting}} \\""")

for r in [r for r in results if r['LLM'] == 'Qwen']:
    w_str = f"{r['Wilcoxon_Stat']:.1f}" if not np.isnan(r['Wilcoxon_Stat']) else "N/A"
    sig = "$^*$" if r['p_value'] < 0.05 else ""
    print(f" & {r['Comparison']} & {r['p_value']:.4f}{sig} & {r['Cliff_Delta']:.2f} ({r['Effect']}) & {w_str} & {r['Effect'].capitalize()} \\\\")

print(r"\midrule")
print(r"\multicolumn{6}{l}{\textit{Within CodeLLaMA Setting}} \\")

for r in [r for r in results if r['LLM'] == 'CodeLLaMA']:
    w_str = f"{r['Wilcoxon_Stat']:.1f}" if not np.isnan(r['Wilcoxon_Stat']) else "N/A"
    sig = "$^*$" if r['p_value'] < 0.05 else ""
    print(f" & {r['Comparison']} & {r['p_value']:.4f}{sig} & {r['Cliff_Delta']:.2f} ({r['Effect']}) & {w_str} & {r['Effect'].capitalize()} \\\\")

print(r"\midrule")
print(r"\multicolumn{6}{l}{\textit{Within DeepSeek Setting}} \\")

for r in [r for r in results if r['LLM'] == 'DeepSeek']:
    w_str = f"{r['Wilcoxon_Stat']:.1f}" if not np.isnan(r['Wilcoxon_Stat']) else "N/A"
    sig = "$^*$" if r['p_value'] < 0.05 else ""
    print(f" & {r['Comparison']} & {r['p_value']:.4f}{sig} & {r['Cliff_Delta']:.2f} ({r['Effect']}) & {w_str} & {r['Effect'].capitalize()} \\\\")

print(r"\midrule")
print(r"\multicolumn{6}{l}{\textit{Cross-LLM Comparisons (IR model)}} \\")

for r in [r for r in results if r['LLM'] == 'Cross-LLM']:
    w_str = f"{r['Wilcoxon_Stat']:.1f}" if not np.isnan(r['Wilcoxon_Stat']) else "N/A"
    sig = "$^*$" if r['p_value'] < 0.05 else ""
    comp_short = r['Comparison'].replace(' (IR)', '')
    print(f" & {comp_short} & {r['p_value']:.4f}{sig} & {r['Cliff_Delta']:.2f} ({r['Effect']}) & {w_str} & {r['Effect'].capitalize()} \\\\")

print(r"\midrule")
print(r"\multicolumn{6}{l}{\textit{Aggregated Across All LLMs}} \\")

for r in [r for r in results if r['LLM'] == 'Aggregated']:
    w_str = f"{r['Wilcoxon_Stat']:.1f}" if not np.isnan(r['Wilcoxon_Stat']) else "N/A"
    sig = "$^*$" if r['p_value'] < 0.05 else ""
    print(f" & {r['Comparison']} & {r['p_value']:.4f}{sig} & {r['Cliff_Delta']:.2f} ({r['Effect']}) & {w_str} & {r['Effect'].capitalize()} \\\\")

print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item $^*$ Statistically significant at $\alpha = 0.05$. Effect size thresholds: $|\delta| < 0.147$ (negligible), $0.147 \leq |\delta| < 0.33$ (small), $0.33 \leq |\delta| < 0.474$ (medium), $|\delta| \geq 0.474$ (large).
\end{tablenotes}
\end{table*}
""")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\nMean MAP values by model across all LLMs:")
print(f"  IR:    {np.mean(all_ir):.4f} (std: {np.std(all_ir):.4f})")
print(f"  IR+LR: {np.mean(all_lr):.4f} (std: {np.std(all_lr):.4f})")
print(f"  IR+RF: {np.mean(all_rf):.4f} (std: {np.std(all_rf):.4f})")

print("\nMean MAP values by LLM (IR model only):")
print(f"  Qwen:      {np.mean(qwen_ir_map):.4f}")
print(f"  CodeLLaMA: {np.mean(codellama_ir_map):.4f}")
print(f"  DeepSeek:  {np.mean(deepseek_ir_map):.4f}")
