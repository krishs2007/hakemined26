"""
SmartContainer Risk Engine — Dashboard Module
Generates a 6-panel matplotlib dashboard and saves as PNG.
"""
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

MODULE_VERSION = "1.0.0"
OUTPUT_DIR = 'outputs'
DASHBOARD_PATH = os.path.join(OUTPUT_DIR, 'summary_report.png')

# Color palette
COLORS = {
    'Critical': '#e74c3c',
    'Medium Risk': '#f39c12',
    'Low Risk': '#2ecc71',
}

def generate_dashboard(results_df):
    """
    Generate a 6-panel dashboard from scored results.

    Panels:
        1. Risk Level Donut Chart
        2. Risk Score Histogram
        3. Top 10 Riskiest Containers
        4. Avg Risk by Origin Country (Top 15)
        5. Avg Risk by Destination Port (Top 10)
        6. Summary Statistics Box

    Args:
        results_df: DataFrame with Risk_Score, Risk_Level, Origin_Country,
                    Destination_Port, Container_ID columns
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[dashboard v{MODULE_VERSION}] Generating 6-panel dashboard...")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('SmartContainer Risk Engine — Dashboard', fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    # --- Panel 1: Risk Level Donut ---
    ax1 = fig.add_subplot(gs[0, 0])
    counts = results_df['Risk_Level'].value_counts()
    order = ['Critical', 'Medium Risk', 'Low Risk']
    ordered_counts = [counts.get(level, 0) for level in order]
    ordered_colors = [COLORS[level] for level in order]
    wedges, texts, autotexts = ax1.pie(
        ordered_counts, labels=order, autopct='%1.1f%%',
        colors=ordered_colors, startangle=90,
        wedgeprops=dict(width=0.4)
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
    ax1.set_title(f'Risk Distribution\n(Total: {len(results_df):,} containers)', fontsize=11)

    # --- Panel 2: Risk Score Histogram ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(results_df['Risk_Score'], bins=50, color='steelblue', edgecolor='white', alpha=0.85)
    ax2.axvline(31, color='orange', linestyle='--', linewidth=1.5, label='Medium Risk threshold (31)')
    ax2.axvline(61, color='red', linestyle='--', linewidth=1.5, label='Critical threshold (61)')
    ax2.set_xlabel('Risk Score (0-100)')
    ax2.set_ylabel('Count')
    ax2.set_title('Risk Score Distribution', fontsize=11)
    ax2.legend(fontsize=8)

    # --- Panel 3: Top 10 Riskiest Containers ---
    ax3 = fig.add_subplot(gs[0, 2])
    top10 = results_df.nlargest(10, 'Risk_Score')[['Container_ID', 'Risk_Score']].copy()
    top10 = top10.sort_values('Risk_Score', ascending=True)
    bar_colors = [COLORS.get(
        'Critical' if s >= 61 else ('Medium Risk' if s >= 31 else 'Low Risk'),
        '#999999'
    ) for s in top10['Risk_Score']]
    ax3.barh(top10['Container_ID'].astype(str), top10['Risk_Score'], color=bar_colors)
    ax3.set_xlabel('Risk Score')
    ax3.set_title('Top 10 Highest Risk Containers', fontsize=11)
    ax3.set_xlim(0, 105)
    for i, (cid, score) in enumerate(zip(top10['Container_ID'], top10['Risk_Score'])):
        ax3.text(score + 1, i, f'{score:.1f}', va='center', fontsize=8)

    # --- Panel 4: Risk by Origin Country (Top 15) ---
    ax4 = fig.add_subplot(gs[1, 0])
    country_risk = results_df.groupby('Origin_Country')['Risk_Score'].mean().nlargest(15)
    ax4.bar(range(len(country_risk)), country_risk.values, color='coral', edgecolor='white')
    ax4.set_xticks(range(len(country_risk)))
    ax4.set_xticklabels(country_risk.index, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Avg Risk Score')
    ax4.set_title('Avg Risk Score by Origin Country (Top 15)', fontsize=11)

    # --- Panel 5: Risk by Destination Port (Top 10) ---
    ax5 = fig.add_subplot(gs[1, 1])
    port_risk = results_df.groupby('Destination_Port')['Risk_Score'].mean().nlargest(10)
    ax5.bar(range(len(port_risk)), port_risk.values, color='mediumpurple', edgecolor='white')
    ax5.set_xticks(range(len(port_risk)))
    ax5.set_xticklabels(port_risk.index, rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('Avg Risk Score')
    ax5.set_title('Avg Risk Score by Port (Top 10)', fontsize=11)

    # --- Panel 6: Summary Stats Box ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    critical_count = (results_df['Risk_Level'] == 'Critical').sum()
    medium_count = (results_df['Risk_Level'] == 'Medium Risk').sum()
    low_count = (results_df['Risk_Level'] == 'Low Risk').sum()
    summary_text = (
        f"SUMMARY REPORT\n"
        f"{'=' * 36}\n\n"
        f"Total Processed:    {len(results_df):>8,}\n\n"
        f"Critical:           {critical_count:>8,}  ({critical_count/len(results_df)*100:5.1f}%)\n"
        f"Medium Risk:        {medium_count:>8,}  ({medium_count/len(results_df)*100:5.1f}%)\n"
        f"Low Risk:           {low_count:>8,}  ({low_count/len(results_df)*100:5.1f}%)\n\n"
        f"{'=' * 36}\n\n"
        f"Avg Risk Score:     {results_df['Risk_Score'].mean():>8.1f}\n"
        f"Median Risk Score:  {results_df['Risk_Score'].median():>8.1f}\n"
        f"Max Risk Score:     {results_df['Risk_Score'].max():>8.1f}\n"
        f"Min Risk Score:     {results_df['Risk_Score'].min():>8.1f}\n"
    )
    ax6.text(
        0.05, 0.50, summary_text, transform=ax6.transAxes,
        fontsize=11, verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.9, edgecolor='#cccccc')
    )

    plt.savefig(DASHBOARD_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Dashboard saved to: {DASHBOARD_PATH}")

    return DASHBOARD_PATH
