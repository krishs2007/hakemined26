"""
SmartContainer Risk Engine — Dashboard Module
Generates a 6-panel matplotlib dashboard (PNG) and an HTML summary report.
"""
import base64
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

MODULE_VERSION = "1.0.0"
OUTPUT_DIR = 'outputs'
DASHBOARD_PATH = os.path.join(OUTPUT_DIR, 'summary_report.png')
HTML_PATH = os.path.join(OUTPUT_DIR, 'summary_report.html')

# Color palette
COLORS = {
    'Critical': '#e74c3c',
    'Medium Risk': '#f39c12',
    'Low Risk': '#2ecc71',
}

# HTML row styles for risk levels (reused in summary and top-N tables)
HTML_ROW_STYLES = {
    'Critical': 'background-color:#ffcccc; color:#cc0000; font-weight:bold',
    'Medium Risk': 'background-color:#fff3cd; color:#856404; font-weight:bold',
    'Low Risk': 'background-color:#d4edda; color:#155724; font-weight:bold',
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

    html_path = _generate_html_report(results_df, DASHBOARD_PATH)

    return DASHBOARD_PATH, html_path


def _generate_html_report(results_df, png_path):
    """
    Generate an HTML summary report with an embedded dashboard image,
    summary statistics table, and top 20 riskiest containers table.

    Args:
        results_df: DataFrame with Risk_Score, Risk_Level, Container_ID columns
        png_path:   Path to the saved PNG dashboard image

    Returns:
        Path to the saved HTML report
    """
    # Embed PNG as base64 so the HTML is self-contained
    with open(png_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    total = len(results_df)
    critical_count = int((results_df['Risk_Level'] == 'Critical').sum())
    medium_count = int((results_df['Risk_Level'] == 'Medium Risk').sum())
    low_count = int((results_df['Risk_Level'] == 'Low Risk').sum())
    avg_score = results_df['Risk_Score'].mean()
    median_score = results_df['Risk_Score'].median()
    max_score = results_df['Risk_Score'].max()
    min_score = results_df['Risk_Score'].min()

    def pct(n):
        return f"{n / total * 100:.1f}%" if total > 0 else "0.0%"

    # Top 20 riskiest containers
    top20 = results_df.nlargest(20, 'Risk_Score')[['Container_ID', 'Risk_Score', 'Risk_Level']].copy()

    top20_rows = ''.join(
        f'<tr style="{HTML_ROW_STYLES.get(row["Risk_Level"], "")}">'
        f'<td>{row["Container_ID"]}</td>'
        f'<td>{row["Risk_Score"]:.1f}</td>'
        f'<td>{row["Risk_Level"]}</td>'
        f'</tr>'
        for _, row in top20.iterrows()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SmartContainer Risk Engine - Summary Report</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      background: #f8f9fa;
      color: #212529;
      margin: 0;
      padding: 20px;
    }}
    h1 {{
      color: #1a1a2e;
      border-bottom: 3px solid #e74c3c;
      padding-bottom: 10px;
    }}
    h2 {{
      color: #333;
      margin-top: 30px;
    }}
    .dashboard-img {{
      max-width: 100%;
      border: 1px solid #ddd;
      border-radius: 6px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      max-width: 800px;
      margin-top: 12px;
      background: #fff;
      border-radius: 6px;
      overflow: hidden;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    th {{
      background: #1a1a2e;
      color: #fff;
      padding: 10px 14px;
      text-align: left;
    }}
    td {{
      padding: 9px 14px;
      border-bottom: 1px solid #eee;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .badge-critical  {{ background:#e74c3c; color:#fff; padding:2px 8px; border-radius:4px; }}
    .badge-medium    {{ background:#f39c12; color:#fff; padding:2px 8px; border-radius:4px; }}
    .badge-low       {{ background:#2ecc71; color:#fff; padding:2px 8px; border-radius:4px; }}
    footer {{
      margin-top: 40px;
      color: #888;
      font-size: 0.85em;
      border-top: 1px solid #ddd;
      padding-top: 10px;
    }}
  </style>
</head>
<body>
  <h1>🚢 SmartContainer Risk Engine - Summary Report</h1>

  <h2>Dashboard</h2>
  <img class="dashboard-img" src="data:image/png;base64,{img_b64}" alt="SmartContainer Risk Dashboard"/>

  <h2>Summary Statistics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total Containers Processed</td><td><strong>{total:,}</strong></td></tr>
    <tr style="{HTML_ROW_STYLES['Critical']}">
      <td><strong>Critical</strong></td><td>{critical_count:,} &nbsp;({pct(critical_count)})</td>
    </tr>
    <tr style="{HTML_ROW_STYLES['Medium Risk']}">
      <td><strong>Medium Risk</strong></td><td>{medium_count:,} &nbsp;({pct(medium_count)})</td>
    </tr>
    <tr style="{HTML_ROW_STYLES['Low Risk']}">
      <td><strong>Low Risk</strong></td><td>{low_count:,} &nbsp;({pct(low_count)})</td>
    </tr>
    <tr><td>Average Risk Score</td><td>{avg_score:.1f}</td></tr>
    <tr><td>Median Risk Score</td><td>{median_score:.1f}</td></tr>
    <tr><td>Max Risk Score</td><td>{max_score:.1f}</td></tr>
    <tr><td>Min Risk Score</td><td>{min_score:.1f}</td></tr>
  </table>

  <h2>Top 20 Riskiest Containers</h2>
  <table>
    <tr><th>Container ID</th><th>Risk Score</th><th>Risk Level</th></tr>
    {top20_rows}
  </table>

  <footer>
    Generated by SmartContainer Risk Engine v{MODULE_VERSION}
  </footer>
</body>
</html>
"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  HTML report saved to: {HTML_PATH}")

    return HTML_PATH
