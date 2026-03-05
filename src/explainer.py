"""
SmartContainer Risk Engine — Explainer Module
Generates plain English 1–2 line explanations per container.
Reasons are tied to measurable features, not vague text.
"""

MODULE_VERSION = "1.0.0"


def generate_explanation(row):
    """
    Generate a concise 1–2 line explanation for a container's risk score.
    Checks each signal in order of feature importance.

    Args:
        row: a single row (Series) from the featured DataFrame

    Returns:
        String with top 2 reasons joined by semicolon
    """
    reasons = []

    # 1. Weight discrepancy (strongest feature: ~37.6% importance)
    wdp = row.get('weight_diff_abs_pct', 0)
    if wdp > 30:
        reasons.append(
            f"Severe weight discrepancy: {wdp:.1f}% diff (declared vs measured)"
        )
    elif wdp > 15:
        reasons.append(
            f"Significant weight discrepancy of {wdp:.1f}%"
        )
    elif wdp > 5:
        reasons.append(
            f"Mild weight variance of {wdp:.1f}% detected"
        )

    # 2. Zero value / weight flags
    if row.get('is_zero_value', 0):
        reasons.append(
            "Declared value is zero — goods shipped with no declared worth"
        )
    if row.get('is_zero_weight', 0):
        reasons.append(
            "Declared weight is zero — data integrity issue"
        )

    # 3. Importer history (~22.5% importance)
    imp_rate = row.get('importer_critical_rate', 0)
    imp_count = row.get('importer_shipment_count', 0)
    is_new = row.get('is_new_importer', 0)
    if imp_rate > 0.15:
        reasons.append(
            f"Importer has {imp_rate*100:.0f}% historical Critical rate "
            f"({int(imp_count)} prior shipments)"
        )
    elif is_new:
        reasons.append(
            "First-time importer — no prior shipment history available"
        )

    # 4. Dwell time
    dwell = row.get('Dwell_Time_Hours', 0)
    port = row.get('Destination_Port', 'unknown')
    if dwell > 118:
        reasons.append(
            f"Extreme dwell time: {dwell:.0f} hrs (99th pct threshold: 118 hrs)"
        )
    elif dwell > 98:
        reasons.append(
            f"Above-normal dwell time: {dwell:.0f} hrs at {port}"
        )

    # 5. HS Code risk (~11.7% importance)
    hs_rate = row.get('hs_critical_rate', 0)
    hs_code = row.get('HS_Code', 'unknown')
    if hs_rate > 0.05:
        reasons.append(
            f"HS Code {hs_code} has elevated risk rate "
            f"({hs_rate*100:.1f}% Critical historically)"
        )

    # 6. Country risk
    country_rate = row.get('country_critical_rate', 0)
    country = row.get('Origin_Country', 'unknown')
    if country_rate > 0.05:
        reasons.append(
            f"Origin country {country} has elevated risk profile"
        )

    # 7. Off-hours declaration
    if row.get('is_off_hours', 0):
        hour = row.get('declaration_hour', 0)
        reasons.append(
            f"Declaration filed at {int(hour):02d}:00 (off-hours window)"
        )

    # 8. No issues found
    if not reasons:
        reasons.append(
            f"No anomalies detected — weight variance {wdp:.1f}%, "
            f"normal dwell time, established importer"
        )

    # Return top 2 reasons only (problem statement: 1–2 lines)
    return "; ".join(reasons[:2])


def generate_explanations(df):
    """
    Apply explanation generation to entire DataFrame.

    Args:
        df: DataFrame with all feature columns

    Returns:
        Series of explanation strings
    """
    print(f"[explainer v{MODULE_VERSION}] Generating explanations for {len(df):,} containers...")
    explanations = df.apply(generate_explanation, axis=1)

    # Print sample explanations
    print(f"  Sample explanations:")
    for i in range(min(3, len(explanations))):
        print(f"    [{i}] {explanations.iloc[i]}")

    return explanations