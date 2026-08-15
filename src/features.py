import numpy as np
import pandas as pd
from src.config import (
    CATEGORICAL_COLS, PROFILE_NUM_COLS, BALANCE_COLS,
    TRANSACTION_TYPES, INFLOW_TYPES, OUTFLOW_TYPES,
    COUNTERPARTY_SUFFIX_MAP, MONTHS
)
from src.utils import get_logger

logger = get_logger()

def compute_linear_slope(y_matrix):
    """
    Computes linear slope across 6 months (x = [1, 2, 3, 4, 5, 6] representing m6 to m1).
    y_matrix shape: (N, 6) where column 0 is M6 (oldest) and column 5 is M1 (most recent).
    """
    x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    x_mean = 3.5
    x_var = 17.5
    
    y_mean = np.mean(y_matrix, axis=1, keepdims=True)
    weights = (x - x_mean).reshape(1, -1)
    slopes = np.sum((y_matrix - y_mean) * weights, axis=1) / x_var
    return slopes

def compute_shannon_entropy(matrix):
    """
    Computes Shannon entropy across transaction streams.
    High entropy = diversified spending; Low entropy = concentrated / distress spending.
    """
    row_sums = np.sum(matrix, axis=1, keepdims=True) + 1e-7
    p = matrix / row_sums
    p_safe = np.where(p > 0, p, 1.0)
    entropy = -np.sum(p * np.log(p_safe + 1e-9), axis=1)
    return entropy

def engineer_features(data_df):
    """
    Grand Master v4 Feature Engineering Pipeline (~450+ high-signal features).
    Extracts deep domain, multi-lag velocity, rolling statistics, group z-scores,
    activity bitmasks, and liquidity strain signals.
    """
    logger.info("Starting Grand Master v4 feature engineering pipeline...")
    df = data_df.copy()
    new_cols = {}
    
    # -------------------------------------------------------------
    # 1. Profile, Interactions & Group Peer Benchmarks
    # -------------------------------------------------------------
    logger.info("Computing profile, cross-interactions & peer benchmarks...")
    new_cols['arpu_per_age'] = df['arpu'] / (df['age'] + 1.0)
    new_cols['activity_per_age'] = df['x_90_d_activity_rate'] / (df['age'] + 1.0)
    new_cols['arpu_activity_interact'] = df['arpu'] * df['x_90_d_activity_rate']
    
    # Log transforms for skewed variables
    new_cols['log_arpu'] = np.log1p(np.maximum(0, df['arpu'].values))
    new_cols['log_age'] = np.log1p(df['age'].values)
    
    age_groups = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], labels=['under_25', '25_35', '35_50', 'over_50']).astype(str)
    new_cols['age_group'] = age_groups
    
    segment_earning = df['segment'].astype(str) + "_" + df['earning_pattern'].astype(str)
    region_smartphone = df['region'].astype(str) + "_" + df['smartphone'].astype(str)
    region_segment = df['region'].astype(str) + "_" + df['segment'].astype(str)
    tri_profile = df['segment'].astype(str) + "_" + df['region'].astype(str) + "_" + df['earning_pattern'].astype(str)
    
    new_cols['segment_earning'] = segment_earning
    new_cols['region_smartphone'] = region_smartphone
    new_cols['region_segment'] = region_segment
    new_cols['tri_profile'] = tri_profile
    
    # Frequency Encoding
    cat_df = pd.DataFrame({
        'age_group': age_groups,
        'segment_earning': segment_earning,
        'region_smartphone': region_smartphone,
        'region_segment': region_segment,
        'tri_profile': tri_profile
    })
    for c in CATEGORICAL_COLS:
        cat_df[c] = df[c]
        
    for c in cat_df.columns:
        freq = cat_df[c].value_counts(normalize=True).to_dict()
        new_cols[f'{c}_freq'] = cat_df[c].map(freq).astype(np.float32)

    # Peer relative group benchmarks & Z-scores
    for group_col in ['segment', 'region', 'earning_pattern', 'segment_earning']:
        col_to_group = cat_df[group_col] if group_col in cat_df else df[group_col]
        grp_arpu_mean = df.groupby(col_to_group)['arpu'].transform('mean')
        grp_arpu_std = df.groupby(col_to_group)['arpu'].transform('std').fillna(1.0)
        grp_act = df.groupby(col_to_group)['x_90_d_activity_rate'].transform('mean')
        grp_m1_bal_mean = df.groupby(col_to_group)['m1_daily_avg_bal'].transform('mean')
        grp_m1_bal_std = df.groupby(col_to_group)['m1_daily_avg_bal'].transform('std').fillna(1.0)
        
        new_cols[f'arpu_vs_{group_col}_mean'] = df['arpu'] / (grp_arpu_mean + 1.0)
        new_cols[f'arpu_{group_col}_zscore'] = (df['arpu'] - grp_arpu_mean) / (grp_arpu_std + 1e-4)
        new_cols[f'activity_vs_{group_col}_mean'] = df['x_90_d_activity_rate'] / (grp_act + 1e-4)
        new_cols[f'm1_bal_vs_{group_col}_mean'] = df['m1_daily_avg_bal'] / (grp_m1_bal_mean + 1.0)
        new_cols[f'm1_bal_{group_col}_zscore'] = (df['m1_daily_avg_bal'] - grp_m1_bal_mean) / (grp_m1_bal_std + 1e-4)
        new_cols[f'm1_bal_pct_in_{group_col}'] = df.groupby(col_to_group)['m1_daily_avg_bal'].rank(pct=True).astype(np.float32)

    # -------------------------------------------------------------
    # 2. Balance Trajectory, Multi-Month Lags, Volatility & Strain
    # -------------------------------------------------------------
    logger.info("Computing balance multi-lags, rolling stats & drawdowns...")
    # Matrix ordered M6 (col 0) to M1 (col 5)
    bal_matrix = df[BALANCE_COLS[::-1]].values.astype(np.float32)
    
    new_cols['bal_slope'] = compute_linear_slope(bal_matrix)
    new_cols['bal_mean'] = np.mean(bal_matrix, axis=1)
    new_cols['bal_std'] = np.std(bal_matrix, axis=1)
    new_cols['bal_min'] = np.min(bal_matrix, axis=1)
    new_cols['bal_max'] = np.max(bal_matrix, axis=1)
    new_cols['bal_cv'] = new_cols['bal_std'] / (new_cols['bal_mean'] + 1.0)
    
    # Balance Multi-Month Lags and Differences (M1-M2, M2-M3, M3-M4, M4-M5, M5-M6)
    for i in range(1, 6):
        curr_b = df[f'm{i}_daily_avg_bal']
        prev_b = df[f'm{i+1}_daily_avg_bal']
        new_cols[f'bal_diff_m{i}_m{i+1}'] = curr_b - prev_b
        new_cols[f'bal_pct_chg_m{i}_m{i+1}'] = (curr_b - prev_b) / (prev_b + 1.0)

    # Rolling window stats on balance
    new_cols['bal_mean_m1_m2'] = df[['m1_daily_avg_bal', 'm2_daily_avg_bal']].mean(axis=1)
    new_cols['bal_mean_m1_m3'] = df[['m1_daily_avg_bal', 'm2_daily_avg_bal', 'm3_daily_avg_bal']].mean(axis=1)
    new_cols['bal_mean_m4_m6'] = df[['m4_daily_avg_bal', 'm5_daily_avg_bal', 'm6_daily_avg_bal']].mean(axis=1)
    new_cols['bal_recent_vs_old_ratio'] = new_cols['bal_mean_m1_m3'] / (new_cols['bal_mean_m4_m6'] + 1.0)
    
    # Log transforms of balance
    new_cols['log_m1_bal'] = np.log1p(np.maximum(0, df['m1_daily_avg_bal'].values))
    new_cols['log_bal_mean'] = np.log1p(np.maximum(0, new_cols['bal_mean']))
    
    # Ratios and Differences
    new_cols['bal_m1_vs_m6_ratio'] = df['m1_daily_avg_bal'] / (df['m6_daily_avg_bal'] + 1.0)
    new_cols['bal_m1_vs_m6_diff'] = df['m1_daily_avg_bal'] - df['m6_daily_avg_bal']
    new_cols['bal_recent_vs_hist_ratio'] = (df['m1_daily_avg_bal'] + df['m2_daily_avg_bal']) / (2.0 * (new_cols['bal_mean_m4_m6'] + 1.0))
    
    # Acceleration / 2nd derivative: (M1 - M2) - (M2 - M3)
    new_cols['bal_acceleration'] = new_cols['bal_diff_m1_m2'] - new_cols['bal_diff_m2_m3']
    
    # Proximity to 6-month historical minimum balance (Rock-Bottom score)
    bal_range = new_cols['bal_max'] - new_cols['bal_min'] + 1e-4
    new_cols['bal_rock_bottom_idx'] = (df['m1_daily_avg_bal'] - new_cols['bal_min']) / bal_range
    new_cols['bal_is_all_time_low'] = (df['m1_daily_avg_bal'] <= (new_cols['bal_min'] + 1.0)).astype(np.float32)
    new_cols['bal_range_dispersion'] = (new_cols['bal_max'] - new_cols['bal_min']) / (new_cols['bal_mean'] + 1.0)
    
    # Consecutive month-over-month balance drops count (0 to 5)
    bal_drops = sum((df[f'm{i}_daily_avg_bal'] < df[f'm{i+1}_daily_avg_bal']).astype(int) for i in range(1, 6))
    new_cols['consecutive_bal_drops_count'] = bal_drops

    # -------------------------------------------------------------
    # 3. Monthly Inflows, Outflows, Imbalance & Liquidity Strain
    # -------------------------------------------------------------
    logger.info("Computing monthly inflows, outflows, strain ratios & cashflow dynamics...")
    m_inflows = []
    m_outflows = []
    m_nets = []
    m_deficit_flags = []
    m_vols = []
    
    for m in range(1, 7):
        m_str = f"m{m}"
        inflow_cols = [f"{m_str}_{t}_total_value" for t in INFLOW_TYPES if f"{m_str}_{t}_total_value" in df.columns]
        inflow = df[inflow_cols].sum(axis=1)
        new_cols[f'{m_str}_inflow_total'] = inflow
        m_inflows.append(inflow)
        
        outflow_cols = [f"{m_str}_{t}_total_value" for t in OUTFLOW_TYPES if f"{m_str}_{t}_total_value" in df.columns]
        outflow = df[outflow_cols].sum(axis=1)
        new_cols[f'{m_str}_outflow_total'] = outflow
        m_outflows.append(outflow)
        
        vol_cols = [f"{m_str}_{t}_volume" for t in TRANSACTION_TYPES if f"{m_str}_{t}_volume" in df.columns]
        m_vol = df[vol_cols].sum(axis=1)
        new_cols[f'{m_str}_total_vol'] = m_vol
        m_vols.append(m_vol)
        
        net = inflow - outflow
        new_cols[f'{m_str}_net_cashflow'] = net
        m_nets.append(net)
        
        # Deficit and Strain Ratios
        m_deficit_flags.append((net < 0).astype(int))
        new_cols[f'{m_str}_coverage_ratio'] = inflow / (outflow + 1.0)
        new_cols[f'{m_str}_outflow_to_bal'] = outflow / (df[f'{m_str}_daily_avg_bal'] + 1.0)
        new_cols[f'{m_str}_inflow_outflow_imbalance'] = (inflow - outflow) / (inflow + outflow + 1.0)

    # Multi-month inflow/outflow velocity differences
    for i in range(1, 6):
        new_cols[f'inflow_diff_m{i}_m{i+1}'] = m_inflows[i-1] - m_inflows[i]
        new_cols[f'outflow_diff_m{i}_m{i+1}'] = m_outflows[i-1] - m_outflows[i]
        new_cols[f'net_cashflow_diff_m{i}_m{i+1}'] = m_nets[i-1] - m_nets[i]

    # 6-Month Deficit Month Count
    new_cols['total_deficit_months_6m'] = np.sum(m_deficit_flags, axis=0)

    inflow_matrix = np.column_stack([m_inflows[5-i] for i in range(6)]).astype(np.float32)
    outflow_matrix = np.column_stack([m_outflows[5-i] for i in range(6)]).astype(np.float32)
    net_matrix = np.column_stack([m_nets[5-i] for i in range(6)]).astype(np.float32)
    vol_matrix = np.column_stack([m_vols[5-i] for i in range(6)]).astype(np.float32)
    
    new_cols['inflow_slope'] = compute_linear_slope(inflow_matrix)
    new_cols['outflow_slope'] = compute_linear_slope(outflow_matrix)
    new_cols['net_cashflow_slope'] = compute_linear_slope(net_matrix)
    new_cols['vol_slope_total'] = compute_linear_slope(vol_matrix)
    new_cols['net_cashflow_std'] = np.std(net_matrix, axis=1)
    
    # Flow Divergence: Inflow Slope minus Outflow Slope
    new_cols['flow_divergence_slope'] = new_cols['inflow_slope'] - new_cols['outflow_slope']
    
    # Cash Burn Rate & Liquidity Runway
    new_cols['cash_burn_rate_m1'] = new_cols['m1_outflow_total'] / (new_cols['bal_mean'] + 1.0)
    new_cols['liquidity_runway_months'] = df['m1_daily_avg_bal'] / (new_cols['m1_outflow_total'] + 1.0)
    
    new_cols['m1_outflow_to_bal_ratio'] = new_cols['m1_outflow_total'] / (df['m1_daily_avg_bal'] + 1.0)
    new_cols['m1_net_cashflow_vs_bal'] = new_cols['m1_net_cashflow'] / (df['m1_daily_avg_bal'] + 1.0)
    new_cols['m1_vs_m6_net_cashflow_diff'] = new_cols['m1_net_cashflow'] - new_cols['m6_net_cashflow']
    
    # Activity Freeze Ratio: M1 volume vs expected 30-day activity from 90-day rate
    expected_active_days = df['x_90_d_activity_rate'] * 30.0
    new_cols['m1_activity_freeze_ratio'] = new_cols['m1_total_vol'] / (expected_active_days + 1.0)
    
    # Inflow composition: Reliance on P2P gifts (received) vs Linked Bank vs Cash Deposit
    for m in [1, 2]:
        m_str = f"m{m}"
        new_cols[f'{m_str}_p2p_inflow_reliance'] = df[f'{m_str}_received_total_value'] / (new_cols[f'{m_str}_inflow_total'] + 1.0)
        new_cols[f'{m_str}_bank_inflow_reliance'] = df[f'{m_str}_transfer_from_bank_total_value'] / (new_cols[f'{m_str}_inflow_total'] + 1.0)

    # -------------------------------------------------------------
    # 4. Outflow Diversity & Entropy (Distress Spending Concentration)
    # -------------------------------------------------------------
    logger.info("Computing spending diversity & Shannon entropy...")
    m1_outflow_mat = df[[f"m1_{t}_total_value" for t in OUTFLOW_TYPES]].values.astype(np.float32)
    m6_outflow_mat = df[[f"m6_{t}_total_value" for t in OUTFLOW_TYPES]].values.astype(np.float32)
    new_cols['m1_outflow_entropy'] = compute_shannon_entropy(m1_outflow_mat)
    new_cols['m6_outflow_entropy'] = compute_shannon_entropy(m6_outflow_mat)
    new_cols['outflow_entropy_change'] = new_cols['m1_outflow_entropy'] - new_cols['m6_outflow_entropy']

    # -------------------------------------------------------------
    # 5. Transaction-Level Slopes, Ratios & Multi-Lag Velocity
    # -------------------------------------------------------------
    logger.info("Computing transaction-level multi-lags, recency & emergency spikes...")
    drop_to_zero_flags = []
    
    for t in TRANSACTION_TYPES:
        val_cols = [f"m{i}_{t}_total_value" for i in range(1, 7)]
        vol_cols = [f"m{i}_{t}_volume" for i in range(1, 7)]
        
        new_cols[f'{t}_total_val_6m'] = df[val_cols].sum(axis=1)
        new_cols[f'{t}_total_vol_6m'] = df[vol_cols].sum(axis=1)
        
        # Multi-month lags for each transaction type
        new_cols[f'{t}_val_diff_m1_m2'] = df[f'm1_{t}_total_value'] - df[f'm2_{t}_total_value']
        new_cols[f'{t}_vol_diff_m1_m2'] = df[f'm1_{t}_volume'] - df[f'm2_{t}_volume']
        new_cols[f'{t}_val_diff_m2_m3'] = df[f'm2_{t}_total_value'] - df[f'm3_{t}_total_value']
        
        new_cols[f'{t}_val_m1_vs_m6_ratio'] = df[f'm1_{t}_total_value'] / (df[f'm6_{t}_total_value'] + 1.0)
        new_cols[f'{t}_vol_m1_vs_m6_ratio'] = df[f'm1_{t}_volume'] / (df[f'm6_{t}_volume'] + 1.0)
        
        val_mat = df[val_cols[::-1]].values.astype(np.float32)
        vol_mat = df[vol_cols[::-1]].values.astype(np.float32)
        new_cols[f'{t}_val_slope'] = compute_linear_slope(val_mat)
        new_cols[f'{t}_vol_slope'] = compute_linear_slope(vol_mat)
        
        # Emergency Spike Ratio: Largest transaction in M1 vs total transaction value
        new_cols[f'{t}_m1_highest_to_total_val'] = df[f'm1_{t}_highest_amount'] / (df[f'm1_{t}_total_value'] + 1.0)
        
        # Sudden silence in M1 flag
        hist_vol_sum = df[vol_cols[1:]].sum(axis=1)
        flag = ((hist_vol_sum > 0) & (df[f'm1_{t}_volume'] == 0)).astype(np.float32)
        new_cols[f'{t}_drop_to_zero_m1'] = flag
        drop_to_zero_flags.append(flag)

    # 6-Month Channel Outflow Shares
    total_outflow_6m = sum(new_cols[f'{t}_total_val_6m'] for t in OUTFLOW_TYPES) + 1.0
    for t in OUTFLOW_TYPES:
        new_cols[f'{t}_share_of_outflows_6m'] = new_cols[f'{t}_total_val_6m'] / total_outflow_6m

    # Emergency Drawdown Proxy: High withdrawals + High bank transfers relative to M1 balance
    new_cols['emergency_cash_drawdown'] = (df['m1_withdraw_total_value'] + df['m1_transfer_from_bank_total_value']) / (df['m1_daily_avg_bal'] + 1.0)

    # -------------------------------------------------------------
    # 6. Counterparty Multi-Month Dynamics
    # -------------------------------------------------------------
    logger.info("Computing counterparty multi-month dynamics...")
    for t, c_suffix in COUNTERPARTY_SUFFIX_MAP.items():
        m1_cp = f"m1_{t}_{c_suffix}"
        m2_cp = f"m2_{t}_{c_suffix}"
        m6_cp = f"m6_{t}_{c_suffix}"
        m1_vol = f"m1_{t}_volume"
        
        if m1_cp in df.columns and m1_vol in df.columns:
            new_cols[f'{t}_cp_per_vol_m1'] = df[m1_cp] / (df[m1_vol] + 1.0)
        if m1_cp in df.columns and m2_cp in df.columns:
            new_cols[f'{t}_cp_diff_m1_m2'] = df[m1_cp] - df[m2_cp]
        if m1_cp in df.columns and m6_cp in df.columns:
            new_cols[f'{t}_cp_m1_vs_m6_ratio'] = df[m1_cp] / (df[m6_cp] + 1.0)

    # -------------------------------------------------------------
    # 7. Activity Bitmasks & Composite Stress Signal
    # -------------------------------------------------------------
    logger.info("Computing activity bitmasks & composite stress indexes...")
    new_cols['total_zero_activity_m1'] = np.sum(drop_to_zero_flags, axis=0)
    
    # 6-Month Active Sequence Bitmask (Bit i is 1 if total volume in M_i > 0)
    bitmask = np.zeros(len(df), dtype=np.int32)
    for i in range(1, 7):
        has_vol = (new_cols[f'm{i}_total_vol'] > 0).astype(np.int32)
        bitmask += has_vol * (2 ** (6 - i))
    new_cols['activity_6m_bitmask'] = bitmask
    
    # Composite Heuristic Stress Index
    stress_idx = (
        (-new_cols['bal_slope'] * 0.25) +
        (new_cols['total_deficit_months_6m'] * 0.3) +
        (new_cols['m1_outflow_to_bal_ratio'] * 0.2) +
        (new_cols['total_zero_activity_m1'] * 0.25)
    )
    new_cols['composite_stress_index'] = stress_idx
    
    # Combine new features dataframe with original dataframe
    new_features_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_features_df], axis=1)
    
    # Clean infs/NaNs if any were created during divisions
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    logger.info(f"Grand Master v4 Feature engineering completed! Total columns: {df.shape[1]}")
    return df
