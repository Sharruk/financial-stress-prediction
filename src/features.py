import numpy as np
import pandas as pd
from src.config import (
    CATEGORICAL_COLS, PROFILE_NUM_COLS, BALANCE_COLS,
    TRANSACTION_TYPES, COUNTERPARTY_SUFFIX_MAP, MONTHS
)
from src.utils import get_logger

logger = get_logger()

def compute_linear_slope(y_matrix):
    """
    Computes linear slope across 6 months (x = [1, 2, 3, 4, 5, 6] representing m6 to m1)
    y_matrix is shape (N, 6) where col 0 is M6 and col 5 is M1.
    """
    x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    x_mean = 3.5
    x_var = 17.5
    
    y_mean = np.mean(y_matrix, axis=1, keepdims=True)
    weights = (x - x_mean).reshape(1, -1)
    slopes = np.sum((y_matrix - y_mean) * weights, axis=1) / x_var
    return slopes

def engineer_features(data_df):
    """
    Main feature engineering pipeline generating high-signal domain features.
    Builds new features cleanly to avoid fragmentation.
    """
    logger.info("Starting feature engineering pipeline...")
    df = data_df.copy()
    new_cols = {}
    
    # 1. Profile & Interaction Features
    logger.info("Computing profile & interaction features...")
    new_cols['arpu_per_age'] = df['arpu'] / (df['age'] + 1.0)
    age_groups = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], labels=['under_25', '25_35', '35_50', 'over_50']).astype(str)
    new_cols['age_group'] = age_groups
    
    segment_earning = df['segment'].astype(str) + "_" + df['earning_pattern'].astype(str)
    region_smartphone = df['region'].astype(str) + "_" + df['smartphone'].astype(str)
    new_cols['segment_earning'] = segment_earning
    new_cols['region_smartphone'] = region_smartphone
    
    # Frequency Encoding for categoricals
    cat_df = pd.DataFrame({'age_group': age_groups, 'segment_earning': segment_earning, 'region_smartphone': region_smartphone})
    for c in CATEGORICAL_COLS:
        cat_df[c] = df[c]
        
    for c in cat_df.columns:
        freq = cat_df[c].value_counts(normalize=True).to_dict()
        new_cols[f'{c}_freq'] = cat_df[c].map(freq).astype(np.float32)

    # 2. Balance Features (M1 = most recent, M6 = oldest)
    logger.info("Computing balance trend & volatility features...")
    bal_matrix = df[BALANCE_COLS[::-1]].values.astype(np.float32) # Order from M6 (col 0) to M1 (col 5)
    
    new_cols['bal_slope'] = compute_linear_slope(bal_matrix)
    new_cols['bal_mean'] = np.mean(bal_matrix, axis=1)
    new_cols['bal_std'] = np.std(bal_matrix, axis=1)
    new_cols['bal_min'] = np.min(bal_matrix, axis=1)
    new_cols['bal_max'] = np.max(bal_matrix, axis=1)
    new_cols['bal_m1_vs_m6_ratio'] = df['m1_daily_avg_bal'] / (df['m6_daily_avg_bal'] + 1.0)
    new_cols['bal_m1_vs_m6_diff'] = df['m1_daily_avg_bal'] - df['m6_daily_avg_bal']
    new_cols['bal_recent_vs_hist_ratio'] = (df['m1_daily_avg_bal'] + df['m2_daily_avg_bal']) / (df[['m3_daily_avg_bal', 'm4_daily_avg_bal', 'm5_daily_avg_bal', 'm6_daily_avg_bal']].mean(axis=1) + 1.0)
    new_cols['bal_cv'] = new_cols['bal_std'] / (new_cols['bal_mean'] + 1.0)

    # 3. Monthly Inflows and Outflows
    logger.info("Computing monthly inflows, outflows & net cash flows...")
    inflow_types = ["deposit", "received", "transfer_from_bank"]
    outflow_types = ["withdraw", "paybill", "merchantpay", "mm_send"]
    
    m_inflows = []
    m_outflows = []
    m_nets = []
    
    for m in range(1, 7):
        m_str = f"m{m}"
        inflow_cols = [f"{m_str}_{t}_total_value" for t in inflow_types if f"{m_str}_{t}_total_value" in df.columns]
        inflow = df[inflow_cols].sum(axis=1)
        new_cols[f'{m_str}_inflow_total'] = inflow
        m_inflows.append(inflow)
        
        outflow_cols = [f"{m_str}_{t}_total_value" for t in outflow_types if f"{m_str}_{t}_total_value" in df.columns]
        outflow = df[outflow_cols].sum(axis=1)
        new_cols[f'{m_str}_outflow_total'] = outflow
        m_outflows.append(outflow)
        
        net = inflow - outflow
        new_cols[f'{m_str}_net_cashflow'] = net
        m_nets.append(net)
        
        new_cols[f'{m_str}_coverage_ratio'] = inflow / (outflow + 1.0)

    inflow_matrix = np.column_stack([m_inflows[5-i] for i in range(6)]).astype(np.float32)
    outflow_matrix = np.column_stack([m_outflows[5-i] for i in range(6)]).astype(np.float32)
    net_matrix = np.column_stack([m_nets[5-i] for i in range(6)]).astype(np.float32)
    
    new_cols['inflow_slope'] = compute_linear_slope(inflow_matrix)
    new_cols['outflow_slope'] = compute_linear_slope(outflow_matrix)
    new_cols['net_cashflow_slope'] = compute_linear_slope(net_matrix)
    
    new_cols['m1_outflow_to_bal_ratio'] = new_cols['m1_outflow_total'] / (df['m1_daily_avg_bal'] + 1.0)
    new_cols['m1_net_cashflow_vs_bal'] = new_cols['m1_net_cashflow'] / (df['m1_daily_avg_bal'] + 1.0)

    # 4. Aggregated Transaction Type Dynamics
    logger.info("Computing transaction-level recency, slopes & ratios...")
    drop_to_zero_flags = []
    
    for t in TRANSACTION_TYPES:
        val_cols = [f"m{i}_{t}_total_value" for i in range(1, 7)]
        vol_cols = [f"m{i}_{t}_volume" for i in range(1, 7)]
        
        new_cols[f'{t}_total_val_6m'] = df[val_cols].sum(axis=1)
        new_cols[f'{t}_total_vol_6m'] = df[vol_cols].sum(axis=1)
        
        new_cols[f'{t}_val_m1_vs_m6_ratio'] = df[f'm1_{t}_total_value'] / (df[f'm6_{t}_total_value'] + 1.0)
        new_cols[f'{t}_vol_m1_vs_m6_ratio'] = df[f'm1_{t}_volume'] / (df[f'm6_{t}_volume'] + 1.0)
        
        val_mat = df[val_cols[::-1]].values.astype(np.float32)
        vol_mat = df[vol_cols[::-1]].values.astype(np.float32)
        new_cols[f'{t}_val_slope'] = compute_linear_slope(val_mat)
        new_cols[f'{t}_vol_slope'] = compute_linear_slope(vol_mat)
        
        hist_vol_sum = df[vol_cols[1:]].sum(axis=1)
        flag = ((hist_vol_sum > 0) & (df[f'm1_{t}_volume'] == 0)).astype(np.float32)
        new_cols[f'{t}_drop_to_zero_m1'] = flag
        drop_to_zero_flags.append(flag)

    # 5. Counterparty Diversity & Intensity
    logger.info("Computing counterparty intensity features...")
    for t, c_suffix in COUNTERPARTY_SUFFIX_MAP.items():
        m1_cp = f"m1_{t}_{c_suffix}"
        m1_vol = f"m1_{t}_volume"
        if m1_cp in df.columns and m1_vol in df.columns:
            new_cols[f'{t}_cp_per_vol_m1'] = df[m1_cp] / (df[m1_vol] + 1.0)
            
        m6_cp = f"m6_{t}_{c_suffix}"
        if m1_cp in df.columns and m6_cp in df.columns:
            new_cols[f'{t}_cp_m1_vs_m6_ratio'] = df[m1_cp] / (df[m6_cp] + 1.0)

    # 6. Overall Activity Summary
    logger.info("Computing overall activity summary features...")
    new_cols['total_zero_activity_m1'] = np.sum(drop_to_zero_flags, axis=0)
    
    # Combine new features dataframe with original dataframe
    new_features_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_features_df], axis=1)
    
    # Clean infs/NaNs if any were created during divisions
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    logger.info(f"Feature engineering completed! Total columns: {df.shape[1]}")
    return df
