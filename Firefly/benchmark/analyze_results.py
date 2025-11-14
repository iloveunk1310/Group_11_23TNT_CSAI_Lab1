"""
Comprehensive benchmark results analysis with unified data loading architecture.
Supports multi-level analysis for both Rastrigin and Knapsack problems.
Enhanced with COCO/BBOB runtime-centric analysis, performance profiles, and diversity metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import re
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# UNIFIED DATA LOADING - CORE FUNCTION (ENHANCED)
# ============================================================================

def load_all_results_to_dataframe(results_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Universal data loader: scan all JSON and JSON.GZ files and create unified DataFrame.
    
    Enhanced to extract:
    - Evaluation axis (pop_size for FA/GA, 1 for SA/HC)
    - Stats_History with diversity metrics for Rastrigin
    - Success_Levels and Thresholds_Used for fixed-target analysis
    
    Supports both .json and .json.gz files.
    
    Returns
    -------
    pd.DataFrame
        Unified dataframe with columns:
        - Problem: 'rastrigin' or 'knapsack'
        - Scenario, Algorithm, Configuration, etc.
        - PopSize, Dim (for Rastrigin), Eval_Axis
        - Stats_History (dict), Success_Levels (dict), Thresholds_Used (dict)
        - History (list of best values per generation)
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return pd.DataFrame()
    
    all_runs = []
    
    # Scan all JSON and JSON.GZ files
    json_files = list(results_path.rglob('*.json'))
    json_gz_files = list(results_path.rglob('*.json.gz'))
    all_files = json_files + json_gz_files
    
    logger.info(f"Found {len(json_files)} JSON files and {len(json_gz_files)} JSON.GZ files in {results_dir}")
    
    for file_path in all_files:
        filename = file_path.name
        is_compressed = filename.endswith('.json.gz')
        
        # Remove .gz extension for pattern matching
        filename_for_match = filename.replace('.json.gz', '.json') if is_compressed else filename
        
        # Try Rastrigin pattern
        match_rast = re.match(
            r'rastrigin_([a-z_]+)_([A-Z]{2})_([a-z_]+)_(\d{8}T\d{6})\.json',
            filename_for_match
        )
        
        # Try Knapsack pattern
        match_knap = re.match(
            r'knapsack_n(\d+)_([a-z_]+)_seed(\d+)_([A-Z]{2})_([a-z_]+)_(\d{8}T\d{6})\.json',
            filename_for_match
        )
        
        # Load JSON safely (handle both compressed and uncompressed)
        try:
            if is_compressed:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}")
            continue
        
        # Extract metadata and results
        if 'metadata' in data and 'all_results' in data:
            metadata = data['metadata']
            runs = data['all_results']
        elif 'metadata' in data and 'results' in data:
            metadata = data['metadata']
            runs = data['results']
        elif 'results' in data:
            metadata = {}
            runs = data['results']
        else:
            logger.warning(f"Unknown structure in {filename}")
            continue
        
        # Process based on problem type
        if match_rast:
            problem = 'rastrigin'
            config_name = match_rast.group(1)
            algorithm = match_rast.group(2)
            scenario = match_rast.group(3)
            timestamp = match_rast.group(4)
            
            # Extract dimension and budget from metadata
            dim = metadata.get('dimension', metadata.get('dim'))
            budget = metadata.get('budget')
            pop_size = metadata.get('pop_size')
            
            # Determine eval_axis: pop_size for FA/GA, 1 for SA/HC
            if algorithm in ['FA', 'GA']:
                eval_axis = pop_size if pop_size else 1
            else:  # SA, HC
                eval_axis = 1
            
            # Process each run
            for run in runs:
                flat_run = {
                    'Problem': problem,
                    'Scenario': scenario,
                    'Algorithm': algorithm,
                    'Configuration': config_name,
                    'Algo_Seed': run.get('algo_seed') or run.get('seed'),
                    'Problem_Seed': run.get('problem_seed'),
                    'Timestamp': timestamp,
                    
                    # Problem parameters
                    'Dim': dim,
                    'Budget': budget,
                    'PopSize': pop_size,
                    'Eval_Axis': eval_axis,
                    
                    # Core metrics
                    'Best_Fitness': run.get('best_fitness'),
                    'Elapsed_Time': run.get('elapsed_time'),
                    'Evaluations': run.get('evaluations'),
                    'Budget_Utilization': run.get('budget_utilization'),
                    'Status': run.get('status', 'ok'),
                    
                    # History and stats (keep as objects to save memory)
                    'History': run.get('history'),
                    'Stats_History': run.get('stats_history'),
                    'Success_Levels': run.get('success_levels'),
                    'Thresholds_Used': run.get('thresholds_used'),
                }
                
                # Flatten success_levels for quick access
                if 'success_levels' in run and run['success_levels']:
                    for level, level_data in run['success_levels'].items():
                        flat_run[f'Success_{level.capitalize()}'] = level_data.get('success', False)
                        flat_run[f'HitEvals_{level.capitalize()}'] = level_data.get('hit_evaluations')
                        flat_run[f'Threshold_{level.capitalize()}'] = level_data.get('threshold')
                
                all_runs.append(flat_run)
        
        elif match_knap:
            problem = 'knapsack'
            n_items = int(match_knap.group(1))
            instance_type = match_knap.group(2)
            instance_seed = int(match_knap.group(3))
            algorithm = match_knap.group(4)
            scenario = match_knap.group(5)
            timestamp = match_knap.group(6)
            
            dp_optimal = metadata.get('dp_optimal')
            has_dp = metadata.get('has_dp_optimal', dp_optimal is not None)
            budget = metadata.get('budget')
            pop_size = metadata.get('pop_size')
            
            # Determine eval_axis
            if algorithm in ['FA', 'GA']:
                eval_axis = pop_size if pop_size else 1
            else:
                eval_axis = 1
            
            # Process each run
            for run in runs:
                flat_run = {
                    'Problem': problem,
                    'Scenario': scenario,
                    'Algorithm': algorithm,
                    'N_Items': n_items,
                    'Instance_Type': instance_type,
                    'Instance_Seed': instance_seed,
                    'Algo_Seed': run.get('algo_seed') or run.get('seed'),
                    'Timestamp': timestamp,
                    
                    # Problem parameters
                    'Budget': budget,
                    'PopSize': pop_size,
                    'Eval_Axis': eval_axis,
                    
                    # Core metrics
                    'Best_Value': run.get('best_value'),
                    'Best_Fitness': run.get('best_fitness'),
                    'Total_Weight': run.get('total_weight'),
                    'Capacity': run.get('capacity'),
                    'Is_Feasible': run.get('is_feasible', False),
                    'Elapsed_Time': run.get('elapsed_time'),
                    'Items_Selected': run.get('items_selected'),
                    'Capacity_Utilization': run.get('capacity_utilization'),
                    'Evaluations': run.get('evaluations'),
                    'Budget_Utilization': run.get('budget_utilization'),
                    'Status': run.get('status', 'ok'),
                    
                    # DP optimal and gap
                    'DP_Optimal': dp_optimal,
                    'Has_DP_Optimal': has_dp,
                    'Optimality_Gap': run.get('optimality_gap'),
                    'Constraint_Handling': metadata.get('constraint_handling'),
                    
                    # History and success_levels
                    'History': run.get('history'),
                    'Success_Levels': run.get('success_levels'),
                    'success_levels': run.get('success_levels'),
                }
                
                # Flatten success_levels for quick access (optional, but consistent with Rastrigin)
                if 'success_levels' in run and run['success_levels']:
                    for level, level_data in run['success_levels'].items():
                        flat_run[f'Success_{level.capitalize()}'] = level_data.get('success', False)
                        flat_run[f'HitEvals_{level.capitalize()}'] = level_data.get('hit_evaluations')
                        flat_run[f'Threshold_{level.capitalize()}'] = level_data.get('threshold')
                
                all_runs.append(flat_run)
        else:
            logger.warning(f"Filename does not match known patterns: {filename}")
            continue
    
    if not all_runs:
        logger.error("No valid data found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_runs)
    logger.info(f"Loaded {len(df)} runs across {df['Problem'].nunique()} problems")
    logger.info(f"  Rastrigin: {len(df[df['Problem']=='rastrigin'])} runs")
    logger.info(f"  Knapsack: {len(df[df['Problem']=='knapsack'])} runs")
    
    return df


# ============================================================================
# FIXED-TARGET ANALYSIS (COCO/BBOB)
# ============================================================================

def build_fixed_target_ecdf(df: pd.DataFrame, problem: str) -> pd.DataFrame:
    """
    Build ECDF (Empirical Cumulative Distribution Function) of runtime to target.
    
    For Rastrigin: use success_levels (gold/silver/bronze)
    For Knapsack: use gap thresholds {1%, 5%, 10%} if DP optimal available
    
    FIXED: Corrected gap computation for maximization problems.
    
    Returns DataFrame: Config/Instance, Level, Algorithm, tau (runtime), ECDF
    """
    df_prob = df[df['Problem'] == problem].copy()
    df_prob = df_prob[df_prob['Status'] == 'ok']
    
    ecdf_data = []
    
    if problem == 'rastrigin':
        levels = ['Gold', 'Silver', 'Bronze']
        
        for (config, scenario), group in df_prob.groupby(['Configuration', 'Scenario']):
            for level in levels:
                hit_col = f'HitEvals_{level}'
                if hit_col not in group.columns:
                    continue
                
                for algo, algo_group in group.groupby('Algorithm'):
                    runtimes = algo_group[hit_col].dropna().values
                    
                    if len(runtimes) == 0:
                        continue
                    
                    # Build ECDF
                    runtimes_sorted = np.sort(runtimes)
                    ecdf_vals = np.arange(1, len(runtimes_sorted) + 1) / len(algo_group)
                    
                    for tau, ecdf in zip(runtimes_sorted, ecdf_vals):
                        ecdf_data.append({
                            'Config': config,
                            'Scenario': scenario,
                            'Level': level,
                            'Algorithm': algo,
                            'tau': float(tau),
                            'ECDF': float(ecdf)
                        })
    
    elif problem == 'knapsack':
        # FIXED: Compute runtime from history instead of using null hit_evaluations
        df_with_dp = df_prob[df_prob['Has_DP_Optimal'] == True].copy()
        
        # Filter valid DP instances
        def is_valid_dp(dp_val):
            return dp_val is not None and not np.isnan(dp_val) and dp_val > 0
        
        df_with_dp = df_with_dp[df_with_dp['DP_Optimal'].apply(is_valid_dp)]
        
        if df_with_dp.empty:
            logger.warning("Knapsack: No DP optimal available, skipping ECDF")
            return pd.DataFrame()
        
        gap_thresholds = {
            'Gold': 1.0,
            'Silver': 5.0,
            'Bronze': 10.0,
        }
        
        for (n_items, inst_type, inst_seed, scenario), group in df_with_dp.groupby(
            ['N_Items', 'Instance_Type', 'Instance_Seed', 'Scenario']
        ):
            dp_opt = group['DP_Optimal'].iloc[0]
            
            if not is_valid_dp(dp_opt):
                continue
            
            for level_name, gap_thr in gap_thresholds.items():
                for algo, algo_group in group.groupby('Algorithm'):
                    runtimes = []
                    
                    for _, run in algo_group.iterrows():
                        history = run['History']
                        if not history or not isinstance(history, list):
                            continue
                        
                        eval_axis = run['Eval_Axis']
                        
                        # Convert history from fitness (negative) to value (positive) if needed
                        best_value = run.get('Best_Value')
                        best_fitness = run.get('Best_Fitness')
                        
                        if best_value is not None and best_fitness is not None \
                           and best_value > 0 and best_fitness < 0:
                            # history contains negative fitness, convert to positive value
                            values = [-v if v is not None else None for v in history]
                        else:
                            # history already contains positive values or unknown format
                            values = history
                        
                        hit = False
                        for gen_idx, val in enumerate(values):
                            if val is None or np.isnan(val):
                                continue
                            
                            # Compute gap: how far from optimal (in %)
                            gap = 100.0 * (dp_opt - val) / max(dp_opt, 1e-9)
                            
                            if gap <= gap_thr:
                                evals = (gen_idx + 1) * eval_axis
                                runtimes.append(evals)
                                hit = True
                                break
                    
                    if len(runtimes) == 0:
                        continue
                    
                    # Build ECDF
                    runtimes_sorted = np.sort(runtimes)
                    ecdf_vals = np.arange(1, len(runtimes_sorted) + 1) / len(algo_group)
                    
                    for tau, ecdf in zip(runtimes_sorted, ecdf_vals):
                        ecdf_data.append({
                            'N_Items': n_items,
                            'Instance_Type': inst_type,
                            'Instance_Seed': inst_seed,
                            'Scenario': scenario,
                            'Level': level_name,
                            'Algorithm': algo,
                            'tau': float(tau),
                            'ECDF': float(ecdf),
                        })
    
    return pd.DataFrame(ecdf_data)


def compute_ert(df: pd.DataFrame, problem: str, n_bootstrap: int = 10000) -> pd.DataFrame:
    """
    Compute Expected Running Time (ERT) with bootstrap confidence intervals.
    
    ERT = sum(evaluations before hitting target) / n_success
    For failed runs, use full budget evaluations.
    
    Returns DataFrame: Config/Instance, Level/Gap, Algorithm, ERT, CI_low, CI_high, N_success, N_total
    """
    df_prob = df[df['Problem'] == problem].copy()
    df_prob = df_prob[df_prob['Status'] == 'ok']
    
    ert_data = []
    
    if problem == 'rastrigin':
        levels = ['Gold', 'Silver', 'Bronze']
        
        for (config, scenario), group in df_prob.groupby(['Configuration', 'Scenario']):
            for level in levels:
                hit_col = f'HitEvals_{level}'
                success_col = f'Success_{level}'
                
                if hit_col not in group.columns or success_col not in group.columns:
                    continue
                
                for algo, algo_group in group.groupby('Algorithm'):
                    runtimes = []
                    budget = algo_group['Budget'].iloc[0]
                    
                    for _, run in algo_group.iterrows():
                        if run[success_col]:
                            runtimes.append(run[hit_col])
                        else:
                            # Failed run: use full budget
                            runtimes.append(budget if budget else run['Evaluations'])
                    
                    n_success = algo_group[success_col].sum()
                    n_total = len(algo_group)
                    
                    if n_success == 0:
                        ert = np.nan
                        ci_low = ci_high = np.nan
                    else:
                        ert = np.sum(runtimes) / n_success
                        
                        # Bootstrap CI
                        bootstrap_erts = []
                        for _ in range(n_bootstrap):
                            sample = np.random.choice(runtimes, size=len(runtimes), replace=True)
                            sample_success = np.random.choice(
                                algo_group[success_col].values, size=n_total, replace=True
                            ).sum()
                            if sample_success > 0:
                                bootstrap_erts.append(np.sum(sample) / sample_success)
                        
                        if bootstrap_erts:
                            ci_low, ci_high = np.percentile(bootstrap_erts, [2.5, 97.5])
                        else:
                            ci_low = ci_high = ert
                    
                    ert_data.append({
                        'Config': config,
                        'Scenario': scenario,
                        'Level': level,
                        'Algorithm': algo,
                        'ERT': float(ert) if np.isfinite(ert) else None,
                        'CI_low': float(ci_low) if np.isfinite(ci_low) else None,
                        'CI_high': float(ci_high) if np.isfinite(ci_high) else None,
                        'N_success': int(n_success),
                        'N_total': int(n_total)
                    })
    
    elif problem == 'knapsack':
        # SAFETY GATE 2: DP Validation
        df_with_dp = df_prob[df_prob['Has_DP_Optimal'] == True].copy()
        
        if df_with_dp.empty:
            logger.warning("Knapsack: No DP optimal available, skipping ERT")
            return pd.DataFrame()
        
        # Filter valid DP instances
        def is_valid_dp(dp_val):
            return dp_val is not None and not np.isnan(dp_val) and dp_val > 0
        
        df_with_dp = df_with_dp[df_with_dp['DP_Optimal'].apply(is_valid_dp)]
        
        # SAFETY GATE 6: Handle empty dataset
        if df_with_dp.empty:
            logger.warning("Knapsack: No instances with valid DP optimal after filtering, skipping ERT")
            return pd.DataFrame()
        
        gap_thresholds = [1.0, 5.0, 10.0]
        
        for (n_items, inst_type, inst_seed, scenario), group in df_with_dp.groupby(
            ['N_Items', 'Instance_Type', 'Instance_Seed', 'Scenario']
        ):
            dp_opt = group['DP_Optimal'].iloc[0]
            
            if not is_valid_dp(dp_opt):
                continue
            
            for gap_thr in gap_thresholds:
                for algo, algo_group in group.groupby('Algorithm'):
                    runtimes = []
                    successes = []
                    budget = algo_group['Budget'].iloc[0]
                    
                    for _, run in algo_group.iterrows():
                        history = run['History']
                        if not history or not isinstance(history, list):
                            runtimes.append(budget if budget else run['Evaluations'])
                            successes.append(False)
                            continue
                        
                        eval_axis = run['Eval_Axis']
                        
                        # Convert history from fitness (negative) to value (positive) if needed
                        best_value = run.get('Best_Value')
                        best_fitness = run.get('Best_Fitness')
                        
                        if best_value is not None and best_fitness is not None \
                           and best_value > 0 and best_fitness < 0:
                            # history contains negative fitness, convert to positive value
                            values = [-v if v is not None else None for v in history]
                        else:
                            # history already contains positive values or unknown format
                            values = history
                        
                        hit = False
                        
                        for gen_idx, val in enumerate(values):
                            if val is None or np.isnan(val):
                                continue
                            
                            # FIXED: Correct gap for maximization
                            gap = 100.0 * (dp_opt - val) / max(dp_opt, 1e-9)
                            
                            if gap <= gap_thr:
                                runtimes.append((gen_idx + 1) * eval_axis)
                                successes.append(True)
                                hit = True
                                break
                    
                    if len(runtimes) == 0:
                        continue
                    
                    n_success = sum(successes)
                    n_total = len(algo_group)
                    
                    if n_success == 0:
                        ert = np.nan
                        ci_low = ci_high = np.nan
                    else:
                        ert = np.sum(runtimes) / n_success
                        
                        # Bootstrap CI
                        bootstrap_erts = []
                        for _ in range(n_bootstrap):
                            sample = np.random.choice(runtimes, size=len(runtimes), replace=True)
                            sample_success = np.random.choice(
                                successes, size=n_total, replace=True
                            ).sum()
                            if sample_success > 0:
                                bootstrap_erts.append(np.sum(sample) / sample_success)
                        
                        if bootstrap_erts:
                            ci_low, ci_high = np.percentile(bootstrap_erts, [2.5, 97.5])
                        else:
                            ci_low = ci_high = ert
                    
                    ert_data.append({
                        'N_Items': n_items,
                        'Instance_Type': inst_type,
                        'Instance_Seed': inst_seed,
                        'Scenario': scenario,
                        'Gap_Threshold': gap_thr,
                        'Algorithm': algo,
                        'ERT': float(ert) if np.isfinite(ert) else None,
                        'CI_low': float(ci_low) if np.isfinite(ci_low) else None,
                        'CI_high': float(ci_high) if np.isfinite(ci_high) else None,
                        'N_success': int(n_success),
                        'N_total': int(n_total)
                    })
    
    return pd.DataFrame(ert_data)


# ============================================================================
# FIXED-BUDGET ANALYSIS
# ============================================================================

def build_fixed_budget(df: pd.DataFrame, problem: str, 
                       budget_fractions: List[float] = [0.1, 0.3, 0.5, 1.0]) -> pd.DataFrame:
    """
    Interpolate best-so-far at fixed budget checkpoints.
    
    For Rastrigin: extract |f(x)| at budget fractions
    For Knapsack: extract gap% (if DP available) or best_value
    
    Returns DataFrame with Error@10%, Error@30%, etc. or Gap@10%, etc.
    """
    df_prob = df[df['Problem'] == problem].copy()
    df_prob = df_prob[df_prob['Status'] == 'ok']
    
    fixed_budget_data = []
    
    if problem == 'rastrigin':
        for (config, scenario), group in df_prob.groupby(['Configuration', 'Scenario']):
            for algo, algo_group in group.groupby('Algorithm'):
                for _, run in algo_group.iterrows():
                    history = run['History']
                    if not history or not isinstance(history, list):
                        continue
                    
                    budget = run['Budget']
                    eval_axis = run['Eval_Axis']
                    
                    row = {
                        'Config': config,
                        'Scenario': scenario,
                        'Algorithm': algo,
                        'Algo_Seed': run['Algo_Seed']
                    }
                    
                    for frac in budget_fractions:
                        target_evals = budget * frac
                        target_gen = int(target_evals / eval_axis) - 1
                        target_gen = max(0, min(target_gen, len(history) - 1))
                        
                        val = history[target_gen]
                        row[f'Error@{int(frac*100)}%'] = abs(val) if val is not None else np.nan
                    
                    fixed_budget_data.append(row)
    
    elif problem == 'knapsack':
        has_dp = df_prob['Has_DP_Optimal'].any()
        
        for group_key, group in df_prob.groupby(['N_Items', 'Instance_Type', 'Instance_Seed', 'Scenario']):
            group_dict = dict(zip(['N_Items', 'Instance_Type', 'Instance_Seed', 'Scenario'], group_key))
            
            dp_opt = group['DP_Optimal'].iloc[0] if has_dp else None
            
            for algo, algo_group in group.groupby('Algorithm'):
                for _, run in algo_group.iterrows():
                    history = run['History']
                    if not history or not isinstance(history, list):
                        continue
                    
                    budget = run['Budget']
                    eval_axis = run['Eval_Axis']
                    
                    row = {
                        **group_dict,
                        'Algorithm': algo,
                        'Algo_Seed': run['Algo_Seed']
                    }
                    
                    for frac in budget_fractions:
                        target_evals = budget * frac
                        target_gen = int(target_evals / eval_axis) - 1
                        target_gen = max(0, min(target_gen, len(history) - 1))
                        
                        val = history[target_gen]
                        
                        if has_dp and dp_opt and dp_opt > 0 and val is not None:
                            gap = 100.0 * abs(dp_opt - val) / dp_opt
                            row[f'Gap@{int(frac*100)}%'] = gap
                        else:
                            row[f'BestValue@{int(frac*100)}%'] = val if val is not None else np.nan
                    
                    fixed_budget_data.append(row)
    
    return pd.DataFrame(fixed_budget_data)


# ============================================================================
# PERFORMANCE PROFILES & DATA PROFILES
# ============================================================================

def build_performance_profiles(df: pd.DataFrame, problem: str) -> pd.DataFrame:
    """
    Build performance profiles (Dolan & Moré).
    
    For each problem instance p and solver s:
    - t_{p,s} = runtime to reach target (or +inf if failed)
    - r_{p,s} = t_{p,s} / min_s(t_{p,s})
    - phi_s(tau) = fraction of problems where r_{p,s} <= tau
    
    Returns DataFrame: Problem, Instance/Config, Level/Gap, Algorithm, tau, phi
    """
    df_prob = df[df['Problem'] == problem].copy()
    df_prob = df_prob[df_prob['Status'] == 'ok']
    
    profile_data = []
    
    if problem == 'rastrigin':
        levels = ['Gold', 'Silver', 'Bronze']
        
        for level in levels:
            hit_col = f'HitEvals_{level}'
            success_col = f'Success_{level}'
            
            if hit_col not in df_prob.columns:
                continue
            
            # Group by problem instance (config + scenario)
            for (config, scenario), group in df_prob.groupby(['Configuration', 'Scenario']):
                # Compute average runtime per algorithm
                algo_runtimes = {}
                
                for algo, algo_group in group.groupby('Algorithm'):
                    runtimes = []
                    budget = algo_group['Budget'].iloc[0]
                    
                    for _, run in algo_group.iterrows():
                        if run[success_col]:
                            runtimes.append(run[hit_col])
                        else:
                            runtimes.append(np.inf)
                    
                    algo_runtimes[algo] = np.mean(runtimes)
                
                # Compute performance ratios
                min_time = min([t for t in algo_runtimes.values() if np.isfinite(t)], default=1.0)
                
                for algo, avg_time in algo_runtimes.items():
                    if np.isfinite(avg_time):
                        ratio = avg_time / min_time
                    else:
                        ratio = np.inf
                    
                    algo_runtimes[algo] = ratio
                
                # Collect for profile building
                for algo, ratio in algo_runtimes.items():
                    profile_data.append({
                        'Problem': problem,
                        'Config': config,
                        'Scenario': scenario,
                        'Level': level,
                        'Algorithm': algo,
                        'ratio': ratio
                    })
    
    elif problem == 'knapsack':
        # SAFETY GATE 2: DP Validation per Instance
        df_with_dp = df_prob[df_prob['Has_DP_Optimal'] == True].copy()
        
        use_best_known = False
        
        if df_with_dp.empty:
            logger.warning("Knapsack: No DP optimal, using best-known baseline for profiles")
            df_with_dp = df_prob.copy()
            use_best_known = True
        else:
            # Filter instances with valid DP
            def is_valid_dp(dp_val):
                return dp_val is not None and not np.isnan(dp_val) and dp_val > 0
            
            df_with_dp_valid = df_with_dp[df_with_dp['DP_Optimal'].apply(is_valid_dp)]
            
            if df_with_dp_valid.empty:
                logger.warning("Knapsack: No valid DP instances, using best-known baseline")
                df_with_dp = df_prob.copy()
                use_best_known = True
            else:
                df_with_dp = df_with_dp_valid
        
        # SAFETY GATE 6: Check if we have data to analyze
        if df_with_dp.empty:
            logger.warning("Knapsack: No data available for performance profiles")
            return pd.DataFrame()
        
        gap_thresholds = [1.0, 5.0, 10.0] if not use_best_known else [None]
        
        for gap_thr in gap_thresholds:
            # SAFETY GATE 3: Problem Key Definition (N_Items, Instance_Type, Instance_Seed)
            for (n_items, inst_type, inst_seed, scenario), group in df_with_dp.groupby(
                ['N_Items', 'Instance_Type', 'Instance_Seed', 'Scenario']
            ):
                if use_best_known:
                    # Best-known = max Best_Value across all algorithms for this instance
                    best_known = group['Best_Value'].max()
                    if best_known <= 0:
                        continue
                else:
                    dp_opt = group['DP_Optimal'].iloc[0]
                    if dp_opt is None or np.isnan(dp_opt) or dp_opt <= 0:
                        continue
                
                # FIXED: Compute cost metric compatible with minimization
                algo_costs = {}
                
                for algo, algo_group in group.groupby('Algorithm'):
                    costs = []
                    
                    for _, run in algo_group.iterrows():
                        best_value = run['Best_Value']
                        
                        if best_value is None or np.isnan(best_value):
                            costs.append(np.inf)
                            continue
                        
                        # SAFETY GATE 4: Sign and Unit Consistency
                        # For maximization, cost = optimal - achieved
                        if use_best_known:
                            cost = best_known - best_value
                        else:
                            cost = dp_opt - best_value
                        
                        # Ensure cost is non-negative (could be negative due to float errors)
                        cost = max(cost, 0.0)
                        
                        costs.append(cost)
                    
                    # Average cost for this algorithm on this instance
                    finite_costs = [c for c in costs if np.isfinite(c)]
                    if finite_costs:
                        algo_costs[algo] = np.mean(finite_costs)
                    else:
                        algo_costs[algo] = np.inf
                
                # SAFETY GATE 3: Compute min_cost per problem instance
                min_cost = min([c for c in algo_costs.values() if np.isfinite(c)], default=1.0)
                
                if min_cost <= 0:
                    min_cost = 1.0  # Fallback to avoid division by zero
                
                # Compute performance ratios
                for algo, cost in algo_costs.items():
                    if np.isfinite(cost):
                        ratio = cost / min_cost
                        # SAFETY GATE 5: Clamp Performance Ratio
                        ratio = max(ratio, 1.0)
                    else:
                        ratio = np.inf
                    
                    profile_data.append({
                        'Problem': problem,
                        'N_Items': n_items,
                        'Instance_Type': inst_type,
                        'Instance_Seed': inst_seed,
                        'Scenario': scenario,
                        'Gap_Threshold': gap_thr if not use_best_known else None,
                        'Algorithm': algo,
                        'ratio': ratio
                    })
    
    # Build performance profiles from ratios
    if not profile_data:
        logger.warning(f"{problem}: No profile data collected")
        return pd.DataFrame()
    
    df_ratios = pd.DataFrame(profile_data)
    
    # Group by algorithm and compute phi(tau)
    algos = df_ratios['Algorithm'].unique()
    finite_ratios = df_ratios[df_ratios['ratio'] != np.inf]['ratio']
    
    if len(finite_ratios) == 0:
        logger.warning(f"{problem}: All ratios are infinite, cannot build profiles")
        return pd.DataFrame()
    
    tau_max = finite_ratios.max()
    tau_values = np.logspace(0, np.log10(tau_max * 2), 100)
    
    phi_data = []
    
    for algo in algos:
        algo_ratios = df_ratios[df_ratios['Algorithm'] == algo]['ratio'].values
        n_problems = len(algo_ratios)
        
        for tau in tau_values:
            phi = np.mean(algo_ratios <= tau)
            phi_data.append({
                'Problem': problem,
                'Algorithm': algo,
                'tau': float(tau),
                'phi': float(phi)
            })
    
    return pd.DataFrame(phi_data)


def build_data_profiles(df: pd.DataFrame, problem: str) -> pd.DataFrame:
    """
    Build data profiles (Moré & Wild).
    
    psi_s(nu) = fraction of problems where solver s reaches target within nu evaluations
    
    Returns DataFrame: Problem, Algorithm, nu (budget), psi
    """
    df_prob = df[df['Problem'] == problem].copy()
    df_prob = df_prob[df_prob['Status'] == 'ok']
    
    # Determine max budget
    max_budget = df_prob['Budget'].max()
    nu_values = np.logspace(np.log10(max_budget * 0.01), np.log10(max_budget), 50)
    
    psi_data = []
    
    if problem == 'rastrigin':
        level = 'Silver'  # Use silver as default target
        hit_col = f'HitEvals_{level}'
        success_col = f'Success_{level}'
        
        if hit_col not in df_prob.columns:
            return pd.DataFrame()
        
        algos = df_prob['Algorithm'].unique()
        
        for algo in algos:
            algo_df = df_prob[df_prob['Algorithm'] == algo]
            
            # Group by config+scenario to count problems
            problems = algo_df.groupby(['Configuration', 'Scenario'])
            n_problems = len(problems)
            
            for nu in nu_values:
                n_solved = 0
                
                for (config, scenario), group in problems:
                    # Check if any run in this problem solved within nu evals
                    solved = ((group[success_col]) & (group[hit_col] <= nu)).any()
                    if solved:
                        n_solved += 1
                
                psi = n_solved / n_problems if n_problems > 0 else 0.0
                
                psi_data.append({
                    'Problem': problem,
                    'Algorithm': algo,
                    'nu': float(nu),
                    'psi': float(psi)
                })
    
    elif problem == 'knapsack':
        # SAFETY GATE 2: DP Validation
        df_with_dp = df_prob[df_prob['Has_DP_Optimal'] == True].copy()
        
        if df_with_dp.empty:
            logger.warning("Knapsack: No DP optimal for data profiles")
            return pd.DataFrame()
        
        # Filter valid DP instances
        def is_valid_dp(dp_val):
            return dp_val is not None and not np.isnan(dp_val) and dp_val > 0
        
        df_with_dp = df_with_dp[df_with_dp['DP_Optimal'].apply(is_valid_dp)]
        
        # SAFETY GATE 6: Handle empty dataset
        if df_with_dp.empty:
            logger.warning("Knapsack: No valid DP instances for data profiles")
            return pd.DataFrame()
        
        gap_thr = 5.0  # Use 5% gap as default target
        algos = df_with_dp['Algorithm'].unique()
        
        for algo in algos:
            algo_df = df_with_dp[df_with_dp['Algorithm'] == algo]
            
            # SAFETY GATE 3: Group by instance (N_Items, Instance_Type, Instance_Seed, Scenario)
            problems = algo_df.groupby(['N_Items', 'Instance_Type', 'Instance_Seed', 'Scenario'])
            n_problems = len(problems)
            
            if n_problems == 0:
                continue
            
            for nu in nu_values:
                n_solved = 0
                
                for group_key, group in problems:
                    dp_opt = group['DP_Optimal'].iloc[0]
                    if not is_valid_dp(dp_opt):
                        continue
                    
                    # SAFETY GATE 1: Feasible Gate - check if any run solved within nu evals
                    solved = False
                    
                    for _, run in group.iterrows():
                        history = run['History']
                        if not history:
                            continue
                        
                        eval_axis = run['Eval_Axis']
                        
                        # Convert history from fitness (negative) to value (positive) if needed
                        best_value = run.get('Best_Value')
                        best_fitness = run.get('Best_Fitness')
                        
                        if best_value is not None and best_fitness is not None \
                           and best_value > 0 and best_fitness < 0:
                            # history contains negative fitness, convert to positive value
                            values = [-v if v is not None else None for v in history]
                        else:
                            # history already contains positive values or unknown format
                            values = history
                        
                        for gen_idx, val in enumerate(values):
                            evals = (gen_idx + 1) * eval_axis
                            if evals > nu:
                                break
                            
                            if val is not None and not np.isnan(val):
                                # CORRECTED GAP COMPUTATION
                                gap = 100.0 * (dp_opt - val) / max(dp_opt, 1e-9)
                                
                                if gap <= gap_thr:
                                    solved = True
                                    break
                        
                        if solved:
                            break
                    
                    if solved:
                        n_solved += 1
                
                psi = n_solved / n_problems if n_problems > 0 else 0.0
                
                psi_data.append({
                    'Problem': problem,
                    'Algorithm': algo,
                    'nu': float(nu),
                    'psi': float(psi)
                })
    
    return pd.DataFrame(psi_data)


# ============================================================================
# DIVERSITY ANALYSIS (RASTRIGIN)
# ============================================================================

def summarize_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract diversity metrics from Stats_History for Rastrigin.
    
    Computes:
    - Div_Initial, Div_Mid50, Div_Final (raw diversity)
    - Div_Norm_* (normalized by sqrt(D))
    - Div_Min, Div_Drop (minimum and total drop)
    
    Returns DataFrame with diversity summary per run
    """
    df_rast = df[(df['Problem'] == 'rastrigin') & (df['Status'] == 'ok')].copy()
    
    div_data = []
    
    for _, run in df_rast.iterrows():
        stats_hist = run['Stats_History']
        dim = run['Dim']
        
        if not stats_hist or not isinstance(stats_hist, list) or dim is None:
            continue
        
        # Extract diversity values
        diversities = [s.get('diversity') for s in stats_hist if s and 'diversity' in s]
        diversities = [d for d in diversities if d is not None and np.isfinite(d)]
        
        if len(diversities) == 0:
            continue
        
        sqrt_d = np.sqrt(dim)
        diversities = np.array(diversities)
        diversities_norm = diversities / sqrt_d
        
        n_gens = len(diversities)
        mid_idx = n_gens // 2
        
        row = {
            'Configuration': run['Configuration'],
            'Scenario': run['Scenario'],
            'Algorithm': run['Algorithm'],
            'Algo_Seed': run['Algo_Seed'],
            'Dim': dim,
            
            'Div_Initial': float(diversities[0]),
            'Div_Mid50': float(diversities[mid_idx]),
            'Div_Final': float(diversities[-1]),
            'Div_Min': float(np.min(diversities)),
            'Div_Drop': float(diversities[0] - diversities[-1]),
            
            'Div_Norm_Initial': float(diversities_norm[0]),
            'Div_Norm_Mid50': float(diversities_norm[mid_idx]),
            'Div_Norm_Final': float(diversities_norm[-1]),
            'Div_Norm_Min': float(np.min(diversities_norm)),
            'Div_Norm_Drop': float(diversities_norm[0] - diversities_norm[-1]),
        }
        
        div_data.append(row)
    
    return pd.DataFrame(div_data)


def compute_stagnation(df: pd.DataFrame, threshold: float = 1e-6) -> pd.DataFrame:
    """
    Compute stagnation metrics (longest period without improvement).
    
    Returns DataFrame with Stagnation_Gens per run
    """
    df_rast = df[(df['Problem'] == 'rastrigin') & (df['Status'] == 'ok')].copy()
    
    stag_data = []
    
    for _, run in df_rast.iterrows():
        history = run['History']
        if not history or not isinstance(history, list):
            continue
        
        history = np.array([h if h is not None else np.nan for h in history])
        history = history[~np.isnan(history)]
        
        if len(history) < 2:
            continue
        
        # Find longest stagnation period
        max_stag = 0
        current_stag = 0
        prev_best = history[0]
        
        for val in history[1:]:
            if abs(val - prev_best) < threshold:
                current_stag += 1
            else:
                max_stag = max(max_stag, current_stag)
                current_stag = 0
                prev_best = val
        
        max_stag = max(max_stag, current_stag)
        
        stag_data.append({
            'Configuration': run['Configuration'],
            'Scenario': run['Scenario'],
            'Algorithm': run['Algorithm'],
            'Algo_Seed': run['Algo_Seed'],
            'Stagnation_Gens': int(max_stag)
        })
    
    return pd.DataFrame(stag_data)


# ============================================================================
# STATISTICAL TESTING (ENHANCED)
# ============================================================================

def pairwise_tests_and_effects(
    df: pd.DataFrame,
    metric_col: str,
    group_cols: List[str],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform pairwise statistical tests with Holm correction and effect sizes.
    
    Computes:
    - Wilcoxon signed-rank test
    - Holm correction for multiple comparisons
    - Vargha-Delaney A12 effect size
    - Cliff's delta effect size
    
    Returns DataFrame: Group, Algo1, Algo2, p_value, p_adj, A12, CliffDelta, Significant
    """
    results = []
    
    for group_key, group in df.groupby(group_cols):
        group_dict = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else [group_key]))
        
        algos = sorted(group['Algorithm'].unique())
        if len(algos) < 2:
            continue
        
        # Collect p-values for Holm correction
        comparisons = []
        
        for i, algo1 in enumerate(algos):
            for algo2 in algos[i+1:]:
                data1 = group[group['Algorithm'] == algo1][metric_col].dropna().values
                data2 = group[group['Algorithm'] == algo2][metric_col].dropna().values
                
                if len(data1) < 2 or len(data2) < 2:
                    continue
                
                # Wilcoxon test
                try:
                    stat, p_val = stats.wilcoxon(data1, data2, alternative='two-sided')
                except Exception as e:
                    logger.warning(f"Wilcoxon test failed: {e}")
                    continue
                
                # Vargha-Delaney A12
                a12 = vargha_delaney_a12(data1, data2)
                
                # Cliff's delta
                cliff_delta = cliffs_delta(data1, data2)
                
                comparisons.append({
                    **group_dict,
                    'Algo1': algo1,
                    'Algo2': algo2,
                    'p_value': float(p_val),
                    'A12': float(a12),
                    'CliffDelta': float(cliff_delta)
                })
        
        if not comparisons:
            continue
        
        # Holm correction
        p_values = [c['p_value'] for c in comparisons]
        p_adj = holm_correction(p_values, alpha)
        
        for comp, p_a in zip(comparisons, p_adj):
            comp['p_adj'] = float(p_a)
            comp['Significant'] = p_a < alpha
            results.append(comp)
    
    return pd.DataFrame(results)


def vargha_delaney_a12(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Compute Vargha-Delaney A12 effect size.
    
    A12 = P(X1 > X2) + 0.5 * P(X1 = X2)
    
    Interpretation:
    - 0.5: no effect
    - > 0.5: data1 dominates data2
    - < 0.5: data2 dominates data1
    """
    n1 = len(data1)
    n2 = len(data2)
    
    r_sum = 0.0
    for x1 in data1:
        r_sum += np.sum(x1 > data2) + 0.5 * np.sum(x1 == data2)
    
    a12 = r_sum / (n1 * n2)
    return a12


def cliffs_delta(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Compute Cliff's delta effect size.
    
    delta = (# pairs where X1 > X2 - # pairs where X1 < X2) / (n1 * n2)
    
    Interpretation:
    - 0: no effect
    - +1: all X1 > X2 (large positive effect)
    - -1: all X1 < X2 (large negative effect)
    """
    n1 = len(data1)
    n2 = len(data2)
    
    greater = 0
    less = 0
    
    for x1 in data1:
        greater += np.sum(x1 > data2)
        less += np.sum(x1 < data2)
    
    delta = (greater - less) / (n1 * n2)
    return delta


def holm_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.
    
    Returns adjusted p-values
    """
    n = len(p_values)
    if n == 0:
        return []
    
    # Sort p-values with original indices
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    
    # Apply Holm correction
    p_adj = [0.0] * n
    
    for rank, (orig_idx, p_val) in enumerate(indexed_p, 1):
        adjusted = min(p_val * (n - rank + 1), 1.0)
        p_adj[orig_idx] = adjusted
    
    # Enforce monotonicity
    for i in range(1, n):
        if p_adj[i] < p_adj[i-1]:
            p_adj[i] = p_adj[i-1]
    
    return p_adj


def nemenyi_test(df: pd.DataFrame, metric_col: str, group_cols: List[str]) -> pd.DataFrame:
    """
    Perform Nemenyi post-hoc test after Friedman test.
    
    Returns DataFrame with critical difference (CD) for CD diagram.
    """
    from scipy.stats import rankdata
    
    cd_data = []
    
    for group_key, group in df.groupby(group_cols):
        algos = sorted(group['Algorithm'].unique())
        if len(algos) < 2:
            continue
        
        # Compute average ranks
        ranks = {}
        
        # Need to rank algorithms per problem instance (seed)
        for seed, seed_group in group.groupby('Algo_Seed'):
            values = []
            for algo in algos:
                algo_val = seed_group[seed_group['Algorithm'] == algo][metric_col].values
                if len(algo_val) > 0:
                    values.append((algo, algo_val[0]))
            
            if len(values) < 2:
                continue
            
            # Rank (lower metric = better rank)
            sorted_values = sorted(values, key=lambda x: x[1])
            for rank, (algo, _) in enumerate(sorted_values, 1):
                if algo not in ranks:
                    ranks[algo] = []
                ranks[algo].append(rank)
        
        # Compute average ranks
        avg_ranks = {algo: np.mean(r) for algo, r in ranks.items()}
        n_problems = max(len(r) for r in ranks.values())
        k = len(algos)
        
        # Nemenyi critical difference
        # CD = q_alpha * sqrt(k(k+1) / (6N))
        # q_alpha for Nemenyi at alpha=0.05
        q_alpha = 2.344  # For k=4, approximate value
        
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n_problems))
        
        for algo, avg_rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
            cd_data.append({
                'Algorithm': algo,
                'Avg_Rank': avg_rank,
                'N_Problems': n_problems,
                'CD': cd
            })
    
    return pd.DataFrame(cd_data)


# ============================================================================
# RASTRIGIN ANALYSIS (ENHANCED)
# ============================================================================

def analyze_rastrigin(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive Rastrigin analysis with COCO/BBOB metrics and diversity.
    """
    df_rast = df[df['Problem'] == 'rastrigin'].copy()
    
    if df_rast.empty:
        logger.warning("No Rastrigin data found")
        return {}
    
    df_rast = df_rast[df_rast['Status'] == 'ok']
    df_rast['Error_To_Optimum'] = df_rast['Best_Fitness'].abs()
    
    print("\n" + "=" * 100)
    print("RASTRIGIN ANALYSIS (ENHANCED WITH COCO/BBOB METRICS)")
    print("=" * 100)
    
    results = {}
    
    # 1. Fixed-target ECDF
    print("\n[Computing Fixed-Target ECDF...]")
    ecdf_df = build_fixed_target_ecdf(df, 'rastrigin')
    if not ecdf_df.empty:
        results['fixed_target_ecdf'] = ecdf_df
        print(f"  Created ECDF data: {len(ecdf_df)} rows")
    
    # 2. ERT
    print("\n[Computing Expected Running Time (ERT)...]")
    ert_df = compute_ert(df, 'rastrigin')
    if not ert_df.empty:
        results['ert'] = ert_df
        print(f"  Created ERT data: {len(ert_df)} rows")
        print("\nERT Summary (Silver level):")
        print(ert_df[ert_df['Level'] == 'Silver'].to_string(index=False))
    
    # 3. Fixed-budget
    print("\n[Computing Fixed-Budget Performance...]")
    fixed_budget_df = build_fixed_budget(df, 'rastrigin')
    if not fixed_budget_df.empty:
        results['fixed_budget'] = fixed_budget_df
        print(f"  Created fixed-budget data: {len(fixed_budget_df)} rows")
    
    # 4. Performance profiles
    print("\n[Computing Performance Profiles...]")
    perf_profiles_df = build_performance_profiles(df, 'rastrigin')
    if not perf_profiles_df.empty:
        results['performance_profiles'] = perf_profiles_df
        print(f"  Created performance profiles: {len(perf_profiles_df)} rows")
    
    # 5. Data profiles
    print("\n[Computing Data Profiles...]")
    data_profiles_df = build_data_profiles(df, 'rastrigin')
    if not data_profiles_df.empty:
        results['data_profiles'] = data_profiles_df
        print(f"  Created data profiles: {len(data_profiles_df)} rows")
    
    # 6. Diversity analysis
    print("\n[Computing Diversity Metrics...]")
    diversity_df = summarize_diversity(df)
    if not diversity_df.empty:
        results['diversity_summary'] = diversity_df
        print(f"  Created diversity summary: {len(diversity_df)} rows")
        
        # Aggregate diversity
        div_agg = diversity_df.groupby(['Configuration', 'Scenario', 'Algorithm']).agg({
            'Div_Norm_Initial': 'mean',
            'Div_Norm_Final': 'mean',
            'Div_Norm_Drop': 'mean'
        }).reset_index()
        print("\nDiversity Summary (Normalized by √D):")
        print(div_agg.to_string(index=False))
    
    # 7. Stagnation analysis
    print("\n[Computing Stagnation Metrics...]")
    stagnation_df = compute_stagnation(df)
    if not stagnation_df.empty:
        results['stagnation'] = stagnation_df
        print(f"  Created stagnation data: {len(stagnation_df)} rows")
    
    # 8. Enhanced statistical testing
    print("\n[Performing Enhanced Statistical Tests...]")
    
    # Friedman test (existing)
    # ...existing code...
    
    # Pairwise tests with effect sizes
    print("\n[Pairwise Comparisons with Effect Sizes...]")
    pairwise_df = pairwise_tests_and_effects(
        df_rast,
        metric_col='Error_To_Optimum',
        group_cols=['Configuration', 'Scenario']
    )
    if not pairwise_df.empty:
        results['pairwise_stats'] = pairwise_df
        print(f"  Created pairwise tests: {len(pairwise_df)} rows")
        print("\nSignificant Pairwise Differences:")
        sig_pairs = pairwise_df[pairwise_df['Significant']]
        if not sig_pairs.empty:
            print(sig_pairs[['Configuration', 'Scenario', 'Algo1', 'Algo2', 'p_adj', 'A12', 'CliffDelta']].to_string(index=False))
    
    # Nemenyi CD diagram input
    print("\n[Computing Nemenyi CD Diagram Input...]")
    cd_df = nemenyi_test(df_rast, 'Error_To_Optimum', ['Configuration', 'Scenario'])
    if not cd_df.empty:
        results['cd_input'] = cd_df
        print(f"  Created CD input: {len(cd_df)} rows")
    
    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, result_df in results.items():
            filename = f'rastrigin_{name}.csv'
            result_df.to_csv(output_dir / filename, index=False)
            logger.info(f"✓ Saved: {filename}")
    
    return results


# ============================================================================
# KNAPSACK ANALYSIS (ENHANCED)
# ============================================================================

def analyze_knapsack(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive Knapsack analysis with fixed-target (if DP available) and profiles.
    """
    df_knap = df[df['Problem'] == 'knapsack'].copy()
    
    if df_knap.empty:
        logger.warning("No Knapsack data found")
        return {}
    
    df_knap = df_knap[df_knap['Status'] == 'ok']
    
    print("\n" + "=" * 100)
    print("KNAPSACK ANALYSIS (ENHANCED)")
    print("=" * 100)
    
    results = {}
    
    # 1. Fixed-target (if DP available)
    has_dp = df_knap['Has_DP_Optimal'].any()
    
    if has_dp:
        print("\n[Computing Fixed-Target ECDF (DP optimal available)...]")
        ecdf_df = build_fixed_target_ecdf(df, 'knapsack')
        if not ecdf_df.empty:
            results['fixed_target_ecdf'] = ecdf_df
            print(f"  Created ECDF data: {len(ecdf_df)} rows")
        
        print("\n[Computing Expected Running Time (ERT)...]")
        ert_df = compute_ert(df, 'knapsack')
        if not ert_df.empty:
            results['ert'] = ert_df
            print(f"  Created ERT data: {len(ert_df)} rows")
    else:
        print("\n[Skipping Fixed-Target Analysis: No DP optimal available]")
    
    # 2. Fixed-budget
    print("\n[Computing Fixed-Budget Performance...]")
    fixed_budget_df = build_fixed_budget(df, 'knapsack')
    if not fixed_budget_df.empty:
        results['fixed_budget'] = fixed_budget_df
        print(f"  Created fixed-budget data: {len(fixed_budget_df)} rows")
    
    # 3. Performance profiles
    print("\n[Computing Performance Profiles...]")
    perf_profiles_df = build_performance_profiles(df, 'knapsack')
    if not perf_profiles_df.empty:
        results['performance_profiles'] = perf_profiles_df
        print(f"  Created performance profiles: {len(perf_profiles_df)} rows")
    
    # 4. Data profiles
    print("\n[Computing Data Profiles...]")
    data_profiles_df = build_data_profiles(df, 'knapsack')
    if not data_profiles_df.empty:
        results['data_profiles'] = data_profiles_df
        print(f"  Created data profiles: {len(data_profiles_df)} rows")
    
    # 5. Enhanced statistical testing
    # Pairwise tests with effect sizes
    metric_col = 'Optimality_Gap' if has_dp else 'Best_Value'
    
    print(f"\n[Pairwise Comparisons with Effect Sizes (metric: {metric_col})...]")
    pairwise_df = pairwise_tests_and_effects(
        df_knap,
        metric_col=metric_col,
        group_cols=['N_Items', 'Instance_Type', 'Instance_Seed', 'Scenario']
    )
    if not pairwise_df.empty:
        results['pairwise_stats'] = pairwise_df
        print(f"  Created pairwise tests: {len(pairwise_df)} rows")
    
    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, result_df in results.items():
            filename = f'knapsack_{name}.csv'
            result_df.to_csv(output_dir / filename, index=False)
            logger.info(f"✓ Saved: {filename}")
    
    return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_summary(
    df: pd.DataFrame,
    group_by_cols: List[str],
    value_cols: List[str],
    agg_funcs: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Flexible summary creation function.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data from load_all_results_to_dataframe()
    group_by_cols : List[str]
        Columns to group by
    value_cols : List[str]
        Columns to aggregate
    agg_funcs : Dict[str, str], optional
        Custom aggregation functions. Default: {'mean', 'std', 'median', 'min', 'max'}
    
    Returns
    -------
    pd.DataFrame
        Summary dataframe
    """
    if agg_funcs is None:
        # Default aggregations
        agg_funcs = {
            'mean': 'mean',
            'std': 'std',
            'median': 'median',
            'min': 'min',
            'max': 'max'
        }
    
    summary_data = []
    
    for group_key, group in df.groupby(group_by_cols):
        row = dict(zip(group_by_cols, group_key if isinstance(group_key, tuple) else [group_key]))
        row['N_Runs'] = len(group)
        
        for col in value_cols:
            if col not in group.columns:
                continue
            
            values = group[col].dropna().values
            if len(values) == 0:
                continue
            
            for agg_name, agg_func in agg_funcs.items():
                if agg_func == 'mean':
                    row[f'{col}_Mean'] = np.mean(values)
                elif agg_func == 'std':
                    row[f'{col}_Std'] = np.std(values)
                elif agg_func == 'median':
                    row[f'{col}_Median'] = np.median(values)
                elif agg_func == 'min':
                    row[f'{col}_Min'] = np.min(values)
                elif agg_func == 'max':
                    row[f'{col}_Max'] = np.max(values)
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def compute_auc(history: List[float]) -> float:
    """Anytime performance: AUC under log(1+|f(x)|) curve."""
    if not history:
        return float('inf')
    
    h = np.array([abs(v) if v is not None else np.nan for v in history], dtype=float)
    h = np.nan_to_num(h, nan=(np.nanmax(h) if np.isfinite(np.nanmax(h)) else 1.0))
    
    x = np.arange(1, len(h) + 1, dtype=float)
    x = x / x[-1]
    y = np.log1p(h)
    
    return float(np.trapz(y, x))


def friedman_test(results: Dict[str, List[float]]) -> Tuple[float, float]:
    """Friedman test for multiple algorithms."""
    data = [results[algo] for algo in sorted(results.keys())]
    statistic, p_value = stats.friedmanchisquare(*data)
    return float(statistic), float(p_value)


def compute_ranks(results: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute average ranks across runs."""
    algorithms = sorted(results.keys())
    n_runs = len(results[algorithms[0]])
    
    ranks = {algo: 0.0 for algo in algorithms}
    
    for run_idx in range(n_runs):
        run_values = [(algo, results[algo][run_idx]) for algo in algorithms]
        run_values.sort(key=lambda x: x[1])
        
        for rank, (algo, _) in enumerate(run_values, 1):
            ranks[algo] += rank
    
    for algo in algorithms:
        ranks[algo] /= n_runs
    
    return ranks


def generate_rastrigin_global_ranks(
    summary_df: pd.DataFrame,
    metric: str = 'AUC_Median'
) -> pd.DataFrame:
    """Compute global average ranks for Rastrigin."""
    grouped = summary_df.groupby(['Configuration', 'Scenario'])
    algo_metrics = defaultdict(list)
    
    for (config, scenario), group in grouped:
        valid = group.dropna(subset=[metric])
        if len(valid) < 2:
            continue
        
        sorted_algos = valid.sort_values(metric)['Algorithm'].tolist()
        ranks = {algo: rank + 1 for rank, algo in enumerate(sorted_algos)}
        
        for algo, rank in ranks.items():
            algo_metrics[algo].append(rank)
    
    avg_ranks = {algo: float(np.mean(rs)) for algo, rs in algo_metrics.items()}
    
    rank_df = pd.DataFrame([
        {'Algorithm': a, 'Avg_Rank': r, 'N_Configs': len(algo_metrics[a])}
        for a, r in sorted(avg_ranks.items(), key=lambda x: x[1])
    ])
    
    print(f"\nGlobal Rastrigin Ranks (by {metric}):")
    print(rank_df.to_string(index=False))
    
    return rank_df


def generate_knapsack_global_ranks(
    summary_df: pd.DataFrame,
    metric: str = 'Mean_Gap_%'
) -> pd.DataFrame:
    """Compute global average ranks for Knapsack."""
    # Group by all instance columns + scenario
    instance_cols = [col for col in ['N_Items', 'Instance_Type', 'Instance_Seed'] if col in summary_df.columns]
    group_cols = instance_cols + ['Scenario']
    
    grouped = summary_df.groupby(group_cols)
    algo_metrics = defaultdict(list)
    
    for group_key, group in grouped:
        valid = group.dropna(subset=[metric])
        if len(valid) < 2:
            continue
        
        sorted_algos = valid.sort_values(metric)['Algorithm'].tolist()
        ranks = {algo: rank + 1 for rank, algo in enumerate(sorted_algos)}
        
        for algo, rank in ranks.items():
            algo_metrics[algo].append(rank)
    
    avg_ranks = {algo: float(np.mean(rs)) for algo, rs in algo_metrics.items()}
    
    rank_df = pd.DataFrame([
        {'Algorithm': a, 'Avg_Rank': r, 'N_Configs': len(algo_metrics[a])}
        for a, r in sorted(avg_ranks.items(), key=lambda x: x[1])
    ])
    
    print(f"\nGlobal Knapsack Ranks (by {metric}):")
    print(rank_df.to_string(index=False))
    
    return rank_df


# ============================================================================
# MAIN FUNCTION WITH CLI (UPDATED)
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive benchmark analysis with unified data loading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Rastrigin results
  python benchmark/analyze_results.py --problem rastrigin
  
  # Analyze Knapsack (creates both by_instance and by_type summaries)
  python benchmark/analyze_results.py --problem knapsack
  
  # Analyze both problems
  python benchmark/analyze_results.py --problem all --output-dir benchmark/results/summaries
        """
    )
    
    parser.add_argument(
        '--problem', type=str, required=True,
        choices=['rastrigin', 'knapsack', 'all'],
        help='Which problem to analyze'
    )
    parser.add_argument(
        '--results-dir', type=str,
        default='benchmark/results',
        help='Directory containing result JSON files'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='benchmark/results/summaries',
        help='Directory to save summary CSV files'
    )
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("UNIFIED BENCHMARK ANALYSIS")
    print("=" * 100)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Problem: {args.problem}")
    
    # CORE: Load all data into unified DataFrame
    print("\nLoading all results...")
    df = load_all_results_to_dataframe(args.results_dir)
    
    if df.empty:
        logger.error("No data loaded, exiting")
        return
    
    print(f"\nLoaded {len(df)} runs:")
    print(df.groupby(['Problem', 'Scenario', 'Algorithm']).size())
    
    output_dir = Path(args.output_dir)
    
    # Run analyses
    if args.problem in ['rastrigin', 'all']:
        analyze_rastrigin(df, output_dir=output_dir)
    
    if args.problem in ['knapsack', 'all']:
        analyze_knapsack(df, output_dir=output_dir)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()