"""
Modern academic-grade visualization using CSV summaries.
All plots are generated from pre-computed summary files, not raw JSON.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Algorithm colors and markers
COLORS = {'FA': '#FF6B6B', 'SA': '#4ECDC4', 'HC': '#45B7D1', 'GA': '#FFA07A'}
MARKERS = {'FA': 'o', 'SA': 's', 'HC': '^', 'GA': 'D'}
SCENARIO_MARKERS = {
    'out_of_the_box': 'o',
    'specialist': 's',
    'repair': '^',
    'penalty': 'D'
}


# ============================================================================
# DATA LOADING (UPDATED: Load Multiple Summaries)
# ============================================================================

def load_summary_data(summary_path: Path) -> Optional[pd.DataFrame]:
    """
    Load summary CSV file.
    
    Parameters
    ----------
    summary_path : Path
        Path to CSV summary file
    
    Returns
    -------
    pd.DataFrame or None
        Loaded dataframe, or None if file not found
    """
    if not summary_path.exists():
        logger.warning(f"Summary file not found: {summary_path}")
        return None
    
    try:
        df = pd.read_csv(summary_path)
        logger.info(f"Loaded {len(df)} rows from {summary_path.name}")
        return df
    except Exception as e:
        logger.error(f"Error loading {summary_path}: {e}")
        return None


def load_summary(name: str, summary_dir: Path = Path('benchmark/results/summaries')) -> Optional[pd.DataFrame]:
    """
    Flexible loader for any summary CSV file.
    
    Parameters
    ----------
    name : str
        Name of the summary file (without .csv extension)
    summary_dir : Path
        Directory containing summary files
    
    Returns
    -------
    pd.DataFrame or None
        Loaded dataframe, or None if file not found
    """
    filepath = summary_dir / f'{name}.csv'
    return load_summary_data(filepath)


def load_available_summaries(summary_dir: Path, prefix: str) -> Dict[str, pd.DataFrame]:
    """
    Load all available summary files with given prefix.
    
    Parameters
    ----------
    summary_dir : Path
        Directory containing summary CSVs
    prefix : str
        Prefix of summary files (e.g., 'knapsack_summary')
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping summary type to DataFrame:
        - 'by_instance': Instance-level summary
        - 'by_type': Type-level summary
    """
    summaries = {}
    
    # Try to load by_instance
    instance_path = summary_dir / f'{prefix}_by_instance.csv'
    if instance_path.exists():
        try:
            df = pd.read_csv(instance_path)
            summaries['by_instance'] = df
            logger.info(f"Loaded {len(df)} rows from {instance_path.name}")
        except Exception as e:
            logger.error(f"Error loading {instance_path}: {e}")
    
    # Try to load by_type
    type_path = summary_dir / f'{prefix}_by_type.csv'
    if type_path.exists():
        try:
            df = pd.read_csv(type_path)
            summaries['by_type'] = df
            logger.info(f"Loaded {len(df)} rows from {type_path.name}")
        except Exception as e:
            logger.error(f"Error loading {type_path}: {e}")
    
    return summaries


def load_rastrigin_artifacts(summary_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all Rastrigin analysis artifacts from CSV files.
    
    Parameters
    ----------
    summary_dir : Path
        Directory containing CSV summary files
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping artifact type to DataFrame:
        - 'summary': Main summary statistics
        - 'fixed_target_ecdf': ECDF of runtime-to-target
        - 'ert': Expected Running Time
        - 'fixed_budget': Fixed-budget performance
        - 'performance_profiles': Performance profiles (Dolan-Moré)
        - 'data_profiles': Data profiles (Moré-Wild)
        - 'diversity': Diversity metrics
        - 'stagnation': Stagnation analysis
    """
    artifacts = {}
    
    file_mappings = {
        'summary': 'rastrigin_summary.csv',
        'fixed_target_ecdf': 'rastrigin_fixed_target_ecdf.csv',
        'ert': 'rastrigin_ert.csv',
        'fixed_budget': 'rastrigin_fixed_budget.csv',
        'performance_profiles': 'rastrigin_performance_profiles.csv',
        'data_profiles': 'rastrigin_data_profiles.csv',
        'diversity': 'rastrigin_diversity_summary.csv',
        'stagnation': 'rastrigin_stagnation.csv'
    }
    
    for key, filename in file_mappings.items():
        filepath = summary_dir / filename
        df = load_summary_data(filepath)
        if df is not None:
            artifacts[key] = df
            logger.info(f"  ✓ Loaded {key}: {len(df)} rows")
        else:
            logger.warning(f"  ✗ Missing {key} ({filename})")
    
    return artifacts


# ============================================================================
# RASTRIGIN VISUALIZATIONS (COCO/BBOB STANDARD)
# ============================================================================

def plot_rastrigin_fixed_target_ecdf(df: pd.DataFrame, output_dir: str):
    """
    Fixed-Target ECDF plots (COCO/BBOB standard).
    Shows empirical cumulative distribution of runtime-to-target.
    
    Small-multiples by Level (Gold/Silver/Bronze), lines for Algorithm,
    linestyle for Scenario.
    """
    if df.empty:
        return
    
    required_cols = ['Config', 'Level', 'Algorithm', 'tau', 'ECDF']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing required columns for ECDF plot: {required_cols}")
        return
    
    configs = sorted(df['Config'].unique())
    levels = sorted(df['Level'].unique())
    
    linestyles = {'out_of_the_box': '-', 'specialist': '--', 'repair': '-.', 'penalty': ':'}
    
    for config in configs:
        df_config = df[df['Config'] == config]
        
        fig, axes = plt.subplots(1, len(levels), figsize=(6 * len(levels), 5),
                                squeeze=False)
        axes = axes.flatten()
        
        for idx, level in enumerate(levels):
            ax = axes[idx]
            df_level = df_config[df_config['Level'] == level]
            
            if df_level.empty:
                ax.axis('off')
                continue
            
            for algo in sorted(df_level['Algorithm'].unique()):
                df_algo = df_level[df_level['Algorithm'] == algo]
                
                for scenario in sorted(df_algo['Scenario'].unique()):
                    df_plot = df_algo[df_algo['Scenario'] == scenario].sort_values('tau')
                    
                    if df_plot.empty:
                        continue
                    
                    ls = linestyles.get(scenario, '-')
                    label = f'{algo}' if scenario == 'out_of_the_box' else f'{algo} ({scenario})'
                    
                    ax.plot(df_plot['tau'], df_plot['ECDF'], 
                           label=label, color=COLORS.get(algo, 'gray'),
                           linestyle=ls, linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Runtime (evaluations)', fontsize=11)
            ax.set_ylabel('ECDF' if idx == 0 else '', fontsize=11)
            ax.set_title(f'{level}', fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='lower right')
            ax.set_ylim([0, 1.05])
        
        plt.suptitle(f'Rastrigin Fixed-Target ECDF - {config.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'rastrigin_ecdf_{config}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path}")


def plot_rastrigin_ert(df: pd.DataFrame, output_dir: str, level: str = 'Silver'):
    """
    Expected Running Time (ERT) bar plots with confidence intervals.
    
    ERT = expected number of evaluations to reach target (includes failed runs).
    Standard metric in COCO/BBOB benchmarking.
    """
    if df.empty:
        return
    
    required_cols = ['Config', 'Level', 'Algorithm', 'ERT', 'CI_low', 'CI_high']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing required columns for ERT plot: {required_cols}")
        return
    
    df_level = df[df['Level'] == level].copy()
    
    if df_level.empty:
        logger.warning(f"No ERT data for level: {level}")
        return
    
    configs = sorted(df_level['Config'].unique())
    scenarios = sorted(df_level['Scenario'].unique()) if 'Scenario' in df_level.columns else ['all']
    
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(6 * n_scenarios, 5),
                             squeeze=False)
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        
        if scenario != 'all':
            df_scenario = df_level[df_level['Scenario'] == scenario]
        else:
            df_scenario = df_level
        
        if df_scenario.empty:
            ax.axis('off')
            continue
        
        # Group by config and algorithm
        algos = sorted(df_scenario['Algorithm'].unique())
        x = np.arange(len(configs))
        width = 0.8 / len(algos)
        
        for i, algo in enumerate(algos):
            df_algo = df_scenario[df_scenario['Algorithm'] == algo]
            
            erts = []
            ci_lows = []
            ci_highs = []
            
            for config in configs:
                row = df_algo[df_algo['Config'] == config]
                if not row.empty and pd.notna(row['ERT'].values[0]):
                    erts.append(row['ERT'].values[0])
                    ci_lows.append(row['CI_low'].values[0] if pd.notna(row['CI_low'].values[0]) else row['ERT'].values[0])
                    ci_highs.append(row['CI_high'].values[0] if pd.notna(row['CI_high'].values[0]) else row['ERT'].values[0])
                else:
                    erts.append(0)
                    ci_lows.append(0)
                    ci_highs.append(0)
            
            # Calculate error bars
            yerr_low = [ert - ci_low for ert, ci_low in zip(erts, ci_lows)]
            yerr_high = [ci_high - ert for ert, ci_high in zip(erts, ci_highs)]
            
            ax.bar(x + i * width, erts, width, label=algo,
                  color=COLORS.get(algo, 'gray'), alpha=0.8,
                  edgecolor='black', linewidth=0.5,
                  yerr=[yerr_low, yerr_high], capsize=3)
        
        ax.set_xlabel('Configuration', fontsize=11)
        ax.set_ylabel('ERT (evaluations)' if idx == 0 else '', fontsize=11)
        ax.set_title(f'{scenario.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (len(algos) - 1) / 2)
        ax.set_xticklabels([c.replace('_', '\n') for c in configs], fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
    
    plt.suptitle(f'Rastrigin Expected Running Time (ERT) - {level} Target',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / f'rastrigin_ert_{level.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path}")


def plot_rastrigin_fixed_budget(df: pd.DataFrame, output_dir: str):
    """
    Fixed-Budget curves showing error achieved at different budget fractions.
    
    X-axis: Budget (evaluations), Y-axis: Median error to optimum.
    Shows anytime performance across the optimization run.
    """
    if df.empty:
        return
    
    # Extract budget fraction columns (e.g., Error@10%, Error@30%, etc.)
    error_cols = [col for col in df.columns if col.startswith('Error@') and col.endswith('%')]
    
    if not error_cols:
        logger.warning("No Error@X% columns found for fixed-budget plot")
        return
    
    configs = sorted(df['Config'].unique()) if 'Config' in df.columns else ['default']
    
    for config in configs:
        df_config = df[df['Config'] == config] if 'Config' in df.columns else df
        
        scenarios = sorted(df_config['Scenario'].unique()) if 'Scenario' in df_config.columns else ['all']
        
        fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5),
                                squeeze=False)
        axes = axes.flatten()
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            
            if scenario != 'all':
                df_plot = df_config[df_config['Scenario'] == scenario]
            else:
                df_plot = df_config
            
            if df_plot.empty:
                ax.axis('off')
                continue
            
            # Extract budget points from column names
            budget_fracs = [int(col.split('@')[1].replace('%', '')) for col in error_cols]
            
            for algo in sorted(df_plot['Algorithm'].unique()):
                df_algo = df_plot[df_plot['Algorithm'] == algo]
                
                # Calculate median error at each budget point
                median_errors = []
                for col in error_cols:
                    median_errors.append(df_algo[col].median())
                
                # Convert to evaluations (assuming max budget is in metadata)
                # For now, use relative scale
                ax.plot(budget_fracs, median_errors, label=algo,
                       color=COLORS.get(algo, 'gray'), linewidth=2.5,
                       marker=MARKERS.get(algo, 'o'), markersize=8)
            
            ax.set_xlabel('Budget (%)', fontsize=11)
            ax.set_ylabel('Median Error' if idx == 0 else '', fontsize=11)
            ax.set_title(f'{scenario.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_yscale('log')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Rastrigin Fixed-Budget Performance - {config.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'rastrigin_fixed_budget_{config}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path}")


def plot_rastrigin_performance_profile(df: pd.DataFrame, output_dir: str):
    """
    Performance Profiles (Dolan-Moré 2002).
    
    Shows fraction of problems solved within τ × best time.
    Standard for comparing optimizer robustness.
    """
    if df.empty:
        return
    
    required_cols = ['Algorithm', 'tau', 'phi']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing required columns for performance profile: {required_cols}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in sorted(df['Algorithm'].unique()):
        df_algo = df[df['Algorithm'] == algo].sort_values('tau')
        
        ax.plot(df_algo['tau'], df_algo['phi'], label=algo,
               color=COLORS.get(algo, 'gray'), linewidth=2.5,
               marker=MARKERS.get(algo, 'o'), markersize=4, markevery=10)
    
    ax.set_xlabel(r'$\tau$ (performance ratio)', fontsize=12)
    ax.set_ylabel(r'Fraction of problems solved within $\tau \times$ best', fontsize=12)
    ax.set_title('Rastrigin Performance Profile (Dolan-Moré)', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'rastrigin_perf_profile.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path}")


def plot_rastrigin_data_profile(df: pd.DataFrame, output_dir: str):
    """
    Data Profiles (Moré-Wild 2009).
    
    Shows fraction of problems solved as function of budget.
    Complements performance profiles by showing budget-quality tradeoff.
    """
    if df.empty:
        return
    
    required_cols = ['Algorithm', 'nu', 'psi']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing required columns for data profile: {required_cols}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in sorted(df['Algorithm'].unique()):
        df_algo = df[df['Algorithm'] == algo].sort_values('nu')
        
        ax.plot(df_algo['nu'], df_algo['psi'], label=algo,
               color=COLORS.get(algo, 'gray'), linewidth=2.5,
               marker=MARKERS.get(algo, 'o'), markersize=4, markevery=5)
    
    ax.set_xlabel(r'Budget $\nu$ (evaluations)', fontsize=12)
    ax.set_ylabel(r'$\psi(\nu)$ (fraction of problems solved)', fontsize=12)
    ax.set_title('Rastrigin Data Profile (Moré-Wild)', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'rastrigin_data_profile.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path}")


def plot_rastrigin_diversity_panel(df: pd.DataFrame, output_dir: str):
    """
    Diversity metrics panel (normalized by √D).
    
    Shows population diversity evolution: Initial, Mid-point, Final, and Drop.
    Critical for diagnosing premature convergence in swarm algorithms.
    """
    if df.empty:
        return
    
    div_cols = ['Div_Norm_Initial', 'Div_Norm_Mid50', 'Div_Norm_Final', 'Div_Norm_Drop']
    available_cols = [col for col in div_cols if col in df.columns]
    
    if not available_cols:
        logger.warning("No diversity columns found")
        return
    
    configs = sorted(df['Configuration'].unique()) if 'Configuration' in df.columns else ['default']
    
    for config in configs:
        df_config = df[df['Configuration'] == config] if 'Configuration' in df.columns else df
        
        scenarios = sorted(df_config['Scenario'].unique()) if 'Scenario' in df_config.columns else ['all']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, div_col in enumerate(available_cols):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            data_to_plot = []
            labels = []
            
            for scenario in scenarios:
                if scenario != 'all':
                    df_scenario = df_config[df_config['Scenario'] == scenario]
                else:
                    df_scenario = df_config
                
                for algo in sorted(df_scenario['Algorithm'].unique()):
                    df_algo = df_scenario[df_scenario['Algorithm'] == algo]
                    values = df_algo[div_col].dropna().values
                    
                    if len(values) > 0:
                        data_to_plot.append(values)
                        labels.append(f'{algo}\n{scenario}' if scenario != 'all' else algo)
            
            if not data_to_plot:
                ax.axis('off')
                continue
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
            
            # Color by algorithm
            for patch, label in zip(bp['boxes'], labels):
                algo = label.split('\n')[0]
                patch.set_facecolor(COLORS.get(algo, 'gray'))
                patch.set_alpha(0.7)
            
            metric_name = div_col.replace('Div_Norm_', '').replace('50', ' (50%)')
            ax.set_title(f'Diversity: {metric_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Normalized Diversity / √D', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Hide unused subplots
        for idx in range(len(available_cols), 4):
            axes[idx].axis('off')
        
        plt.suptitle(f'Rastrigin Diversity Analysis - {config.replace("_", " ").title()}\n(Normalized by √D)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'rastrigin_diversity_{config}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path}")


def plot_rastrigin_stagnation(df: pd.DataFrame, output_dir: str):
    """
    Stagnation analysis: longest period without improvement.
    
    Shows histograms and cumulative proportions of stagnation lengths.
    Helps identify premature convergence and search stalling.
    """
    if df.empty or 'Stagnation_Gens' not in df.columns:
        logger.warning("No stagnation data found")
        return
    
    configs = sorted(df['Configuration'].unique()) if 'Configuration' in df.columns else ['default']
    
    for config in configs:
        df_config = df[df['Configuration'] == config] if 'Configuration' in df.columns else df
        
        scenarios = sorted(df_config['Scenario'].unique()) if 'Scenario' in df_config.columns else ['all']
        algos = sorted(df_config['Algorithm'].unique())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Histogram
        ax_hist = axes[0]
        
        for algo in algos:
            df_algo = df_config[df_config['Algorithm'] == algo]
            stag_values = df_algo['Stagnation_Gens'].dropna().values
            
            if len(stag_values) > 0:
                ax_hist.hist(stag_values, bins=20, alpha=0.6, label=algo,
                            color=COLORS.get(algo, 'gray'), edgecolor='black')
        
        ax_hist.set_xlabel('Stagnation Length (generations)', fontsize=11)
        ax_hist.set_ylabel('Frequency', fontsize=11)
        ax_hist.set_title('Stagnation Length Distribution', fontsize=12, fontweight='bold')
        ax_hist.legend(fontsize=10)
        ax_hist.grid(True, alpha=0.3, axis='y')
        
        # Right: ECDF
        ax_ecdf = axes[1]
        
        for algo in algos:
            df_algo = df_config[df_config['Algorithm'] == algo]
            stag_values = np.sort(df_algo['Stagnation_Gens'].dropna().values)
            
            if len(stag_values) > 0:
                ecdf = np.arange(1, len(stag_values) + 1) / len(stag_values)
                ax_ecdf.plot(stag_values, ecdf, label=algo,
                            color=COLORS.get(algo, 'gray'), linewidth=2.5,
                            marker=MARKERS.get(algo, 'o'), markersize=4, markevery=5)
        
        ax_ecdf.set_xlabel('Stagnation Length (generations)', fontsize=11)
        ax_ecdf.set_ylabel('ECDF', fontsize=11)
        ax_ecdf.set_title('Cumulative Stagnation Distribution', fontsize=12, fontweight='bold')
        ax_ecdf.legend(fontsize=10)
        ax_ecdf.grid(True, alpha=0.3)
        ax_ecdf.set_ylim([0, 1.05])
        
        plt.suptitle(f'Rastrigin Stagnation Analysis - {config.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'rastrigin_stagnation_{config}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path}")


def plot_rastrigin_anytime_auc(df: pd.DataFrame, output_dir: str):
    """
    Anytime AUC (Area Under Curve) comparison.
    
    AUC measures integrated performance over the entire run.
    Lower AUC = better anytime performance (steeper convergence).
    """
    if df.empty or 'AUC_Median' not in df.columns:
        logger.warning("No AUC_Median column found")
        return
    
    configs = sorted(df['Configuration'].unique()) if 'Configuration' in df.columns else ['default']
    
    for config in configs:
        df_config = df[df['Configuration'] == config] if 'Configuration' in df.columns else df
        
        scenarios = sorted(df_config['Scenario'].unique()) if 'Scenario' in df_config.columns else ['all']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algos = sorted(df_config['Algorithm'].unique())
        x = np.arange(len(algos))
        width = 0.8 / len(scenarios)
        
        for i, scenario in enumerate(scenarios):
            if scenario != 'all':
                df_scenario = df_config[df_config['Scenario'] == scenario]
            else:
                df_scenario = df_config
            
            values = []
            for algo in algos:
                row = df_scenario[df_scenario['Algorithm'] == algo]
                if not row.empty and pd.notna(row['AUC_Median'].values[0]):
                    values.append(row['AUC_Median'].values[0])
                else:
                    values.append(0)
            
            label = scenario.replace('_', ' ').title() if scenario != 'all' else 'All'
            ax.bar(x + i * width, values, width, label=label,
                  alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Median AUC (lower is better)', fontsize=12)
        ax.set_title(f'Rastrigin Anytime Performance (AUC) - {config.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
        ax.set_xticklabels(algos)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'rastrigin_anytime_auc_{config}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path}")


# ============================================================================
# RASTRIGIN VISUALIZATIONS (EXISTING - KEEP)
# ============================================================================

def plot_rastrigin_boxplots_by_scenario(df: pd.DataFrame, output_dir: str):
    """
    Boxplots of final error for each configuration and scenario.
    Separate subplot for each (config, scenario) pair.
    """
    if df.empty:
        return
    
    configs = sorted(df['Configuration'].unique())
    scenarios = sorted(df['Scenario'].unique())
    
    n_configs = len(configs)
    n_scenarios = len(scenarios)
    
    fig, axes = plt.subplots(n_scenarios, n_configs, 
                             figsize=(5 * n_configs, 4 * n_scenarios),
                             squeeze=False)
    
    for i, scenario in enumerate(scenarios):
        for j, config in enumerate(configs):
            ax = axes[i, j]
            
            subset = df[(df['Configuration'] == config) & (df['Scenario'] == scenario)]
            
            if subset.empty:
                ax.axis('off')
                continue
            
            # Pivot for boxplot
            algos = sorted(subset['Algorithm'].unique())
            data_to_plot = [subset[subset['Algorithm'] == algo]['Median_Error'].values 
                           for algo in algos]
            
            bp = ax.boxplot(data_to_plot, labels=algos, patch_artist=True,
                           showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
            
            for patch, algo in zip(bp['boxes'], algos):
                patch.set_facecolor(COLORS.get(algo, 'gray'))
                patch.set_alpha(0.7)
            
            ax.set_title(f'{config}\n{scenario}', fontsize=10, fontweight='bold')
            ax.set_ylabel('Error to Optimum' if j == 0 else '')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Rastrigin: Final Error Distribution by Configuration & Scenario',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'rastrigin_boxplots_by_scenario.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path}")


def plot_rastrigin_success_breakdown(df: pd.DataFrame, output_dir: str):
    """
    Multi-level success rate visualization (Gold, Silver, Bronze).
    Grouped bar chart for each configuration x scenario.
    """
    if df.empty:
        return
    
    # Check if multi-level columns exist
    level_cols = [col for col in df.columns if col.startswith('SR_')]
    if not level_cols:
        logger.warning("No success rate columns found")
        return
    
    configs = sorted(df['Configuration'].unique())
    
    for config in configs:
        df_config = df[df['Configuration'] == config]
        scenarios = sorted(df_config['Scenario'].unique())
        
        fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5),
                                squeeze=False)
        axes = axes.flatten()
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            subset = df_config[df_config['Scenario'] == scenario]
            
            if subset.empty:
                ax.axis('off')
                continue
            
            algos = sorted(subset['Algorithm'].unique())
            
            # Find all SR columns (e.g., SR_Gold_%, SR_Silver_%, SR_Bronze_%)
            sr_cols = [col for col in subset.columns if col.startswith('SR_') and col.endswith('_%')]
            
            if not sr_cols:
                ax.axis('off')
                continue
            
            x = np.arange(len(algos))
            width = 0.8 / len(sr_cols)
            
            for i, sr_col in enumerate(sorted(sr_cols)):
                level_name = sr_col.replace('SR_', '').replace('_%', '')
                values = [subset[subset['Algorithm'] == algo][sr_col].values[0] 
                         if len(subset[subset['Algorithm'] == algo]) > 0 else 0
                         for algo in algos]
                
                color_intensity = 0.4 + 0.4 * i / len(sr_cols)
                ax.bar(x + i * width, values, width, label=level_name,
                      color=plt.cm.Blues(color_intensity), alpha=0.8,
                      edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Algorithm', fontsize=11)
            ax.set_ylabel('Success Rate (%)' if idx == 0 else '', fontsize=11)
            ax.set_title(f'{scenario.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * (len(sr_cols) - 1) / 2)
            ax.set_xticklabels(algos)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 105])
        
        plt.suptitle(f'Rastrigin Multi-Level Success Rates - {config.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'rastrigin_success_breakdown_{config}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path}")


def plot_rastrigin_cross_scenario_comparison(df: pd.DataFrame, output_dir: str):
    """
    Cross-scenario comparison: grouped bar chart showing performance of each
    algorithm across different scenarios.
    """
    if df.empty or 'Scenario' not in df.columns:
        return
    
    configs = sorted(df['Configuration'].unique())
    
    for config in configs:
        df_config = df[df['Configuration'] == config]
        
        # Pivot table: Algorithm x Scenario
        pivot = df_config.pivot_table(
            values='Median_Error',
            index='Algorithm',
            columns='Scenario',
            aggfunc='mean'
        )
        
        if pivot.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pivot.plot(kind='bar', ax=ax, color=plt.cm.Set2.colors, alpha=0.8,
                  edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Median Error to Optimum', fontsize=12)
        ax.set_title(f'Rastrigin Cross-Scenario Comparison - {config.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(title='Scenario', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'rastrigin_cross_scenario_{config}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path}")


def plot_rastrigin_hitting_times(df: pd.DataFrame, output_dir: str):
    """
    Hitting time comparison across algorithms, configs, and scenarios.
    Uses first available hitting time column.
    """
    if df.empty:
        return
    
    # Find hitting time columns
    ht_cols = [col for col in df.columns if col.startswith('HT_Med_')]
    if not ht_cols:
        logger.warning("No hitting time columns found")
        return
    
    # Use first HT column
    ht_col = ht_cols[0]
    
    configs = sorted(df['Configuration'].unique())
    
    for config in configs:
        df_config = df[df['Configuration'] == config]
        scenarios = sorted(df_config['Scenario'].unique())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algos = sorted(df_config['Algorithm'].unique())
        x = np.arange(len(algos))
        width = 0.8 / len(scenarios)
        
        for i, scenario in enumerate(scenarios):
            subset = df_config[df_config['Scenario'] == scenario]
            
            values = []
            for algo in algos:
                row = subset[subset['Algorithm'] == algo]
                if not row.empty and pd.notna(row[ht_col].values[0]):
                    values.append(row[ht_col].values[0])
                else:
                    values.append(0)
            
            ax.bar(x + i * width, values, width, label=scenario.replace('_', ' ').title(),
                  alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Median Hitting Time (evaluations)', fontsize=12)
        ax.set_title(f'Rastrigin Hitting Times - {config.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
        ax.set_xticklabels(algos)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'rastrigin_hitting_times_{config}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path}")


# ============================================================================
# ADVANCED PLOTS (UPDATED: Fix Instance Column Detection)
# ============================================================================

def plot_performance_profile(
    df: pd.DataFrame,
    metric_col: str,
    output_file: str,
    minimize: bool = True,
    title: str = ''
):
    """Dolan–Moré performance profile from summary data."""
    if df.empty or metric_col not in df.columns:
        logger.warning(f"Cannot plot performance profile: {metric_col} not found")
        return
    
    algos = sorted(df['Algorithm'].unique())
    
    # Identify instance columns (FIXED FOR KNAPSACK)
    instance_cols = []
    
    # For Rastrigin
    if 'Configuration' in df.columns:
        instance_cols.append('Configuration')
    
    # For Knapsack (instance-level)
    if 'N_Items' in df.columns:
        instance_cols.append('N_Items')
    if 'Instance_Type' in df.columns:
        instance_cols.append('Instance_Type')
    if 'Instance_Seed' in df.columns:
        instance_cols.append('Instance_Seed')
    
    # Scenario is always included if present
    if 'Scenario' in df.columns:
        instance_cols.append('Scenario')
    
    if not instance_cols:
        logger.warning("Cannot identify instance grouping columns for Performance Profile")
        return
    
    logger.info(f"Performance Profile grouping by: {instance_cols}")
    
    # Group by instances
    perf_ratios = {algo: [] for algo in algos}
    
    for _, group in df.groupby(instance_cols):
        valid = group.dropna(subset=[metric_col])
        if len(valid) < 1:
            continue
        
        if minimize:
            best = valid[metric_col].min()
        else:
            best = valid[metric_col].max()
        
        if best == 0:
            continue  # Skip if best is zero (would cause division by zero)
        
        for _, row in valid.iterrows():
            algo = row['Algorithm']
            metric_val = row[metric_col]
            
            if minimize:
                ratio = metric_val / best if best > 0 else 1.0
            else:
                ratio = best / metric_val if metric_val > 0 else 1.0
            
            perf_ratios[algo].append(ratio)
    
    # Plot CDF
    fig, ax = plt.subplots(figsize=(10, 6))
    
    taus = np.linspace(1.0, 5.0, 500)
    
    for algo in algos:
        ratios = np.array(perf_ratios[algo])
        if len(ratios) == 0:
            logger.warning(f"No performance ratios for {algo}")
            continue
        
        ys = [np.mean(ratios <= tau) for tau in taus]
        
        ax.plot(taus, ys, label=algo, color=COLORS.get(algo, 'gray'),
                linewidth=2.5, marker=MARKERS.get(algo, 'o'), markersize=3, markevery=50)
    
    ax.set_xlabel(r'$\tau$ (performance ratio)', fontsize=12)
    ax.set_ylabel('Fraction of instances solved within $\\tau \\times$ best', fontsize=12)
    ax.set_title(title or f'Performance Profile ({metric_col})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


# ============================================================================
# KNAPSACK VISUALIZATIONS - DOLAN-MORÉ & MORÉ-WILD STANDARD
# ============================================================================

def plot_global_ranks(df: pd.DataFrame, output_file: str, title: str = ''):
    """Plot global average ranks from ranks CSV."""
    if df is None or df.empty:
        logger.warning("No ranks data available")
        return
    
    if 'Algorithm' not in df.columns or 'Avg_Rank' not in df.columns:
        logger.warning("Missing required columns for global ranks plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    algos = df['Algorithm'].values
    ranks = df['Avg_Rank'].values
    colors = [COLORS.get(a, 'gray') for a in algos]
    
    bars = ax.bar(algos, ranks, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, rank in zip(bars, ranks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{rank:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Average Rank (lower is better)', fontsize=12)
    ax.set_title(title or 'Global Algorithm Ranks', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(ranks) * 1.15])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


def plot_knapsack_performance_profiles_dolan_more(df: pd.DataFrame, output_dir: str):
    """
    Performance Profiles (Dolan-Moré 2002) for Knapsack.
    
    Shows φ(τ) = fraction of problems where r_ps ≤ τ
    Standard metric for comparing algorithm robustness across instances.
    
    NOTE: With multi-tier gap thresholds (Bronze/Silver/Gold), we can generate
    separate profiles for each tier, showing how algorithm performance varies
    with target quality requirements.
    
    Reference: https://arxiv.org/abs/cs/0102001
    """
    if df is None or df.empty:
        logger.warning("No performance profile data available")
        return
    
    required_cols = ['Algorithm', 'tau', 'phi']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing required columns for Performance Profile: {required_cols}")
        return
    
    # Check for faceting columns
    has_scenario = 'Scenario' in df.columns
    has_gap_threshold = 'Gap_Threshold' in df.columns
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Overall plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in sorted(df['Algorithm'].unique()):
        df_algo = df[df['Algorithm'] == algo].sort_values('tau')
        
        ax.plot(df_algo['tau'], df_algo['phi'], label=algo,
               color=COLORS.get(algo, 'gray'), linewidth=2.5,
               marker=MARKERS.get(algo, 'o'), markersize=4, markevery=10)
    
    ax.set_xlabel(r'$\tau$ (performance ratio)', fontsize=12)
    ax.set_ylabel(r'$\varphi(\tau)$ = Fraction of problems solved within $\tau \times$ best', fontsize=12)
    ax.set_title('Knapsack Performance Profile (Dolan-Moré)\nφ(τ) = fraction{ r_ps ≤ τ }',
                fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Footer note
    fig.text(0.5, 0.01, 'Reference: Dolan & Moré (2002) - arXiv:cs/0102001',
            ha='center', fontsize=8, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path / 'knapsack_performance_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path / 'knapsack_performance_profiles.png'}")
    
    # Faceted plots by Scenario
    if has_scenario:
        scenarios = sorted(df['Scenario'].unique())
        n_scenarios = len(scenarios)
        
        fig, axes = plt.subplots(1, n_scenarios, figsize=(6 * n_scenarios, 5), squeeze=False)
        axes = axes.flatten()
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            df_scenario = df[df['Scenario'] == scenario]
            
            for algo in sorted(df_scenario['Algorithm'].unique()):
                df_algo = df_scenario[df_scenario['Algorithm'] == algo].sort_values('tau')
                
                ax.plot(df_algo['tau'], df_algo['phi'], label=algo,
                       color=COLORS.get(algo, 'gray'), linewidth=2.5,
                       marker=MARKERS.get(algo, 'o'), markersize=4, markevery=10)
            
            ax.set_xlabel(r'$\tau$ (performance ratio)', fontsize=11)
            ax.set_ylabel(r'$\varphi(\tau)$' if idx == 0 else '', fontsize=11)
            ax.set_title(f'{scenario.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        
        plt.suptitle('Knapsack Performance Profiles by Scenario', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'knapsack_performance_profiles_by_scenario.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path / 'knapsack_performance_profiles_by_scenario.png'}")


def plot_knapsack_data_profiles_more_wild(df: pd.DataFrame, output_dir: str):
    """
    Data Profiles (Moré-Wild 2009) for Knapsack.
    
    Shows ψ(ν) = fraction of problems solved within ν evaluations.
    Complements performance profiles by showing budget-quality tradeoff.
    
    Reference: https://epubs.siam.org/doi/10.1137/080724083
    """
    if df is None or df.empty:
        logger.warning("No data profile data available")
        return
    
    required_cols = ['Algorithm', 'nu', 'psi']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing required columns for Data Profile: {required_cols}")
        return
    
    has_scenario = 'Scenario' in df.columns
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Overall plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in sorted(df['Algorithm'].unique()):
        df_algo = df[df['Algorithm'] == algo].sort_values('nu')
        
        ax.plot(df_algo['nu'], df_algo['psi'], label=algo,
               color=COLORS.get(algo, 'gray'), linewidth=2.5,
               marker=MARKERS.get(algo, 'o'), markersize=4, markevery=5)
    
    ax.set_xlabel(r'Budget $\nu$ (evaluations)', fontsize=12)
    ax.set_ylabel(r'$\psi(\nu)$ = Fraction of problems solved within $\nu$ evals', fontsize=12)
    ax.set_title('Knapsack Data Profile (Moré-Wild)\nψ(ν) = fraction solved within ν evaluations',
                fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Footer note
    fig.text(0.5, 0.01, 'Reference: Moré & Wild (2009) - SIAM J. Optim.',
            ha='center', fontsize=8, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path / 'knapsack_data_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path / 'knapsack_data_profiles.png'}")
    
    # Faceted by Scenario
    if has_scenario:
        scenarios = sorted(df['Scenario'].unique())
        n_scenarios = len(scenarios)
        
        fig, axes = plt.subplots(1, n_scenarios, figsize=(6 * n_scenarios, 5), squeeze=False)
        axes = axes.flatten()
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            df_scenario = df[df['Scenario'] == scenario]
            
            for algo in sorted(df_scenario['Algorithm'].unique()):
                df_algo = df_scenario[df_scenario['Algorithm'] == algo].sort_values('nu')
                
                ax.plot(df_algo['nu'], df_algo['psi'], label=algo,
                       color=COLORS.get(algo, 'gray'), linewidth=2.5,
                       marker=MARKERS.get(algo, 'o'), markersize=4, markevery=5)
            
            ax.set_xlabel(r'Budget $\nu$ (evaluations)', fontsize=11)
            ax.set_ylabel(r'$\psi(\nu)$' if idx == 0 else '', fontsize=11)
            ax.set_title(f'{scenario.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        
        plt.suptitle('Knapsack Data Profiles by Scenario', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'knapsack_data_profiles_by_scenario.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path / 'knapsack_data_profiles_by_scenario.png'}")


def plot_knapsack_fixed_budget_analysis(df: pd.DataFrame, output_dir: str, summary_dir: str):
    """
    Fixed-Budget analysis for Knapsack.
    
    Auto-detects whether we have DP-optimal (Gap@x%) or not (BestValue@x%).
    Generates boxplots and summary tables for each budget milestone.
    """
    if df is None or df.empty:
        logger.warning("No fixed-budget data available")
        return
    
    output_path = Path(output_dir)
    summary_path = Path(summary_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect metric type
    gap_cols = [col for col in df.columns if col.startswith('Gap@') and col.endswith('%')]
    value_cols = [col for col in df.columns if col.startswith('BestValue@') and col.endswith('%')]
    
    if gap_cols:
        metric_cols = gap_cols
        metric_type = 'Gap'
        ylabel = 'Gap to Optimal (%)'
    elif value_cols:
        metric_cols = value_cols
        metric_type = 'BestValue'
        ylabel = 'Best Value Found'
    else:
        logger.warning("No Gap@x% or BestValue@x% columns found in fixed-budget data")
        return
    
    logger.info(f"Fixed-budget analysis using metric: {metric_type}")
    
    has_n_items = 'N_Items' in df.columns
    
    # Summary statistics
    summary_data = []
    
    for col in metric_cols:
        budget_pct = col.split('@')[1].replace('%', '')
        
        # Boxplot
        if has_n_items:
            sizes = sorted(df['N_Items'].unique())
            fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 5), squeeze=False)
            axes = axes.flatten()
            
            for idx, n_items in enumerate(sizes):
                ax = axes[idx]
                df_n = df[df['N_Items'] == n_items]
                
                algos = sorted(df_n['Algorithm'].unique())
                data_to_plot = [df_n[df_n['Algorithm'] == algo][col].dropna().values for algo in algos]
                
                bp = ax.boxplot(data_to_plot, labels=algos, patch_artist=True,
                               showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
                
                for patch, algo in zip(bp['boxes'], algos):
                    patch.set_facecolor(COLORS.get(algo, 'gray'))
                    patch.set_alpha(0.7)
                
                ax.set_ylabel(ylabel if idx == 0 else '', fontsize=11)
                ax.set_title(f'N={n_items}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                if metric_type == 'Gap':
                    ax.set_yscale('log')
            
            plt.suptitle(f'Knapsack Fixed-Budget {metric_type} @ {budget_pct}%',
                        fontsize=14, fontweight='bold')
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            algos = sorted(df['Algorithm'].unique())
            data_to_plot = [df[df['Algorithm'] == algo][col].dropna().values for algo in algos]
            
            bp = ax.boxplot(data_to_plot, labels=algos, patch_artist=True,
                           showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
            
            for patch, algo in zip(bp['boxes'], algos):
                patch.set_facecolor(COLORS.get(algo, 'gray'))
                patch.set_alpha(0.7)
            
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'Knapsack Fixed-Budget {metric_type} @ {budget_pct}%',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            if metric_type == 'Gap':
                ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_path / f'knapsack_fixed_budget_{metric_type}_{budget_pct}pct.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path / f'knapsack_fixed_budget_{metric_type}_{budget_pct}pct.png'}")
        
        # Compute summary statistics
        for algo in df['Algorithm'].unique():
            df_algo = df[df['Algorithm'] == algo]
            
            if has_n_items:
                for n_items in df_algo['N_Items'].unique():
                    df_subset = df_algo[df_algo['N_Items'] == n_items]
                    values = df_subset[col].dropna()
                    
                    if len(values) > 0:
                        summary_data.append({
                            'Algorithm': algo,
                            'N_Items': n_items,
                            'Budget_%': budget_pct,
                            'Metric': metric_type,
                            'Median': values.median(),
                            'Q25': values.quantile(0.25),
                            'Q75': values.quantile(0.75),
                            'IQR': values.quantile(0.75) - values.quantile(0.25),
                            'Mean': values.mean(),
                            'Std': values.std(),
                            'N': len(values)
                        })
            else:
                values = df_algo[col].dropna()
                
                if len(values) > 0:
                    summary_data.append({
                        'Algorithm': algo,
                        'Budget_%': budget_pct,
                        'Metric': metric_type,
                        'Median': values.median(),
                        'Q25': values.quantile(0.25),
                        'Q75': values.quantile(0.75),
                        'IQR': values.quantile(0.75) - values.quantile(0.25),
                        'Mean': values.mean(),
                        'Std': values.std(),
                        'N': len(values)
                    })
    
    # Save summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = summary_path / f'knapsack_fixed_budget_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"✓ Saved: {summary_file}")


def plot_knapsack_pairwise_stats_heatmap(df: pd.DataFrame, output_dir: str, summary_dir: str):
    """
    Pairwise comparison heatmap and Copeland ranking.
    
    Uses pre-computed Wilcoxon + Holm + A12 + Cliff's Delta from pairwise_stats.
    
    Reference: https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    """
    if df is None or df.empty:
        logger.warning("No pairwise stats data available")
        return
    
    required_cols = ['Algo1', 'Algo2', 'p_adj', 'Significant', 'A12']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing required columns for pairwise stats: {required_cols}")
        return
    
    output_path = Path(output_dir)
    summary_path = Path(summary_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path.mkdir(parents=True, exist_ok=True)
    
    has_scenario = 'Scenario' in df.columns
    
    # Filter only significant comparisons
    df_sig = df[df['Significant'] == True].copy()
    
    if df_sig.empty:
        logger.warning("No significant pairwise differences found")
        return
    
    algos = sorted(set(df['Algo1'].unique()) | set(df['Algo2'].unique()))
    
    # Overall win-loss matrix
    win_matrix = pd.DataFrame(0, index=algos, columns=algos)
    a12_matrix = pd.DataFrame(np.nan, index=algos, columns=algos)
    
    for _, row in df_sig.iterrows():
        # Determine winner (higher A12 means Algo1 is better)
        if row['A12'] > 0.5:
            winner, loser = row['Algo1'], row['Algo2']
            a12_val = row['A12']
        else:
            winner, loser = row['Algo2'], row['Algo1']
            a12_val = 1.0 - row['A12']
        
        win_matrix.loc[winner, loser] += 1
        
        # Store median A12 for annotation
        if pd.isna(a12_matrix.loc[winner, loser]):
            a12_matrix.loc[winner, loser] = a12_val
        else:
            a12_matrix.loc[winner, loser] = (a12_matrix.loc[winner, loser] + a12_val) / 2
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(win_matrix, annot=True, fmt='d', cmap='RdYlGn', center=0,
                cbar_kws={'label': '# Significant Wins'},
                linewidths=0.5, linecolor='gray', ax=ax)
    
    ax.set_xlabel('Loser', fontsize=12)
    ax.set_ylabel('Winner', fontsize=12)
    ax.set_title('Knapsack Pairwise Win-Loss Heatmap\n(Cell (i,j) = # times algo i beats algo j significantly)',
                fontsize=14, fontweight='bold')
    
    # Footer
    fig.text(0.5, 0.01, 'Wilcoxon signed-rank + Holm correction; A12 (Vargha-Delaney) / Cliff\'s δ effect size',
            ha='center', fontsize=8, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path / 'knapsack_pairwise_heatmap_overall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path / 'knapsack_pairwise_heatmap_overall.png'}")
    
    # Save win-loss CSV
    win_matrix.to_csv(summary_path / 'knapsack_pairwise_wins_overall.csv')
    logger.info(f"✓ Saved: {summary_path / 'knapsack_pairwise_wins_overall.csv'}")
    
    # Copeland ranking
    copeland_scores = []
    for algo in algos:
        wins = win_matrix.loc[algo].sum()
        losses = win_matrix[algo].sum()
        score = wins - losses
        copeland_scores.append({'Algorithm': algo, 'Wins': wins, 'Losses': losses, 'Copeland_Score': score})
    
    copeland_df = pd.DataFrame(copeland_scores).sort_values('Copeland_Score', ascending=False)
    copeland_df['Rank'] = range(1, len(copeland_df) + 1)
    
    copeland_file = summary_path / 'knapsack_copeland_overall.csv'
    copeland_df.to_csv(copeland_file, index=False)
    logger.info(f"✓ Saved: {copeland_file}")
    
    # Copeland ranking plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors_list = [COLORS.get(a, 'gray') for a in copeland_df['Algorithm']]
    bars = ax.barh(copeland_df['Algorithm'], copeland_df['Copeland_Score'],
                   color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, score in zip(bars, copeland_df['Copeland_Score']):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2.,
                f'{score:+d}', ha='left' if width >= 0 else 'right',
                va='center', fontsize=11, fontweight='bold')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Copeland Score (Wins - Losses)', fontsize=12)
    ax.set_title('Knapsack: Copeland Ranking\n(Higher = More dominant)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path / 'knapsack_copeland_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path / 'knapsack_copeland_ranking.png'}")
    
    # By Scenario
    if has_scenario:
        for scenario in df['Scenario'].unique():
            df_scenario = df_sig[df_sig['Scenario'] == scenario]
            
            if df_scenario.empty:
                continue
            
            win_matrix_sc = pd.DataFrame(0, index=algos, columns=algos)
            
            for _, row in df_scenario.iterrows():
                if row['A12'] > 0.5:
                    winner, loser = row['Algo1'], row['Algo2']
                else:
                    winner, loser = row['Algo2'], row['Algo1']
                
                win_matrix_sc.loc[winner, loser] += 1
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(win_matrix_sc, annot=True, fmt='d', cmap='RdYlGn', center=0,
                       cbar_kws={'label': '# Significant Wins'},
                       linewidths=0.5, linecolor='gray', ax=ax)
            
            ax.set_xlabel('Loser', fontsize=12)
            ax.set_ylabel('Winner', fontsize=12)
            ax.set_title(f'Knapsack Pairwise Win-Loss: {scenario.replace("_", " ").title()}',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path / f'knapsack_pairwise_heatmap_{scenario}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ Saved: {output_path / f'knapsack_pairwise_heatmap_{scenario}.png'}")


def visualize_knapsack(summary_dir: str = 'benchmark/results/summaries',
                       output_dir: str = 'benchmark/results/figures') -> Dict[str, str]:
    """
    Master function to generate all Knapsack visualizations following
    Dolan-Moré / Moré-Wild standards and non-parametric statistics.
    
    Parameters
    ----------
    summary_dir : str
        Directory containing CSV summary files
    output_dir : str
        Output directory for figures
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping visualization type to output file path
    """
    logger.info("=" * 80)
    logger.info("KNAPSACK VISUALIZATIONS - DOLAN-MORÉ & MORÉ-WILD STANDARDS")
    logger.info("=" * 80)
    
    summary_path = Path(summary_dir)
    output_path = Path(output_dir)
    
    generated_files = {}
    
    # 1. Performance Profiles (Dolan-Moré)
    logger.info("\n[1/4] Performance Profiles (Dolan-Moré)...")
    df_perf = load_summary('knapsack_performance_profiles', summary_path)
    if df_perf is not None:
        plot_knapsack_performance_profiles_dolan_more(df_perf, output_dir)
        generated_files['performance_profiles'] = str(output_path / 'knapsack_performance_profiles.png')
    
    # 2. Data Profiles (Moré-Wild)
    logger.info("\n[2/4] Data Profiles (Moré-Wild)...")
    df_data = load_summary('knapsack_data_profiles', summary_path)
    if df_data is not None:
        plot_knapsack_data_profiles_more_wild(df_data, output_dir)
        generated_files['data_profiles'] = str(output_path / 'knapsack_data_profiles.png')
    
    # 3. Fixed-Budget Analysis
    logger.info("\n[3/4] Fixed-Budget Analysis...")
    df_budget = load_summary('knapsack_fixed_budget', summary_path)
    if df_budget is not None:
        plot_knapsack_fixed_budget_analysis(df_budget, output_dir, summary_dir)
        generated_files['fixed_budget'] = str(output_path / 'knapsack_fixed_budget_*.png')
        generated_files['fixed_budget_summary'] = str(summary_path / 'knapsack_fixed_budget_summary.csv')
    
    # 4. Pairwise Statistics & Copeland Ranking
    logger.info("\n[4/4] Pairwise Statistics & Copeland Ranking...")
    df_pairwise = load_summary('knapsack_pairwise_stats', summary_path)
    if df_pairwise is not None:
        plot_knapsack_pairwise_stats_heatmap(df_pairwise, output_dir, summary_dir)
        generated_files['pairwise_heatmap'] = str(output_path / 'knapsack_pairwise_heatmap_overall.png')
        generated_files['copeland'] = str(summary_path / 'knapsack_copeland_overall.csv')
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Knapsack visualizations complete")
    logger.info(f"  Generated {len(generated_files)} artifact types")
    logger.info("=" * 80)
    
    return generated_files


# ============================================================================
# MASTER GENERATION FUNCTION (UPDATED)
# ============================================================================

def generate_all_plots(
    summary_dir: str = 'benchmark/results/summaries',
    output_dir: str = 'benchmark/results/plots',
    problem_type: str = 'all'
):
    """
    Generate all visualizations from CSV summaries.
    
    This is the ONLY entry point - reads CSV, not JSON.
    
    Parameters
    ----------
    summary_dir : str
        Directory containing CSV summary files
    output_dir : str
        Output directory for plots
    problem_type : str
        Problem type to visualize: 'all', 'rastrigin', or 'knapsack'
    """
    summary_path = Path(summary_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print(f"GENERATING VISUALIZATIONS FROM CSV SUMMARIES - {problem_type.upper()}")
    print("=" * 100)
    
    # ========================================================================
    # RASTRIGIN: Load all artifacts and generate COCO/BBOB plots
    # ========================================================================
    
    if problem_type in ['all', 'rastrigin']:
        print("\n[LOADING RASTRIGIN ARTIFACTS]")
        rastrigin_artifacts = load_rastrigin_artifacts(summary_path)
        rastrigin_ranks_df = load_summary_data(summary_path / 'rastrigin_global_ranks.csv')
        
        if rastrigin_artifacts:
            print("\n[RASTRIGIN VISUALIZATIONS]")
            
            # 1. Fixed-Target ECDF (COCO standard)
            if 'fixed_target_ecdf' in rastrigin_artifacts:
                plot_rastrigin_fixed_target_ecdf(rastrigin_artifacts['fixed_target_ecdf'], output_dir)
            
            # 2. Expected Running Time (ERT)
            if 'ert' in rastrigin_artifacts:
                for level in ['Gold', 'Silver', 'Bronze']:
                    plot_rastrigin_ert(rastrigin_artifacts['ert'], output_dir, level=level)
            
            # 3. Fixed-Budget Performance
            if 'fixed_budget' in rastrigin_artifacts:
                plot_rastrigin_fixed_budget(rastrigin_artifacts['fixed_budget'], output_dir)
            
            # 4. Performance Profiles (Dolan-Moré)
            if 'performance_profiles' in rastrigin_artifacts:
                plot_rastrigin_performance_profile(rastrigin_artifacts['performance_profiles'], output_dir)
            
            # 5. Data Profiles (Moré-Wild)
            if 'data_profiles' in rastrigin_artifacts:
                plot_rastrigin_data_profile(rastrigin_artifacts['data_profiles'], output_dir)
            
            # 6. Diversity Analysis
            if 'diversity' in rastrigin_artifacts:
                plot_rastrigin_diversity_panel(rastrigin_artifacts['diversity'], output_dir)
            
            # 7. Stagnation Analysis
            if 'stagnation' in rastrigin_artifacts:
                plot_rastrigin_stagnation(rastrigin_artifacts['stagnation'], output_dir)
            
            # 8. Anytime AUC
            if 'summary' in rastrigin_artifacts:
                plot_rastrigin_anytime_auc(rastrigin_artifacts['summary'], output_dir)
            
            # 9. Existing plots (boxplots, success breakdown, etc.)
            if 'summary' in rastrigin_artifacts:
                plot_rastrigin_boxplots_by_scenario(rastrigin_artifacts['summary'], output_dir)
                plot_rastrigin_success_breakdown(rastrigin_artifacts['summary'], output_dir)
                plot_rastrigin_cross_scenario_comparison(rastrigin_artifacts['summary'], output_dir)
                plot_rastrigin_hitting_times(rastrigin_artifacts['summary'], output_dir)
            
            # 10. Global Ranks
            if rastrigin_ranks_df is not None:
                plot_global_ranks(
                    rastrigin_ranks_df,
                    str(output_path / 'rastrigin_global_ranks.png'),
                    title='Rastrigin: Global Algorithm Ranks'
                )
        else:
            logger.warning("No Rastrigin artifacts found")
    
    # ========================================================================
    # KNAPSACK: Use new Dolan-Moré / Moré-Wild functions
    # ========================================================================
    
    if problem_type in ['all', 'knapsack']:
        print("\n[KNAPSACK VISUALIZATIONS - DOLAN-MORÉ & MORÉ-WILD]")
        visualize_knapsack(summary_dir, output_dir)
    
    print("\n" + "=" * 100)
    print(f"All plots saved to: {output_path}")
    print("=" * 100)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate visualizations from CSV summaries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots (both Knapsack and Rastrigin)
  python benchmark/visualize.py
  
  # Generate only Knapsack plots
  python benchmark/visualize.py --problem knapsack
  
  # Generate only Rastrigin plots
  python benchmark/visualize.py --problem rastrigin
  
  # Custom summary and output directories with specific problem
  python benchmark/visualize.py --problem knapsack --summary-dir analysis/results --output-dir plots
        """
    )
    
    parser.add_argument(
        '--summary-dir', type=str,
        default='benchmark/results/summaries',
        help='Directory containing CSV summary files'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='benchmark/results/plots',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--problem', '-p', type=str,
        choices=['all', 'knapsack', 'rastrigin'],
        default='all',
        help='Problem type to visualize (default: all)'
    )
    
    args = parser.parse_args()
    
    generate_all_plots(args.summary_dir, args.output_dir, args.problem)


if __name__ == "__main__":
    main()