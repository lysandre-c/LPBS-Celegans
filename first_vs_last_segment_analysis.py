#!/usr/bin/env python3
"""
Comprehensive analysis of feature differences between first and last segments.
This script provides multiple statistical tests, effect size calculations, 
and visualizations to identify the most discriminative features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

def calculate_effect_size(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    return (group1.mean() - group2.mean()) / pooled_std

def interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def analyze_feature_differences(df, segment_definition='percentage', threshold=50):
    """
    Analyze feature differences between first and last segments.
    
    Args:
        df: DataFrame with segment features
        segment_definition: 'percentage' or 'absolute'
        threshold: If percentage, split at this percentile. If absolute, number of segments from start/end
    """
    
    print(f"=== First vs Last Segment Analysis ===")
    print(f"Segment definition: {segment_definition}")
    print(f"Threshold: {threshold}")
    print()
    
    # Extract segment index from filename if not present
    if df['segment_index'].isna().all():
        print("Extracting segment index from filename...")
        df['segment_index'] = df['filename'].str.extract(r'segment(\d+(?:\.\d+)?)', expand=False).astype(float)
        print(f"Extracted segment indices: {df['segment_index'].min():.0f} to {df['segment_index'].max():.0f}")
        print()
    
    # Calculate segment positions
    if segment_definition == 'percentage':
        # Calculate lifespan percentage for each segment
        worm_stats = df.groupby('original_file').agg({'segment_index': ['max', 'count']}).reset_index()
        worm_stats.columns = ['original_file', 'max_segment', 'total_segments']
        df = df.merge(worm_stats, on='original_file', how='left')
        df['lifespan_percentage'] = (df['segment_index'] / df['max_segment']) * 100
        
        # Split data
        first_segments = df[df['lifespan_percentage'] <= threshold]
        last_segments = df[df['lifespan_percentage'] > threshold]
        
        print(f"First {threshold}% of life: {len(first_segments)} segments")
        print(f"Last {100-threshold}% of life: {len(last_segments)} segments")
        
    elif segment_definition == 'absolute':
        # Use absolute segment numbers from start and end
        first_segments_list = []
        last_segments_list = []
        
        for worm_file in df['original_file'].unique():
            worm_df = df[df['original_file'] == worm_file].sort_values('segment_index')
            total_segments = len(worm_df)
            
            if total_segments > 2 * threshold:  # Only include if worm has enough segments
                first_segments_list.append(worm_df.head(threshold))
                last_segments_list.append(worm_df.tail(threshold))
        
        if first_segments_list:
            first_segments = pd.concat(first_segments_list, ignore_index=True)
            last_segments = pd.concat(last_segments_list, ignore_index=True)
        else:
            print("No worms have enough segments for absolute threshold analysis")
            return None
            
        print(f"First {threshold} segments: {len(first_segments)} total segments")
        print(f"Last {threshold} segments: {len(last_segments)} total segments")
    
    print()
    
    # Get feature columns (exclude metadata)
    metadata_columns = ['label', 'filename', 'relative_path', 'file', 'worm_id', 'segment_number', 
                       'segment_index', 'original_file', 'lifespan_percentage', 'max_segment', 'total_segments']
    feature_columns = [col for col in df.columns if col not in metadata_columns and df[col].dtype in ['float64', 'int64']]
    
    print(f"Analyzing {len(feature_columns)} features...")
    print()
    
    # Analyze each feature
    results = {}
    
    for feature in feature_columns:
        if feature in df.columns and not first_segments[feature].isna().all() and not last_segments[feature].isna().all():
            
            first_vals = first_segments[feature].dropna()
            last_vals = last_segments[feature].dropna()
            
            if len(first_vals) == 0 or len(last_vals) == 0:
                continue
                
            # Basic statistics
            first_mean = first_vals.mean()
            last_mean = last_vals.mean()
            first_std = first_vals.std()
            last_std = last_vals.std()
            first_median = first_vals.median()
            last_median = last_vals.median()
            
            # Calculate percentage change
            if first_mean != 0:
                pct_change = ((last_mean - first_mean) / abs(first_mean)) * 100
            else:
                pct_change = np.nan
            
            # Statistical tests
            # 1. T-test (parametric)
            try:
                t_stat, t_p_value = stats.ttest_ind(first_vals, last_vals, equal_var=False)
            except:
                t_stat, t_p_value = np.nan, np.nan
            
            # 2. Mann-Whitney U test (non-parametric)
            try:
                u_stat, u_p_value = mannwhitneyu(first_vals, last_vals, alternative='two-sided')
            except:
                u_stat, u_p_value = np.nan, np.nan
            
            # 3. Effect size (Cohen's d)
            effect_size = calculate_effect_size(first_vals, last_vals)
            effect_interpretation = interpret_effect_size(effect_size)
            
            # 4. Levene's test for equal variances
            try:
                levene_stat, levene_p = stats.levene(first_vals, last_vals)
            except:
                levene_stat, levene_p = np.nan, np.nan
            
            results[feature] = {
                'first_mean': first_mean,
                'last_mean': last_mean,
                'first_std': first_std,
                'last_std': last_std,
                'first_median': first_median,
                'last_median': last_median,
                'absolute_difference': last_mean - first_mean,
                'percentage_change': pct_change,
                'effect_size_cohens_d': effect_size,
                'effect_interpretation': effect_interpretation,
                't_statistic': t_stat,
                't_p_value': t_p_value,
                'mannwhitney_u': u_stat,
                'mannwhitney_p_value': u_p_value,
                'levene_statistic': levene_stat,
                'levene_p_value': levene_p,
                't_significant': t_p_value < 0.05 if not np.isnan(t_p_value) else False,
                'mannwhitney_significant': u_p_value < 0.05 if not np.isnan(u_p_value) else False,
                'variance_equal': levene_p > 0.05 if not np.isnan(levene_p) else True,
                'n_first': len(first_vals),
                'n_last': len(last_vals)
            }
    
    return results, first_segments, last_segments

def rank_features_by_criteria(results):
    """Rank features by different criteria."""
    
    rankings = {}
    
    # 1. By absolute effect size
    rankings['effect_size'] = sorted(
        results.items(),
        key=lambda x: abs(x[1]['effect_size_cohens_d']) if not np.isnan(x[1]['effect_size_cohens_d']) else 0,
        reverse=True
    )
    
    # 2. By percentage change magnitude
    rankings['percentage_change'] = sorted(
        results.items(),
        key=lambda x: abs(x[1]['percentage_change']) if not np.isnan(x[1]['percentage_change']) else 0,
        reverse=True
    )
    
    # 3. By statistical significance (lowest p-value from Mann-Whitney test)
    rankings['significance'] = sorted(
        results.items(),
        key=lambda x: x[1]['mannwhitney_p_value'] if not np.isnan(x[1]['mannwhitney_p_value']) else 1,
        reverse=False
    )
    
    # 4. Combined score: effect size * -log10(p_value)
    combined_scores = {}
    for feature, stats in results.items():
        effect = abs(stats['effect_size_cohens_d']) if not np.isnan(stats['effect_size_cohens_d']) else 0
        p_val = stats['mannwhitney_p_value'] if not np.isnan(stats['mannwhitney_p_value']) else 1
        # Avoid log(0) by adding small epsilon
        log_p = -np.log10(max(p_val, 1e-10))
        combined_scores[feature] = effect * log_p
    
    rankings['combined'] = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return rankings

def create_visualizations(results, first_segments, last_segments, top_n=10):
    """Create visualizations for top differentiating features."""
    
    # Get top features by effect size
    top_features = sorted(
        results.items(),
        key=lambda x: abs(x[1]['effect_size_cohens_d']) if not np.isnan(x[1]['effect_size_cohens_d']) else 0,
        reverse=True
    )[:top_n]
    
    # Create subplots
    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, (feature, stats) in enumerate(top_features):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get data
        first_vals = first_segments[feature].dropna()
        last_vals = last_segments[feature].dropna()
        
        # Create box plot
        data_to_plot = [first_vals, last_vals]
        labels = ['First', 'Last']
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        # Add statistics to title
        effect_size = stats['effect_size_cohens_d']
        p_value = stats['mannwhitney_p_value']
        pct_change = stats['percentage_change']
        
        title = f"{feature}\n"
        title += f"Effect size: {effect_size:.3f} ({stats['effect_interpretation']})\n"
        title += f"p-value: {p_value:.2e}\n"
        title += f"Change: {pct_change:.1f}%"
        
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(len(top_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('first_vs_last_segments_top_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create effect size summary plot
    plt.figure(figsize=(12, 8))
    
    features = [item[0] for item in top_features]
    effect_sizes = [item[1]['effect_size_cohens_d'] for item in top_features]
    p_values = [item[1]['mannwhitney_p_value'] for item in top_features]
    
    # Color by significance
    colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow' if p < 0.05 else 'gray' 
              for p in p_values]
    
    bars = plt.barh(range(len(features)), effect_sizes, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Effect Size (Cohen\'s d)')
    plt.title('Top Features by Effect Size\n(Red: p<0.001, Orange: p<0.01, Yellow: p<0.05, Gray: p≥0.05)')
    
    # Add effect size reference lines
    plt.axvline(x=0.2, color='black', linestyle='--', alpha=0.5, label='Small effect')
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Medium effect')
    plt.axvline(x=0.8, color='black', linestyle='--', alpha=0.9, label='Large effect')
    plt.axvline(x=-0.2, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=-0.5, color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=-0.8, color='black', linestyle='--', alpha=0.9)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('effect_sizes_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_results(results, rankings, top_n=15):
    """Print detailed results in multiple formats."""
    
    print("=" * 120)
    print("DETAILED RESULTS: TOP FEATURES BY DIFFERENT CRITERIA")
    print("=" * 120)
    
    criteria_names = {
        'effect_size': 'Effect Size (Cohen\'s d)',
        'percentage_change': 'Percentage Change',
        'significance': 'Statistical Significance (Mann-Whitney U)',
        'combined': 'Combined Score (Effect Size × -log10(p-value))'
    }
    
    for criterion, name in criteria_names.items():
        print(f"\n{'='*50}")
        print(f"TOP {top_n} FEATURES BY {name.upper()}")
        print(f"{'='*50}")
        
        if criterion == 'combined':
            # For combined ranking, we need to access the scores differently
            top_items = rankings[criterion][:top_n]
            print(f"{'Rank':<4} {'Feature':<25} {'Combined Score':<15} {'Effect Size':<12} {'p-value':<10} {'Change %':<10}")
            print("-" * 90)
            
            for rank, (feature, combined_score) in enumerate(top_items, 1):
                stats = results[feature]
                effect_size = stats['effect_size_cohens_d']
                p_value = stats['mannwhitney_p_value']
                pct_change = stats['percentage_change']
                
                print(f"{rank:<4} {feature:<25} {combined_score:<15.3f} {effect_size:<12.3f} "
                      f"{p_value:<10.2e} {pct_change:<10.1f}%")
        else:
            # For other rankings
            top_items = rankings[criterion][:top_n]
            
            if criterion == 'effect_size':
                print(f"{'Rank':<4} {'Feature':<25} {'Effect Size':<12} {'Interpretation':<12} {'p-value':<10} {'Change %':<10}")
                print("-" * 85)
                
                for rank, (feature, stats) in enumerate(top_items, 1):
                    effect_size = stats['effect_size_cohens_d']
                    interpretation = stats['effect_interpretation']
                    p_value = stats['mannwhitney_p_value']
                    pct_change = stats['percentage_change']
                    
                    print(f"{rank:<4} {feature:<25} {effect_size:<12.3f} {interpretation:<12} "
                          f"{p_value:<10.2e} {pct_change:<10.1f}%")
                          
            elif criterion == 'percentage_change':
                print(f"{'Rank':<4} {'Feature':<25} {'Change %':<12} {'Effect Size':<12} {'p-value':<10}")
                print("-" * 75)
                
                for rank, (feature, stats) in enumerate(top_items, 1):
                    pct_change = stats['percentage_change']
                    effect_size = stats['effect_size_cohens_d']
                    p_value = stats['mannwhitney_p_value']
                    
                    print(f"{rank:<4} {feature:<25} {pct_change:<12.1f}% {effect_size:<12.3f} {p_value:<10.2e}")
                    
            elif criterion == 'significance':
                print(f"{'Rank':<4} {'Feature':<25} {'p-value':<12} {'Effect Size':<12} {'Change %':<10}")
                print("-" * 75)
                
                for rank, (feature, stats) in enumerate(top_items, 1):
                    p_value = stats['mannwhitney_p_value']
                    effect_size = stats['effect_size_cohens_d']
                    pct_change = stats['percentage_change']
                    
                    print(f"{rank:<4} {feature:<25} {p_value:<12.2e} {effect_size:<12.3f} {pct_change:<10.1f}%")

def save_detailed_results(results, rankings, filename='first_vs_last_segments_detailed_results.csv'):
    """Save detailed results to CSV."""
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    
    # Add ranking information
    for criterion, ranking in rankings.items():
        if criterion != 'combined':
            rank_dict = {feature: rank+1 for rank, (feature, _) in enumerate(ranking)}
            results_df[f'rank_{criterion}'] = results_df.index.map(rank_dict)
        else:
            rank_dict = {feature: rank+1 for rank, (feature, _) in enumerate(ranking)}
            results_df[f'rank_{criterion}'] = results_df.index.map(rank_dict)
            score_dict = {feature: score for feature, score in ranking}
            results_df['combined_score'] = results_df.index.map(score_dict)
    
    # Sort by combined ranking
    results_df = results_df.sort_values('rank_combined')
    
    # Save to CSV
    results_df.to_csv(filename)
    print(f"\nDetailed results saved to: {filename}")
    
    return results_df

def print_summary_statistics(results, first_segments, last_segments):
    """Print summary statistics."""
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Count significant features
    t_significant = sum(1 for stats in results.values() if stats['t_significant'])
    mw_significant = sum(1 for stats in results.values() if stats['mannwhitney_significant'])
    
    # Count by effect size
    large_effect = sum(1 for stats in results.values() if stats['effect_interpretation'] == 'large')
    medium_effect = sum(1 for stats in results.values() if stats['effect_interpretation'] == 'medium')
    small_effect = sum(1 for stats in results.values() if stats['effect_interpretation'] == 'small')
    negligible_effect = sum(1 for stats in results.values() if stats['effect_interpretation'] == 'negligible')
    
    # Count by percentage change
    large_change = sum(1 for stats in results.values() 
                      if not np.isnan(stats['percentage_change']) and abs(stats['percentage_change']) > 100)
    moderate_change = sum(1 for stats in results.values() 
                         if not np.isnan(stats['percentage_change']) and 50 < abs(stats['percentage_change']) <= 100)
    small_change = sum(1 for stats in results.values() 
                      if not np.isnan(stats['percentage_change']) and 20 < abs(stats['percentage_change']) <= 50)
    
    print(f"Total features analyzed: {len(results)}")
    print(f"First segments: {len(first_segments)} from {first_segments['original_file'].nunique()} worms")
    print(f"Last segments: {len(last_segments)} from {last_segments['original_file'].nunique()} worms")
    print()
    
    if len(results) == 0:
        print("No features were analyzed. Check your data and filtering criteria.")
        return
    
    print("Statistical Significance:")
    print(f"  - t-test significant (p < 0.05): {t_significant} ({t_significant/len(results)*100:.1f}%)")
    print(f"  - Mann-Whitney U significant (p < 0.05): {mw_significant} ({mw_significant/len(results)*100:.1f}%)")
    print()
    
    print("Effect Sizes:")
    print(f"  - Large effect (|d| > 0.8): {large_effect} ({large_effect/len(results)*100:.1f}%)")
    print(f"  - Medium effect (0.5 < |d| ≤ 0.8): {medium_effect} ({medium_effect/len(results)*100:.1f}%)")
    print(f"  - Small effect (0.2 < |d| ≤ 0.5): {small_effect} ({small_effect/len(results)*100:.1f}%)")
    print(f"  - Negligible effect (|d| ≤ 0.2): {negligible_effect} ({negligible_effect/len(results)*100:.1f}%)")
    print()
    
    print("Percentage Changes:")
    print(f"  - Large change (>100%): {large_change} ({large_change/len(results)*100:.1f}%)")
    print(f"  - Moderate change (50-100%): {moderate_change} ({moderate_change/len(results)*100:.1f}%)")
    print(f"  - Small change (20-50%): {small_change} ({small_change/len(results)*100:.1f}%)")

def main():
    """Main analysis function."""
    
    print("Loading segment features data...")
    df = pd.read_csv('feature_data/segments_features.csv')
    
    print(f"Loaded {len(df)} segments from {df['original_file'].nunique()} worms")
    print(f"Available features: {len([col for col in df.columns if df[col].dtype in ['float64', 'int64']])}")
    print()
    
    # Run analysis with percentage-based splitting (default)
    results, first_segments, last_segments = analyze_feature_differences(
        df, 
        segment_definition='percentage', 
        threshold=50
    )
    
    if results is None or len(results) == 0:
        print("No results to analyze. Check your data structure and filtering criteria.")
        return
    
    # Rank features by different criteria
    rankings = rank_features_by_criteria(results)
    
    # Print detailed results
    print_detailed_results(results, rankings, top_n=20)
    
    # Print summary statistics
    print_summary_statistics(results, first_segments, last_segments)
    
    # Save results
    results_df = save_detailed_results(results, rankings)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results, first_segments, last_segments, top_n=12)
    
    print("\nAnalysis complete!")
    print("Files generated:")
    print("- first_vs_last_segments_detailed_results.csv")
    print("- first_vs_last_segments_top_features.png")
    print("- effect_sizes_summary.png")

if __name__ == "__main__":
    main()
