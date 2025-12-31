"""
SECTOR REGIME ANALYSIS - Deep Dive into 110-Stock Data
=======================================================

PIVOT FROM: "Kurtosis predicts regime duration" (fragile)
PIVOT TO:   "What do stocks tell us about sector-level regime behavior?"

KEY QUESTIONS:
1. Which sectors have the most persistent regimes?
2. Which sectors behave similarly? (clustering)
3. Can we create "sector fingerprints" for regime prediction?
4. Are there sector-specific trading implications?

Created: 2025-12-31
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD EXISTING VALIDATION DATA
# ============================================================================

def load_validation_data() -> List[Dict]:
    """Load the 110-stock validation data from JSON."""
    with open('research/outputs/kurtosis_validation_data.json', 'r') as f:
        data = json.load(f)
    return data['results']


# ============================================================================
# SECTOR STATISTICS
# ============================================================================

class SectorRegimeAnalyzer:
    """
    Analyze regime behavior at the sector level.
    Uses the 110-stock validation data as foundation.
    """
    
    def __init__(self, results: List[Dict]):
        self.results = [r for r in results if 'error' not in r]
        self.sector_stats = {}
        self.sector_fingerprints = {}
        
    def compute_sector_statistics(self) -> pd.DataFrame:
        """
        Compute comprehensive statistics for each sector.
        
        Returns DataFrame with:
        - n_stocks: number of stocks in sector
        - avg_duration: mean regime duration
        - std_duration: std of regime duration
        - avg_kurtosis: mean kurtosis
        - avg_volatility: mean annualized volatility
        - avg_jump_freq: mean jump frequency
        - duration_range: max - min duration
        - kurtosis_range: max - min kurtosis
        """
        sector_data = {}
        
        for r in self.results:
            sector = r.get('sector', 'Unknown')
            if sector not in sector_data:
                sector_data[sector] = []
            sector_data[sector].append(r)
        
        rows = []
        for sector, stocks in sector_data.items():
            durations = [s['avg_duration'] for s in stocks]
            kurtosis = [s['kurtosis'] for s in stocks]
            volatility = [s['volatility'] for s in stocks]
            jump_freq = [s['jump_freq'] for s in stocks]
            vol_cluster = [s['vol_cluster'] for s in stocks]
            
            row = {
                'sector': sector,
                'n_stocks': len(stocks),
                'avg_duration': np.mean(durations),
                'std_duration': np.std(durations),
                'median_duration': np.median(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'avg_kurtosis': np.mean(kurtosis),
                'std_kurtosis': np.std(kurtosis),
                'median_kurtosis': np.median(kurtosis),
                'avg_volatility': np.mean(volatility),
                'std_volatility': np.std(volatility),
                'avg_jump_freq': np.mean(jump_freq),
                'avg_vol_cluster': np.mean(vol_cluster),
                # Derived metrics
                'duration_range': np.max(durations) - np.min(durations),
                'kurtosis_range': np.max(kurtosis) - np.min(kurtosis),
                'duration_cv': np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0,
                'kurtosis_cv': np.std(kurtosis) / np.mean(kurtosis) if np.mean(kurtosis) > 0 else 0,
            }
            
            # Store for fingerprinting
            self.sector_stats[sector] = {
                'stocks': stocks,
                'stats': row,
            }
            
            rows.append(row)
        
        return pd.DataFrame(rows).sort_values('avg_duration', ascending=False)
    
    def create_sector_fingerprints(self) -> Dict:
        """
        Create normalized "fingerprints" for each sector.
        
        A fingerprint is a vector of normalized metrics:
        [duration, volatility, kurtosis, jump_freq, vol_cluster]
        
        This allows sector comparison and clustering.
        """
        if not self.sector_stats:
            self.compute_sector_statistics()
        
        # Collect all values for normalization
        all_durations = [s['stats']['avg_duration'] for s in self.sector_stats.values()]
        all_volatilities = [s['stats']['avg_volatility'] for s in self.sector_stats.values()]
        all_kurtosis = [s['stats']['avg_kurtosis'] for s in self.sector_stats.values()]
        all_jump = [s['stats']['avg_jump_freq'] for s in self.sector_stats.values()]
        all_vol_cluster = [s['stats']['avg_vol_cluster'] for s in self.sector_stats.values()]
        
        # Z-score normalization
        def zscore(values, val):
            return (val - np.mean(values)) / np.std(values) if np.std(values) > 0 else 0
        
        for sector, data in self.sector_stats.items():
            stats = data['stats']
            fingerprint = {
                'duration_z': zscore(all_durations, stats['avg_duration']),
                'volatility_z': zscore(all_volatilities, stats['avg_volatility']),
                'kurtosis_z': zscore(all_kurtosis, stats['avg_kurtosis']),
                'jump_freq_z': zscore(all_jump, stats['avg_jump_freq']),
                'vol_cluster_z': zscore(all_vol_cluster, stats['avg_vol_cluster']),
            }
            
            # Create interpretable labels
            labels = []
            if fingerprint['duration_z'] > 0.5:
                labels.append('Long Regimes')
            elif fingerprint['duration_z'] < -0.5:
                labels.append('Short Regimes')
            
            if fingerprint['volatility_z'] > 0.5:
                labels.append('High Vol')
            elif fingerprint['volatility_z'] < -0.5:
                labels.append('Low Vol')
            
            if fingerprint['kurtosis_z'] > 0.5:
                labels.append('Fat Tails')
            elif fingerprint['kurtosis_z'] < -0.5:
                labels.append('Normal Tails')
            
            if fingerprint['jump_freq_z'] > 0.5:
                labels.append('Jump-prone')
            
            fingerprint['labels'] = labels if labels else ['Average']
            self.sector_fingerprints[sector] = fingerprint
        
        return self.sector_fingerprints
    
    def cluster_sectors(self, n_clusters: int = 4) -> Dict:
        """
        Cluster sectors by their fingerprint similarity.
        
        Returns:
            Dict mapping cluster_id to list of sectors
        """
        if not self.sector_fingerprints:
            self.create_sector_fingerprints()
        
        # Create feature matrix
        sectors = list(self.sector_fingerprints.keys())
        features = ['duration_z', 'volatility_z', 'kurtosis_z', 'jump_freq_z', 'vol_cluster_z']
        
        X = np.array([
            [self.sector_fingerprints[s][f] for f in features]
            for s in sectors
        ])
        
        # Hierarchical clustering
        linkage_matrix = linkage(X, method='ward')
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Group sectors by cluster
        cluster_groups = {}
        for sector, cluster_id in zip(sectors, clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(sector)
        
        return cluster_groups, linkage_matrix, sectors
    
    def find_similar_sectors(self, target_sector: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Find sectors most similar to the target sector.
        
        Returns: List of (sector, similarity_score) tuples
        """
        if not self.sector_fingerprints:
            self.create_sector_fingerprints()
        
        if target_sector not in self.sector_fingerprints:
            return []
        
        target = self.sector_fingerprints[target_sector]
        features = ['duration_z', 'volatility_z', 'kurtosis_z', 'jump_freq_z', 'vol_cluster_z']
        target_vec = np.array([target[f] for f in features])
        
        similarities = []
        for sector, fp in self.sector_fingerprints.items():
            if sector == target_sector:
                continue
            vec = np.array([fp[f] for f in features])
            # Euclidean distance (lower = more similar)
            distance = np.linalg.norm(target_vec - vec)
            # Convert to similarity (higher = more similar)
            similarity = 1 / (1 + distance)
            similarities.append((sector, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    
    def regime_behavior_summary(self) -> str:
        """Generate human-readable regime behavior summary."""
        if not self.sector_fingerprints:
            self.create_sector_fingerprints()
        
        lines = [
            "=" * 70,
            "SECTOR REGIME BEHAVIOR SUMMARY",
            "=" * 70,
            ""
        ]
        
        # Sort by average duration
        sorted_sectors = sorted(
            self.sector_stats.items(),
            key=lambda x: x[1]['stats']['avg_duration'],
            reverse=True
        )
        
        for sector, data in sorted_sectors:
            stats = data['stats']
            fp = self.sector_fingerprints[sector]
            labels = ', '.join(fp['labels'])
            
            lines.append(f"📊 {sector}")
            lines.append(f"   Duration: {stats['avg_duration']:.1f}d (±{stats['std_duration']:.1f})")
            lines.append(f"   Volatility: {stats['avg_volatility']*100:.1f}%")
            lines.append(f"   Kurtosis: {stats['avg_kurtosis']:.1f}")
            lines.append(f"   Pattern: {labels}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def trading_implications(self) -> Dict:
        """
        Generate trading implications based on sector analysis.
        
        Returns actionable insights for each sector.
        """
        if not self.sector_fingerprints:
            self.create_sector_fingerprints()
        
        implications = {}
        
        for sector, fp in self.sector_fingerprints.items():
            stats = self.sector_stats[sector]['stats']
            
            impl = {
                'regime_horizon': 'long' if fp['duration_z'] > 0 else 'short',
                'expected_duration_days': stats['avg_duration'],
                'volatility_profile': 'high' if fp['volatility_z'] > 0 else 'low',
                'tail_risk': 'elevated' if fp['kurtosis_z'] > 0 else 'normal',
                'recommendations': []
            }
            
            # Generate recommendations
            if fp['duration_z'] > 0.5:
                impl['recommendations'].append("Use longer lookback for regime detection")
                impl['recommendations'].append("Trend-following strategies may work well")
            else:
                impl['recommendations'].append("Use shorter lookback for regime detection")
                impl['recommendations'].append("Mean-reversion may dominate")
            
            if fp['volatility_z'] > 0.5:
                impl['recommendations'].append("Reduce position sizes vs other sectors")
                impl['recommendations'].append("Consider volatility scaling")
            
            if fp['kurtosis_z'] > 0.5:
                impl['recommendations'].append("Use fat-tailed distributions in Monte Carlo")
                impl['recommendations'].append("Wider stop-losses for jump protection")
            
            if fp['jump_freq_z'] > 0.5:
                impl['recommendations'].append("Event-driven moves common - use news filters")
            
            implications[sector] = impl
        
        return implications


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_sector_analysis(analyzer: SectorRegimeAnalyzer):
    """Create comprehensive visualization of sector analysis."""
    
    # Get data
    df = analyzer.compute_sector_statistics()
    fingerprints = analyzer.create_sector_fingerprints()
    clusters, linkage_matrix, sectors = analyzer.cluster_sectors()
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Sector Regime Analysis - 110 Stock Universe', fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Duration by Sector (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    df_sorted = df.sort_values('avg_duration', ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
    bars = ax1.barh(df_sorted['sector'].str[:15], df_sorted['avg_duration'], color=colors, edgecolor='white')
    ax1.errorbar(df_sorted['avg_duration'], range(len(df_sorted)), 
                xerr=df_sorted['std_duration'], fmt='none', color='black', capsize=3, alpha=0.5)
    ax1.set_xlabel('Avg Regime Duration (days)')
    ax1.set_title('Regime Duration by Sector')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Volatility by Sector (bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    df_vol = df.sort_values('avg_volatility', ascending=True)
    colors = plt.cm.plasma(np.linspace(0, 1, len(df_vol)))
    ax2.barh(df_vol['sector'].str[:15], df_vol['avg_volatility']*100, color=colors, edgecolor='white')
    ax2.set_xlabel('Avg Volatility (%)')
    ax2.set_title('Annualized Volatility by Sector')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Kurtosis by Sector (bar chart)
    ax3 = fig.add_subplot(gs[0, 2])
    df_kurt = df.sort_values('avg_kurtosis', ascending=True)
    colors = ['red' if k > 5 else 'steelblue' for k in df_kurt['avg_kurtosis']]
    ax3.barh(df_kurt['sector'].str[:15], df_kurt['avg_kurtosis'], color=colors, edgecolor='white')
    ax3.axvline(x=3, color='black', linestyle='--', alpha=0.5, label='Normal dist')
    ax3.set_xlabel('Avg Kurtosis')
    ax3.set_title('Fat Tails by Sector (>3 = fat)')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Duration vs Volatility Scatter
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(df['avg_volatility']*100, df['avg_duration'], 
                         s=df['n_stocks']*20, c=df['avg_kurtosis'], 
                         cmap='coolwarm', alpha=0.7, edgecolors='white', linewidth=1)
    for i, row in df.iterrows():
        ax4.annotate(row['sector'][:10], (row['avg_volatility']*100, row['avg_duration']),
                    fontsize=7, alpha=0.8)
    ax4.set_xlabel('Volatility (%)')
    ax4.set_ylabel('Duration (days)')
    ax4.set_title('Duration vs Volatility (size=n_stocks, color=kurtosis)')
    plt.colorbar(scatter, ax=ax4, label='Kurtosis')
    ax4.grid(True, alpha=0.3)
    
    # 5. Sector Fingerprint Heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    features = ['duration_z', 'volatility_z', 'kurtosis_z', 'jump_freq_z', 'vol_cluster_z']
    feature_labels = ['Duration', 'Volatility', 'Kurtosis', 'Jump Freq', 'Vol Cluster']
    
    heatmap_data = []
    sector_names = []
    for sector in df['sector']:
        fp = fingerprints[sector]
        heatmap_data.append([fp[f] for f in features])
        sector_names.append(sector[:12])
    
    heatmap_data = np.array(heatmap_data)
    im = ax5.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax5.set_xticks(range(len(feature_labels)))
    ax5.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=8)
    ax5.set_yticks(range(len(sector_names)))
    ax5.set_yticklabels(sector_names, fontsize=8)
    ax5.set_title('Sector Fingerprints (z-scores)')
    plt.colorbar(im, ax=ax5, label='Z-score')
    
    # 6. Cluster Dendrogram
    ax6 = fig.add_subplot(gs[1, 2])
    dendrogram(linkage_matrix, labels=[s[:10] for s in sectors], ax=ax6, 
               leaf_rotation=90, leaf_font_size=8)
    ax6.set_title('Sector Clustering (Ward)')
    ax6.set_ylabel('Distance')
    
    # 7. Box plot of Duration within Sectors
    ax7 = fig.add_subplot(gs[2, 0:2])
    sector_durations = {}
    for r in analyzer.results:
        sector = r['sector'][:12]
        if sector not in sector_durations:
            sector_durations[sector] = []
        sector_durations[sector].append(r['avg_duration'])
    
    # Sort by median
    sorted_sectors = sorted(sector_durations.items(), key=lambda x: np.median(x[1]))
    labels, data = zip(*sorted_sectors)
    
    bp = ax7.boxplot(data, labels=labels, patch_artist=True, vert=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax7.set_ylabel('Regime Duration (days)')
    ax7.set_title('Duration Distribution by Sector')
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Summary Statistics Table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Calculate cross-sector stats
    all_durations = [r['avg_duration'] for r in analyzer.results]
    all_kurtosis = [r['kurtosis'] for r in analyzer.results]
    
    summary = f"""
CROSS-SECTOR SUMMARY
====================

Total Stocks: {len(analyzer.results)}
Total Sectors: {len(df)}

Duration:
  Mean: {np.mean(all_durations):.1f} days
  Std:  {np.std(all_durations):.1f} days
  Range: [{np.min(all_durations):.0f}, {np.max(all_durations):.0f}]

Longest Regimes:
  {df.iloc[0]['sector']}: {df.iloc[0]['avg_duration']:.0f}d
  {df.iloc[1]['sector']}: {df.iloc[1]['avg_duration']:.0f}d

Shortest Regimes:
  {df.iloc[-1]['sector']}: {df.iloc[-1]['avg_duration']:.0f}d
  {df.iloc[-2]['sector']}: {df.iloc[-2]['avg_duration']:.0f}d

CLUSTER ANALYSIS
================
"""
    for cluster_id, cluster_sectors in clusters.items():
        summary += f"\nCluster {cluster_id}: {', '.join([s[:10] for s in cluster_sectors])}"
    
    ax8.text(0.05, 0.95, summary, transform=ax8.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('research/outputs/sector_regime_analysis.png', dpi=150, 
                facecolor='white', bbox_inches='tight')
    print("✓ Saved: research/outputs/sector_regime_analysis.png")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 70)
    print("SECTOR REGIME ANALYSIS")
    print("Leveraging 110-Stock Validation Data")
    print("=" * 70 + "\n")
    
    # Load data
    print("Loading validation data...")
    results = load_validation_data()
    print(f"✓ Loaded {len(results)} stocks\n")
    
    # Create analyzer
    analyzer = SectorRegimeAnalyzer(results)
    
    # 1. Compute sector statistics
    print("Computing sector statistics...")
    stats_df = analyzer.compute_sector_statistics()
    print("\n" + "=" * 70)
    print("SECTOR STATISTICS (sorted by avg duration)")
    print("=" * 70)
    print(stats_df[['sector', 'n_stocks', 'avg_duration', 'std_duration', 'avg_kurtosis', 'avg_volatility']].to_string(index=False))
    
    # 2. Create fingerprints
    print("\n" + "=" * 70)
    print("SECTOR FINGERPRINTS (z-scores)")
    print("=" * 70)
    fingerprints = analyzer.create_sector_fingerprints()
    for sector, fp in sorted(fingerprints.items()):
        labels = ', '.join(fp['labels'])
        print(f"{sector:25s} | dur:{fp['duration_z']:+.2f}  vol:{fp['volatility_z']:+.2f}  kurt:{fp['kurtosis_z']:+.2f} | {labels}")
    
    # 3. Cluster sectors
    print("\n" + "=" * 70)
    print("SECTOR CLUSTERS (similar behavioral patterns)")
    print("=" * 70)
    clusters, _, _ = analyzer.cluster_sectors(n_clusters=4)
    for cluster_id, sector_list in sorted(clusters.items()):
        print(f"\nCluster {cluster_id}:")
        for s in sector_list:
            fp = fingerprints[s]
            print(f"  - {s}: {', '.join(fp['labels'])}")
    
    # 4. Find similar sectors
    print("\n" + "=" * 70)
    print("SECTOR SIMILARITY (who trades like whom?)")
    print("=" * 70)
    for sector in ['Large Cap Tech', 'Financials', 'Utilities']:
        similar = analyzer.find_similar_sectors(sector, top_n=2)
        print(f"\n{sector} is most similar to:")
        for s, score in similar:
            print(f"  → {s} (similarity: {score:.2%})")
    
    # 5. Generate trading implications
    print("\n" + "=" * 70)
    print("TRADING IMPLICATIONS")
    print("=" * 70)
    implications = analyzer.trading_implications()
    for sector in ['Large Cap Tech', 'High Volatility', 'Utilities']:
        if sector in implications:
            impl = implications[sector]
            print(f"\n📊 {sector}:")
            print(f"   Regime horizon: {impl['regime_horizon']} ({impl['expected_duration_days']:.0f} days)")
            print(f"   Volatility: {impl['volatility_profile']}")
            print(f"   Tail risk: {impl['tail_risk']}")
            print(f"   Recommendations:")
            for rec in impl['recommendations'][:3]:
                print(f"     • {rec}")
    
    # 6. Create visualization
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 70)
    visualize_sector_analysis(analyzer)
    
    # 7. Save detailed report
    print("\nGenerating detailed report...")
    
    report = f"""# Sector Regime Analysis Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

Analysis of **{len(results)} stocks** across **{len(stats_df)} sectors** to understand 
sector-level regime behavior patterns.

## Key Findings

### 1. Regime Duration Varies Dramatically by Sector

| Sector | Avg Duration | Std | Volatility |
|--------|-------------|-----|------------|
"""
    for _, row in stats_df.head(5).iterrows():
        report += f"| {row['sector']} | {row['avg_duration']:.1f}d | {row['std_duration']:.1f} | {row['avg_volatility']*100:.1f}% |\n"
    
    report += """
### 2. Sector Clusters (Similar Behavior Patterns)

"""
    for cluster_id, sector_list in sorted(clusters.items()):
        report += f"**Cluster {cluster_id}:** {', '.join(sector_list)}\n\n"
    
    report += """
### 3. Sector Fingerprints

| Sector | Duration | Volatility | Kurtosis | Pattern |
|--------|----------|------------|----------|---------|
"""
    for sector, fp in sorted(fingerprints.items(), key=lambda x: x[1]['duration_z'], reverse=True):
        labels = ', '.join(fp['labels'])
        report += f"| {sector[:20]} | {fp['duration_z']:+.2f} | {fp['volatility_z']:+.2f} | {fp['kurtosis_z']:+.2f} | {labels} |\n"
    
    report += """
## Trading Implications

### For Long Regime Sectors (Materials, Communication, REITs)
- Use longer lookback windows for regime detection
- Trend-following strategies may outperform
- Lower rebalancing frequency

### For Short Regime Sectors (Healthcare, Consumer Staples, Utilities)
- Use shorter lookback windows
- Mean-reversion signals more relevant
- Higher rebalancing frequency may help

### For High Volatility Sectors (High Volatility, Large Cap Tech)
- Reduce position sizes relative to other sectors
- Wider stop-losses needed
- Consider volatility-scaling positions

### For Fat-Tail Sectors (Large Cap Tech, Communication)
- Use fat-tailed distributions in Monte Carlo simulations
- Account for jump risk in VaR calculations
- News-driven events are common

## Actionable Next Steps

1. **Implement sector-specific regime parameters** in RegimeRiskPlatform
2. **Create sector-based position sizing rules**
3. **Build sector rotation signals** based on regime state
4. **Test walk-forward validation** by sector

## Files Generated

- `sector_regime_analysis.png` - Visualization dashboard
- `sector_regime_analysis_report.md` - This report
"""
    
    with open('research/outputs/sector_regime_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print("✓ Saved: research/outputs/sector_regime_analysis_report.md")
    
    # 8. Export data for further analysis
    stats_df.to_csv('research/outputs/sector_statistics.csv', index=False)
    print("✓ Saved: research/outputs/sector_statistics.csv")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return analyzer, stats_df


if __name__ == '__main__':
    analyzer, stats_df = main()
