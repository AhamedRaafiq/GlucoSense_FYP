import os
import sys
import json
import pickle
import argparse
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def compute_node_depths(df):
    depths = {}
    for tree_id in df['Tree'].unique():
        tree_df = df[df['Tree'] == tree_id]
        
        # Build children map for fast lookup
        children = {}
        for _, row in tree_df.iterrows():
            if row['Feature'] != 'Leaf':
                children[row['ID']] = (row['Yes'], row['No'])
        
        # Find root node (node in tree_df that is not a child of any node)
        all_children = set(tree_df['Yes'].dropna()).union(set(tree_df['No'].dropna()))
        root_nodes = [nid for nid in tree_df['ID'] if nid not in all_children]
        root_id = root_nodes[0] if root_nodes else f"{tree_id}-0"
        
        # BFS to find depth of each node in the tree
        queue = [(root_id, 0)]
        while queue:
            node_id, depth = queue.pop(0)
            depths[node_id] = depth
            if node_id in children:
                left, right = children[node_id]
                queue.append((left, depth + 1))
                queue.append((right, depth + 1))
    return depths

def main():
    parser = argparse.ArgumentParser(description="Extract structure and metrics from trained XGBoost model.")
    parser.add_argument("--model-path", type=str, help="Path to xgboost model (.json or .pkl)")
    parser.add_argument("--out-dir", type=str, help="Directory to save the reports")
    args = parser.parse_args()
    
    # Auto-discover if not provided
    if not args.model_path:
        base_dir = Path("C:/Users/DELL/Documents/GitHub/fyp/08_Results_and_Visualizations/XGBoost_Results_&_Conclusions")
        dirs = glob.glob(str(base_dir / "XGBoost results & Conclusions *"))
        if dirs:
            dirs.sort()
            latest_dir = Path(dirs[-1])
            print(f"Auto-discovered latest model directory: {latest_dir.name}")
            # Try to load .json first, then .pkl
            json_path = latest_dir / "model" / "xgboost_glucose_model.json"
            pkl_path = latest_dir / "model" / "xgboost_glucose_model.pkl"
            if json_path.exists():
                args.model_path = str(json_path)
            elif pkl_path.exists():
                args.model_path = str(pkl_path)
            
            if not args.out_dir:
                args.out_dir = str(latest_dir / "report")
        else:
            print("Error: No models found. Please specify --model-path.")
            sys.exit(1)
            
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist.")
        sys.exit(1)
        
    print(f"Loading model from: {model_path}")
    
    # Load model
    booster = xgb.Booster()
    if model_path.suffix == '.json':
        booster.load_model(str(model_path))
    elif model_path.suffix == '.pkl':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        if isinstance(model, xgb.Booster):
            booster = model
        else:
            booster = model.get_booster()
    else:
        # Try both load methods as fallback
        try:
            booster.load_model(str(model_path))
        except Exception:
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                booster = model if isinstance(model, xgb.Booster) else model.get_booster()
            except Exception as e:
                print(f"Error loading model: {e}")
                sys.exit(1)
                
    # 1. Convert booster to dataframe
    df = booster.trees_to_dataframe()
    
    # 2. Compute depths
    depths_dict = compute_node_depths(df)
    df['Depth'] = df['ID'].map(depths_dict)
    
    # --- CALCULATE METRICS ---
    
    # 1. Number of trees
    num_trees = int(df['Tree'].nunique())
    
    # 2. Tree depth
    tree_max_depths = df.groupby('Tree')['Depth'].max()
    depth_stats = {
        'max_depth_overall': int(df['Depth'].max()),
        'mean_max_depth': float(tree_max_depths.mean()),
        'min_max_depth': int(tree_max_depths.min()),
        'max_max_depth': int(tree_max_depths.max()),
        'all_max_depths': [int(d) for d in tree_max_depths.tolist()]
    }
    
    # 3. Number of nodes
    total_nodes = len(df)
    nodes_per_tree = df.groupby('Tree')['Node'].count()
    node_stats = {
        'total_nodes': total_nodes,
        'mean_nodes_per_tree': float(nodes_per_tree.mean()),
        'min_nodes_per_tree': int(nodes_per_tree.min()),
        'max_nodes_per_tree': int(nodes_per_tree.max()),
        'all_nodes_per_tree': [int(n) for n in nodes_per_tree.tolist()]
    }
    
    # 4. Number of leaf nodes
    leaf_df = df[df['Feature'] == 'Leaf']
    total_leaves = len(leaf_df)
    leaves_per_tree = leaf_df.groupby('Tree')['Node'].count()
    leaf_stats = {
        'total_leaves': total_leaves,
        'mean_leaves_per_tree': float(leaves_per_tree.mean()),
        'min_leaves_per_tree': int(leaves_per_tree.min()),
        'max_leaves_per_tree': int(leaves_per_tree.max()),
        'all_leaves_per_tree': [int(l) for l in leaves_per_tree.tolist()]
    }
    
    # 5. Split features
    split_df = df[df['Feature'] != 'Leaf']
    split_features_counts = split_df['Feature'].value_counts()
    split_features_list = sorted(split_features_counts.index.tolist())
    
    # 6. Split thresholds
    threshold_stats = {}
    for feat in split_features_list:
        feat_thresholds = split_df[split_df['Feature'] == feat]['Split'].dropna().tolist()
        threshold_stats[feat] = {
            'count': len(feat_thresholds),
            'min': float(np.min(feat_thresholds)),
            'max': float(np.max(feat_thresholds)),
            'mean': float(np.mean(feat_thresholds)),
            'std': float(np.std(feat_thresholds)) if len(feat_thresholds) > 1 else 0.0,
            'thresholds': sorted([float(t) for t in feat_thresholds])
        }
        
    # 7. Leaf values
    leaf_values = leaf_df['Gain'].dropna().tolist() # Gain column stores the leaf prediction value for leaves
    leaf_value_stats = {
        'count': len(leaf_values),
        'min': float(np.min(leaf_values)),
        'max': float(np.max(leaf_values)),
        'mean': float(np.mean(leaf_values)),
        'std': float(np.std(leaf_values)) if len(leaf_values) > 1 else 0.0,
        'leaf_values': sorted([float(v) for v in leaf_values])
    }
    
    # 8-11. Feature Importance, Gain, Weight, Cover
    # Retrieve all types of scores
    importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    feature_scores = {}
    for imp_type in importance_types:
        score = booster.get_score(importance_type=imp_type)
        # Populate all features, even if they have 0 score
        full_score = {feat: float(score.get(feat, 0.0)) for feat in split_features_list}
        feature_scores[imp_type] = full_score
        
    # Normalize Total Gain to get percentage importance
    total_gain_sum = sum(feature_scores['total_gain'].values())
    feature_scores['gain_percentage'] = {
        feat: (val / total_gain_sum * 100.0) if total_gain_sum > 0 else 0.0
        for feat, val in feature_scores['total_gain'].items()
    }
    
    # 12. Overfitting / Underfitting Analysis — paths resolved after out_dir is set below
    fit_analysis = None  # filled in after out_dir is determined

    # Create final metrics dict
    analysis = {
        'model_metadata': {
            'loaded_from': str(model_path.resolve()),
            'file_format': model_path.suffix
        },
        'number_of_trees': num_trees,
        'tree_depth': depth_stats,
        'number_of_nodes': node_stats,
        'number_of_leaf_nodes': leaf_stats,
        'split_features': split_features_list,
        'split_thresholds': threshold_stats,
        'leaf_values': leaf_value_stats,
        'feature_importance_metrics': feature_scores,
        'fit_analysis': {}  # placeholder; filled after out_dir is resolved
    }
    
    # Save output directory
    out_dir = Path(args.out_dir) if args.out_dir else model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 12. Overfitting / Underfitting Analysis — resolved now that out_dir is known
    pred_dir = out_dir.parent / "predictions" if args.out_dir else model_path.parent.parent / "predictions"
    train_pred_path = pred_dir / "train_predictions.csv"
    test_pred_path  = pred_dir / "test_predictions.csv"
    fit_analysis = compute_fit_analysis(train_pred_path, test_pred_path)
    analysis['fit_analysis'] = fit_analysis
    
    json_out_path = out_dir / "xgboost_model_parameters_details.json"
    md_out_path = out_dir / "xgboost_model_parameters_report.md"
    
    # Save JSON details
    with open(json_out_path, 'w') as f:
        json.dump(analysis, f, indent=4)
        
    # Save Markdown report
    markdown_content = generate_markdown_report(analysis)
    with open(md_out_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
        
    print(f"\nAnalysis completed successfully!")
    print(f"JSON data saved to: {json_out_path.resolve()}")
    print(f"Markdown report saved to: {md_out_path.resolve()}")
    
    # Print high-level summary to console
    print("\n" + "="*55)
    print("XGBOOST MODEL PARAMETERS SUMMARY")
    print("="*55)
    print(f"Number of Trees:       {num_trees}")
    print(f"Tree Max Depth:        {depth_stats['max_depth_overall']} (Mean tree max depth: {depth_stats['mean_max_depth']:.2f})")
    print(f"Total Nodes:           {total_nodes} (Mean nodes/tree: {node_stats['mean_nodes_per_tree']:.2f})")
    print(f"Total Leaf Nodes:      {total_leaves} (Mean leaves/tree: {leaf_stats['mean_leaves_per_tree']:.2f})")
    print(f"Leaf Value Range:      [{leaf_value_stats['min']:.4f}, {leaf_value_stats['max']:.4f}] (Mean: {leaf_value_stats['mean']:.4f})")
    print("="*55)
    print("TOP 5 FEATURES BY GAIN IMPORTANCE:")
    sorted_by_gain = sorted(feature_scores['gain_percentage'].items(), key=lambda x: x[1], reverse=True)
    for i, (feat, pct) in enumerate(sorted_by_gain[:5], 1):
        print(f" {i}. {feat:<30} {pct:.2f}% (Total Gain: {feature_scores['total_gain'][feat]:.1f})")
    print("="*55)
    if fit_analysis.get('status') == 'ok':
        fa = fit_analysis
        print("OVERFITTING / UNDERFITTING ANALYSIS:")
        print(f" Train  R²={fa['train']['r2']:.4f}  RMSE={fa['train']['rmse']:.4f}  MAE={fa['train']['mae']:.4f}  MAPE={fa['train']['mape']:.2f}%")
        print(f" Test   R²={fa['test']['r2']:.4f}  RMSE={fa['test']['rmse']:.4f}  MAE={fa['test']['mae']:.4f}  MAPE={fa['test']['mape']:.2f}%")
        print(f" R² Gap (Train-Test):   {fa['gaps']['r2_gap']:+.4f}")
        print(f" RMSE Ratio (Test/Train): {fa['gaps']['rmse_ratio']:.4f}")
        print(f" MAE  Ratio (Test/Train): {fa['gaps']['mae_ratio']:.4f}")
        print(f" MAPE Ratio (Test/Train): {fa['gaps']['mape_ratio']:.4f}")
        print(f" Overall Verdict:       >>> {fa['verdict']} <<<")
        print(f" Overfitting Score:     {fa['overfitting_score']:.4f}  (0=none, >0=overfit)")
        print(f" Underfitting Score:    {fa['underfitting_score']:.4f}  (0=none, >0=underfit)")
    else:
        print(f" Fit Analysis: {fit_analysis.get('error', 'N/A')}")
    print("="*55 + "\n")

def compute_fit_analysis(train_pred_path, test_pred_path):
    """
    Load train/test prediction CSVs and compute overfitting / underfitting ratios.

    Metrics computed per split: R², RMSE, MAE, MAPE.
    Ratios computed (Test / Train for error metrics, Train-Test gap for R²).

    Verdict logic:
      - Overfitting  : R² gap > 0.05 OR RMSE ratio > 1.15 OR MAE ratio > 1.15
      - Underfitting : train R² < 0.70 AND test R² < 0.70
      - Well-Generalised: otherwise
    """
    if not Path(train_pred_path).exists() or not Path(test_pred_path).exists():
        return {'status': 'error', 'error': f'Prediction CSVs not found. Looked in: {train_pred_path}'}

    train_df = pd.read_csv(train_pred_path)
    test_df  = pd.read_csv(test_pred_path)

    def _metrics(df):
        actual = df['Actual_Glucose_mg_dL'].values
        pred   = df['Predicted_Glucose_mg_dL'].values
        r2   = float(r2_score(actual, pred))
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        mae  = float(mean_absolute_error(actual, pred))
        # MAPE — avoid division by zero
        mape = float(np.mean(np.abs((actual - pred) / np.where(actual == 0, 1e-9, actual))) * 100.0)
        n    = len(actual)
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape, 'n_samples': n}

    train_m = _metrics(train_df)
    test_m  = _metrics(test_df)

    # Gap / ratio metrics
    r2_gap     = train_m['r2']   - test_m['r2']        # positive = overfit tendency
    rmse_ratio = test_m['rmse']  / max(train_m['rmse'], 1e-9)
    mae_ratio  = test_m['mae']   / max(train_m['mae'],  1e-9)
    mape_ratio = test_m['mape']  / max(train_m['mape'], 1e-9)

    # Composite overfitting score (0 = perfect generalisation, positive = degree of overfit)
    overfitting_score  = max(0.0, r2_gap) * 0.4 + max(0.0, rmse_ratio - 1.0) * 0.3 + max(0.0, mae_ratio - 1.0) * 0.3
    # Underfitting score: penalise low train R² and low test R²
    underfitting_score = max(0.0, 0.70 - train_m['r2']) * 0.5 + max(0.0, 0.70 - test_m['r2']) * 0.5

    # Verdict
    if r2_gap > 0.05 or rmse_ratio > 1.15 or mae_ratio > 1.15:
        verdict = 'OVERFITTING'
    elif train_m['r2'] < 0.70 and test_m['r2'] < 0.70:
        verdict = 'UNDERFITTING'
    else:
        verdict = 'WELL-GENERALISED'

    return {
        'status': 'ok',
        'train': train_m,
        'test':  test_m,
        'gaps': {
            'r2_gap':     float(r2_gap),
            'rmse_ratio': float(rmse_ratio),
            'mae_ratio':  float(mae_ratio),
            'mape_ratio': float(mape_ratio)
        },
        'overfitting_score':  float(overfitting_score),
        'underfitting_score': float(underfitting_score),
        'verdict': verdict,
        'thresholds_used': {
            'r2_gap_overfit_threshold':    0.05,
            'rmse_ratio_overfit_threshold':1.15,
            'mae_ratio_overfit_threshold': 1.15,
            'r2_underfit_threshold':       0.70
        }
    }


def generate_markdown_report(analysis):
    m = []
    m.append("# XGBoost Model Parameter Analysis Report")
    m.append("")
    m.append(f"**Model Loaded From:** `{analysis['model_metadata']['loaded_from']}`  ")
    m.append(f"**File Format:** `{analysis['model_metadata']['file_format']}`")
    m.append("")
    m.append("## 1. High-Level Model Structure")
    m.append("")
    m.append("| Metric | Value | Details / Range |")
    m.append("|---|---|---|")
    m.append(f"| **Number of Trees** | {analysis['number_of_trees']} | Boosting iterations |")
    m.append(f"| **Tree Depth** | {analysis['tree_depth']['max_depth_overall']} max | Range of max depths: {analysis['tree_depth']['min_max_depth']} to {analysis['tree_depth']['max_max_depth']} (Mean: {analysis['tree_depth']['mean_max_depth']:.2f}) |")
    m.append(f"| **Total Nodes** | {analysis['number_of_nodes']['total_nodes']} | Range per tree: {analysis['number_of_nodes']['min_nodes_per_tree']} to {analysis['number_of_nodes']['max_nodes_per_tree']} (Mean: {analysis['number_of_nodes']['mean_nodes_per_tree']:.2f}) |")
    m.append(f"| **Total Leaf Nodes** | {analysis['number_of_leaf_nodes']['total_leaves']} | Range per tree: {analysis['number_of_leaf_nodes']['min_leaves_per_tree']} to {analysis['number_of_leaf_nodes']['max_leaves_per_tree']} (Mean: {analysis['number_of_leaf_nodes']['mean_leaves_per_tree']:.2f}) |")
    m.append("")
    
    m.append("## 2. Leaf Value Distribution")
    m.append("")
    leaf = analysis['leaf_values']
    m.append(f"- **Count:** {leaf['count']} leaf predictions")
    m.append(f"- **Minimum:** `{leaf['min']:.6f}`")
    m.append(f"- **Maximum:** `{leaf['max']:.6f}`")
    m.append(f"- **Mean:** `{leaf['mean']:.6f}`")
    m.append(f"- **Standard Deviation:** `{leaf['std']:.6f}`")
    m.append("")
    
    m.append("## 3. Feature Importance Metrics")
    m.append("")
    m.append("XGBoost calculates feature importance using multiple criteria:")
    m.append("1. **Weight / Frequency**: Number of times the feature is split on across all trees.")
    m.append("2. **Gain (Average)**: Average gain contribution of splits using this feature.")
    m.append("3. **Cover (Average)**: Average coverage (number of samples affected) of splits using this feature.")
    m.append("4. **Total Gain**: Sum of gain contribution across all splits of this feature.")
    m.append("5. **Total Cover**: Sum of coverage across all splits of this feature.")
    m.append("6. **Gain %**: Normalized Total Gain (representing the relative contribution of each feature to the model's predictions).")
    m.append("")
    
    m.append("| Feature | Weight (Splits) | Avg Gain | Avg Cover | Total Gain | Total Cover | Gain % |")
    m.append("|---|---|---|---|---|---|---|")
    
    f_metrics = analysis['feature_importance_metrics']
    sorted_features = sorted(f_metrics['gain_percentage'].items(), key=lambda x: x[1], reverse=True)
    for feat, pct in sorted_features:
        w = f_metrics['weight'].get(feat, 0)
        g = f_metrics['gain'].get(feat, 0)
        c = f_metrics['cover'].get(feat, 0)
        tg = f_metrics['total_gain'].get(feat, 0)
        tc = f_metrics['total_cover'].get(feat, 0)
        m.append(f"| **{feat}** | {w:.0f} | {g:.4f} | {c:.4f} | {tg:.2f} | {tc:.2f} | {pct:.2f}% |")
    m.append("")
    
    m.append("## 4. Split Threshold Statistics per Feature")
    m.append("")
    m.append("| Feature | Splits | Min Threshold | Max Threshold | Mean Threshold | Threshold SD |")
    m.append("|---|---|---|---|---|---|")
    for feat in analysis['split_features']:
        t_stat = analysis['split_thresholds'][feat]
        m.append(f"| **{feat}** | {t_stat['count']} | {t_stat['min']:.6f} | {t_stat['max']:.6f} | {t_stat['mean']:.6f} | {t_stat['std']:.6f} |")
    m.append("")
    m.append("*(Note: Detailed threshold values are saved in the raw JSON file `xgboost_model_parameters_details.json`)*")
    m.append("")

    # ── Section 5: Overfitting / Underfitting Analysis ─────────────────────────
    m.append("## 5. Overfitting / Underfitting Analysis")
    m.append("")
    fa = analysis.get('fit_analysis', {})
    if fa.get('status') != 'ok':
        m.append(f"> ⚠️ Could not compute fit analysis: {fa.get('error', 'Unknown error')}")
    else:
        verdict = fa['verdict']
        verdict_emoji = {'OVERFITTING': '🔴', 'UNDERFITTING': '🟡', 'WELL-GENERALISED': '🟢'}.get(verdict, '⚪')

        m.append(f"### Overall Verdict: {verdict_emoji} **{verdict}**")
        m.append("")
        m.append("> **How to read this section:**  ")
        m.append("> - **R² gap** = Train R² − Test R². Large positive gap → overfitting.  ")
        m.append("> - **Ratio metrics** = Test metric ÷ Train metric. >1 means test errors are larger than train errors.  ")
        m.append("> - **Overfitting Score** is a composite of R² gap + RMSE ratio + MAE ratio (0 = perfect, higher = worse).  ")
        m.append("> - **Underfitting Score** penalises low R² on both splits (0 = no underfitting).")
        m.append("")

        m.append("#### 5.1 Per-Split Performance Metrics")
        m.append("")
        m.append("| Metric | Train Set | Test Set |")
        m.append("|---|---|---|")
        m.append(f"| **Samples** | {fa['train']['n_samples']} | {fa['test']['n_samples']} |")
        m.append(f"| **R² (Coefficient of Determination)** | {fa['train']['r2']:.6f} | {fa['test']['r2']:.6f} |")
        m.append(f"| **RMSE (mg/dL)** | {fa['train']['rmse']:.6f} | {fa['test']['rmse']:.6f} |")
        m.append(f"| **MAE (mg/dL)** | {fa['train']['mae']:.6f} | {fa['test']['mae']:.6f} |")
        m.append(f"| **MAPE (%)** | {fa['train']['mape']:.4f}% | {fa['test']['mape']:.4f}% |")
        m.append("")

        m.append("#### 5.2 Generalisation Gap / Ratio Metrics")
        m.append("")
        m.append("| Metric | Value | Interpretation |")
        m.append("|---|---|---|")
        r2_g = fa['gaps']['r2_gap']
        rmse_r = fa['gaps']['rmse_ratio']
        mae_r  = fa['gaps']['mae_ratio']
        mape_r = fa['gaps']['mape_ratio']
        m.append(f"| **R² Gap (Train − Test)** | {r2_g:+.6f} | Threshold: >0.05 → overfitting risk |")
        m.append(f"| **RMSE Ratio (Test / Train)** | {rmse_r:.6f} | Threshold: >1.15 → overfitting risk |")
        m.append(f"| **MAE Ratio (Test / Train)** | {mae_r:.6f} | Threshold: >1.15 → overfitting risk |")
        m.append(f"| **MAPE Ratio (Test / Train)** | {mape_r:.6f} | Informational only |")
        m.append(f"| **Overfitting Score** | {fa['overfitting_score']:.6f} | 0 = none, higher = more overfit |")
        m.append(f"| **Underfitting Score** | {fa['underfitting_score']:.6f} | 0 = none, higher = more underfit |")
        m.append("")

        m.append("#### 5.3 Verdict Summary")
        m.append("")
        m.append(f"| Verdict | {verdict_emoji} {verdict} |")
        m.append("|---|---|")
        thresh = fa['thresholds_used']
        m.append(f"| R² Gap threshold | {thresh['r2_gap_overfit_threshold']} |")
        m.append(f"| RMSE ratio threshold | {thresh['rmse_ratio_overfit_threshold']} |")
        m.append(f"| MAE ratio threshold | {thresh['mae_ratio_overfit_threshold']} |")
        m.append(f"| Underfit R² threshold (both splits) | {thresh['r2_underfit_threshold']} |")
        m.append("")

    return "\n".join(m)

if __name__ == "__main__":
    main()
