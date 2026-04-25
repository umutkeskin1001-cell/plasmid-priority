#!/usr/bin/env python3
"""Train meta-sovereign ensemble using existing predictions."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from plasmid_priority.config import build_context

# Metrics
from plasmid_priority.validation.metrics import average_precision, brier_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("=" * 80)
print("🎯 META-SOVEREIGN ENSEMBLE - MODEL EĞİTİMİ")
print("=" * 80)

context = build_context(PROJECT_ROOT)

# Load predictions file
preds_file = context.data_dir / "analysis/module_a_predictions.tsv"
if not preds_file.exists():
    print(f"❌ Predictions file not found: {preds_file}")
    sys.exit(1)

print(f"\n[1/5] Loading predictions: {preds_file}")
df = pd.read_csv(preds_file, sep="\t")

print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns):,}")

# Find prediction columns
pred_cols = [c for c in df.columns if c.startswith("pred_")]
print(f"  Prediction columns: {len(pred_cols)}")

if len(pred_cols) < 3:
    print("❌ Not enough prediction columns found!")
    print(f"  Available: {list(df.columns)[:20]}")
    sys.exit(1)

# Find ground truth
target_col = None
for col in ["spread_label", "y_true", "target"]:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    print("❌ Ground truth column not found!")
    sys.exit(1)

print(f"  Target column: {target_col}")

# Prepare data
y = df[target_col].values

# Filter valid rows
valid_mask = ~pd.isna(y)
if valid_mask.sum() < 100:
    print("❌ Too few valid samples!")
    sys.exit(1)

y = y[valid_mask]
X = df.loc[valid_mask, pred_cols].fillna(0.5).values  # Fill missing with 0.5

print("\n[2/5] Data prepared")
print(f"  Samples: {len(y):,}")
print(f"  Positive rate: {y.mean():.3f}")  # type: ignore
print(f"  Features (base model preds): {len(pred_cols)}")

print("\n[3/5] Base model performances:")
base_aucs = {}
for i, col in enumerate(pred_cols):
    auc = roc_auc_score(y, X[:, i])  # type: ignore
    base_aucs[col] = auc
    print(f"  {col:40s}: {auc:.4f}")

# Sort by AUC
sorted_models = sorted(base_aucs.items(), key=lambda x: x[1], reverse=True)
top_models = [m[0] for m in sorted_models[:5]]
top_indices = [pred_cols.index(m) for m in top_models]

print("\n  Top 5 models selected for ensemble:")
for m in top_models:
    print(f"    - {m}: {base_aucs[m]:.4f}")

X_top = X[:, top_indices]

# Cross-validation training
print("\n[4/5] Training ensemble (5-fold CV)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_top, y)):
    X_train, X_val = X_top[train_idx], X_top[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Meta-learner: logistic regression
    meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    meta.fit(X_train, y_train)

    # Predict
    preds = meta.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = preds

    # Compute AUC
    auc = roc_auc_score(y_val, preds)
    fold_aucs.append(auc)
    print(f"  Fold {fold + 1}: AUC={auc:.4f}")

print("\n[5/5] Results")
print("-" * 80)

final_auc = roc_auc_score(y, oof_preds)  # type: ignore
final_ap = average_precision(y, oof_preds)  # type: ignore
final_brier = brier_score(y, oof_preds)  # type: ignore

# Calibrate
iso_cal = IsotonicRegression(out_of_bounds="clip")
iso_cal.fit(oof_preds.reshape(-1, 1), y)
cal_preds = iso_cal.predict(oof_preds.reshape(-1, 1))
cal_auc = roc_auc_score(y, cal_preds)  # type: ignore

print(f"  Mean CV AUC:     {np.mean(fold_aucs):.4f} (±{np.std(fold_aucs):.4f})")
print("")
print(f"  OOF ROC AUC:     {final_auc:.4f}")
print(f"  OOF AP:          {final_ap:.4f}")
print(f"  OOF Brier:       {final_brier:.4f}")
print(f"  Calibrated AUC:  {cal_auc:.4f}")

# Compare
best_base = max(base_aucs.values())
improvement = final_auc - best_base

print("\n  Comparison:")
print(f"    Best base model:   {best_base:.4f}")
print(f"    Ensemble:          {final_auc:.4f}")
print(f"    Improvement:       {improvement:+.4f}")

if final_auc >= 0.83:
    print("\n  ✅ 83+ AUC HEDEFİ ULAŞILDI! 🎉")
elif improvement > 0:
    print("\n  ✅ En iyi modeli geçti!")
else:
    print("\n  ⚠️  En iyi modelin altında")

# Save results
results = {
    "model_name": "meta_sovereign_ensemble",
    "roc_auc": float(final_auc),
    "average_precision": float(final_ap),
    "brier_score": float(final_brier),
    "calibrated_roc_auc": float(cal_auc),
    "fold_aucs": [float(a) for a in fold_aucs],
    "mean_fold_auc": float(np.mean(fold_aucs)),
    "std_fold_auc": float(np.std(fold_aucs)),
    "best_base_auc": float(best_base),
    "improvement": float(improvement),
    "n_base_models": len(pred_cols),
    "n_top_models_used": len(top_models),
    "target_83_achieved": final_auc >= 0.83,
}

output_dir = context.data_dir / "analysis"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "meta_sovereign_ensemble_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved: {output_file}")

# Save predictions
pred_df = pd.DataFrame(
    {
        "y_true": y,
        "y_pred": oof_preds,
        "y_pred_calibrated": cal_preds,
    },
)

pred_file = output_dir / "meta_sovereign_ensemble_predictions.tsv"
pred_df.to_csv(pred_file, sep="\t", index=False)
print(f"💾 Predictions saved: {pred_file}")

print("=" * 80)
print("✅ EĞİTİM TAMAMLANDI")
print("=" * 80)
