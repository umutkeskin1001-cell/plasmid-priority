import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Load data
from plasmid_priority.config import build_context
ctx = build_context()
scored = pd.read_csv(ctx.root / "data/scores/backbone_scored.tsv", sep="\t")
scored = scored[scored["spread_label"].notna()]
y = scored["spread_label"].astype(int).values

features = ["T_eff_norm", "H_specialization_norm", "A_eff_norm", "coherence_score", 
            "backbone_purity_norm", "assignment_confidence_norm", "mash_neighbor_distance_train_norm", 
            "replicon_architecture_norm", "clinical_context_fraction_norm", "A_recurrence_norm", 
            "pmlst_coherence_norm", "ecology_context_diversity_norm", "H_external_host_range_support", 
            "pmlst_presence_fraction_train", "amr_support_norm", "metadata_support_depth_norm", 
            "H_external_host_range_norm", "external_t_synergy_norm"]

X = scored[features].fillna(0).values

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_linear = []
auc_poly = []

for train_idx, test_idx in kf.split(X, y):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    # Linear
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    clf = LogisticRegressionCV(scoring="roc_auc", max_iter=1000)
    clf.fit(X_tr_s, y_tr)
    auc_linear.append(roc_auc_score(y_te, clf.predict_proba(X_te_s)[:, 1]))
    
    # Poly
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_tr_p = poly.fit_transform(X_tr_s)
    X_te_p = poly.transform(X_te_s)
    
    clf_p = LogisticRegressionCV(scoring="roc_auc", max_iter=1000, penalty='l1', solver='liblinear')
    clf_p.fit(X_tr_p, y_tr)
    auc_poly.append(roc_auc_score(y_te, clf_p.predict_proba(X_te_p)[:, 1]))

print(f"Linear AUC: {np.mean(auc_linear):.4f}")
print(f"Poly+L1 AUC: {np.mean(auc_poly):.4f}")
