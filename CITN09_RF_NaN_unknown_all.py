import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils import class_weight
from joblib import dump, load
import warnings; warnings.simplefilter('ignore')

# load data
adata = sc.read(r'all_samples_GE_clusters.h5ad')
print ('loaded data')

filterNaN = False
filterNaive = False
filterGenes = False

# filter out NaNs
if (filterNaN):
    adata = adata[adata.obs['specificity'].notnull()]
    print('filtered out NaNs')
else: # change NaNs to "unknown"
    adata.obs['specificity'] = adata.obs['specificity'].fillna('unknown')
    print('NaN specificity changed to unknown')

# filter out Naive cells
if (filterNaive):
    adata = adata[adata.obs['cluster'] != 'Naive']
    print('filtered out Naive cells')


# filter out TCR/BCR/ribosomal/MT genes
if (filterGenes):
    tcr = adata.var['Gene_Name'].str.contains(r'^TR[ABGD][VDJC]')
    bcr = adata.var['Gene_Name'].str.contains(r'^IG[KGLH][JVCMGAED]')
    rib = adata.var['Gene_Name'].str.contains(r'^RP[SL][[:digit:]]|^RPLP[[:digit:]]|^RPSA')
    mt = adata.var['Gene_Name'].str.contains(r'^MT-')
    adata = adata[:,(~tcr & ~bcr & ~rib & ~mt)]
    print('filtered out TCR, BCR, Ribosomal, and MT genes')

# labels known specific vs unknown
X = pd.DataFrame(adata.X, columns=adata.var.index.values)

Y = adata.obs['specificity']
Y = pd.Series(Y, dtype="category")
Y_known = (Y == 'MCPyV')
print('labeled known MCPyV vs not')

#----tune hyperparameters

# set parameters, scoring metrics
params = {
    "eta": [0.1, 0.2, 0.25, 0.3, 0.35],
    "max_depth": [1, 2, 3, 4, 5, 6],
    "min_child_weight": [1, 2, 3, 4, 5, 6, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7],
    "n_estimators": [100, 500, 1000]
}
scoring = ['accuracy', 'f1', 'recall', 'precision']

# fit tuning model  
cross_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

xgb_model = XGBClassifier()

rs_model = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=100, scoring=scoring, n_jobs=32, refit='f1', cv=cross_val)

print('fitting RSCV model...')
rs_model.fit(X, Y_known)

# save model param results
print('saving RSCV model and results\n')
pd.DataFrame(rs_model.cv_results_).to_csv('RSCV_results_GE_NaN_unknown_all.csv')
dump(rs_model, 'RSCV_model_GE_NaN_unknown_all.joblib')

print(f'best score (f1): {rs_model.best_score_}')
print(f'best index: {rs_model.best_index_}')
print(f'best params: {rs_model.best_params_}')

for measure in scoring:
    measure = 'mean_test_' + measure
    print(measure)
    print(min(rs_model.cv_results_[measure]))
    print(max(rs_model.cv_results_[measure]))
