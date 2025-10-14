import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


colors_dict = {'mse':'tab:orange',  
'comboptnet': 'navy',
'sfl': 'c', 
'odece0.2':'m',
'odece0.3':'m',
'odece0.4':'m',
'odece0.5':'m',
'odece0.6':'m',
'odece0.7':'m',
'odece0.8':'m',
'TwoStageIntOpt' :'g'
        }

markers_dict = {'mse':'o',  
'comboptnet': 'P',
'sfl': 'X', 
'odece0.2':'v',
'odece0.3':'>',
'odece0.4':'1',
'odece0.5':'s',
'odece0.6':'2',
'odece0.7': '<',
'odece0.8':'^',
'TwoStageIntOpt' : 'D'
        }
ial_reduction = 'mean'#'mean'
### First Knapsack Weights
infeasibility_aversion_coeff = 0.2

knapsack_weights = pd.read_csv( "../CombinedResults/KnapsackWeightsNoFixedCosts.csv")

filtered = knapsack_weights[knapsack_weights['val_infeasibility'].notna()]
model_param_map = {
    'comboptnet': ['tau'],
    'sfl': ['temp'],
    'odece': ['ial_reduction', "use_pcgrad", "fix_alpha",'infeasibility_aversion_coeff'],
    'mse': []  # No extra parameter
}

# all_best = []
# ### Choosing best lr
# for model, params in model_param_map.items():
#     # Get only this model's rows
#     model_df = filtered[filtered['model'] == model]
#     # Group by params + lr
#     group_cols = ['model'] + params + ['lr']
#     # Find last epoch for each group
#     last_epochs = model_df.groupby(group_cols)['epoch'].max().reset_index()
#     # Merge to get the last valid epoch rows
#     last_epoch_rows = pd.merge(model_df, last_epochs, on=group_cols + ['epoch'])
#     # For each unique param set, pick lr with min val_infeasibility
#     if model == 'TwoStageIntOpt':
#         if params:
#             idx = last_epoch_rows.groupby(['model'] + params)['val_posthoc_regret'].idxmin()
#         else:
#             idx = last_epoch_rows.groupby(['model'])['val_posthoc_regret'].idxmin()
#     else:
#         if params:
#             idx = last_epoch_rows.groupby(['model'] + params)['val_infeasibility'].idxmin()
#         else:
#             idx = last_epoch_rows.groupby(['model'])['val_infeasibility'].idxmin()
#     best = last_epoch_rows.loc[idx]
#     all_best.append(best)

# best_lrs = pd.concat(all_best, ignore_index=True)

# df_selected = []
# for _, row in best_lrs.iterrows():
#     model = row['model']
#     lr = row['lr']
#     params = model_param_map[model]
#     # Filter for this model, lr, and parameter values
#     cond = (knapsack_weights['model'] == model) & (knapsack_weights['lr'] == lr)
#     for p in params:
#         cond &= (knapsack_weights[p] == row[p])
#     df_plot = knapsack_weights[cond]
#     df_selected.append (df_plot)
# df_selected = pd.concat ( df_selected, ignore_index=True)

df_selected = knapsack_weights
### Filtering parameter set
model_filters = {
    'odece': {'use_pcgrad': True, 'ial_reduction': ial_reduction,  'lr':0.05,
                                        'infeasibility_aversion_coeff':infeasibility_aversion_coeff,
                                        'degree':6, 'noise':0.25  },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
    'sfl':{"temp": 0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
    'mse': {'lr':0.05,
                                        'degree':6, 'noise':0.25  },
}
filtered_dfs = []
for model in df_selected['model'].unique():
    model_df = df_selected[df_selected['model'] == model]
    if model in model_filters:
        for col, val in model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    filtered_dfs.append(model_df)
filtered_custom = pd.concat(filtered_dfs, ignore_index=True)
filtered_custom['model']= filtered_custom['model'].where(( filtered_custom['model']!='odece' ),
                                           filtered_custom['model'].astype(str) + filtered_custom['infeasibility_aversion_coeff'].astype(str) )
# filtered_custom = filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
#                                                      "test_regret":['mean','std'],}).reset_index()

# # Flatten MultiIndex columns
# filtered_custom.columns = [
#     col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
#     for col in filtered_custom.columns.values
# ]
combined_order = filtered_custom.model.unique().tolist()
combined_order = filtered_custom.model.unique().tolist()
print (combined_order)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
# Validation Infeasibility vs Epoch
sns.lineplot(
    data=filtered_custom,
    x='epoch',
    y='val_infeasibility',
    hue='model',
    palette=colors_dict,
    ax=axes[0],
    marker='o'
)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Infeasibility')
axes[0].set_title('Validation Infeasibility vs Epoch')
axes[0].legend()
axes[0].grid(True)

# Validation Regret vs Epoch
sns.lineplot(
    data=filtered_custom,
    x='epoch',
    y='val_regret',
    hue='model',
    palette=colors_dict,
    ax=axes[1],
    marker='o'
)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Regret')
axes[1].set_title('Validation Regret vs Epoch')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"figures/LRPlot_KPWeight_alpha{infeasibility_aversion_coeff}_IAL_{ial_reduction}_PCGradNoFixedCost.png")

plt.close()

### Filtering parameter set
model_filters = {
    'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction, 'lr':0.05,
                                        'infeasibility_aversion_coeff':infeasibility_aversion_coeff,
                                        'degree':6, 'noise':0.25  },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
    'sfl':{"temp": 0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
}

filtered_dfs = []
for model in df_selected['model'].unique():
    model_df = df_selected[df_selected['model'] == model]
    if model in model_filters:
        for col, val in model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    filtered_dfs.append(model_df)
filtered_custom = pd.concat(filtered_dfs, ignore_index=True)
filtered_custom['model']= filtered_custom['model'].where(( filtered_custom['model']!='odece' ),
                                           filtered_custom['model'].astype(str) + filtered_custom['infeasibility_aversion_coeff'].astype(str) )
# filtered_custom = filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
#                                                      "test_regret":['mean','std'],}).reset_index()

# # Flatten MultiIndex columns
# filtered_custom.columns = [
#     col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
#     for col in filtered_custom.columns.values
# ]
combined_order = filtered_custom.model.unique().tolist()
combined_order = filtered_custom.model.unique().tolist()
print (combined_order)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
# Validation Infeasibility vs Epoch
sns.lineplot(
    data=filtered_custom,
    x='epoch',
    y='val_infeasibility',
    hue='model',
    palette=colors_dict,
    ax=axes[0],
    marker='o'
)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Infeasibility')
axes[0].set_title('Validation Infeasibility vs Epoch')
axes[0].legend()
axes[0].grid(True)

# Validation Regret vs Epoch
sns.lineplot(
    data=filtered_custom,
    x='epoch',
    y='val_regret',
    hue='model',
    palette=colors_dict,
    ax=axes[1],
    marker='o'
)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Regret')
axes[1].set_title('Validation Regret vs Epoch')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"figures/LRPlot_KPWeight_alpha{infeasibility_aversion_coeff}_IAL_{ial_reduction}_FixedAlphaNoFixedCost.png")

plt.close()



knapsack_weights = pd.read_csv( "../CombinedResults/KnapsackCapacityNoFixedCosts.csv")

filtered = knapsack_weights[knapsack_weights['val_infeasibility'].notna()]
model_param_map = {
    'comboptnet': ['tau'],
    'sfl': ['temp'],
    'odece': ['ial_reduction', "use_pcgrad", "fix_alpha",'infeasibility_aversion_coeff'],
    'mse': []  # No extra parameter
}

# all_best = []
# ### Choosing best lr
# for model, params in model_param_map.items():
#     # Get only this model's rows
#     model_df = filtered[filtered['model'] == model]
#     # Group by params + lr
#     group_cols = ['model'] + params + ['lr']
#     # Find last epoch for each group
#     last_epochs = model_df.groupby(group_cols)['epoch'].max().reset_index()
#     # Merge to get the last valid epoch rows
#     last_epoch_rows = pd.merge(model_df, last_epochs, on=group_cols + ['epoch'])
#     # For each unique param set, pick lr with min val_infeasibility
#     if model == 'TwoStageIntOpt':
#         if params:
#             idx = last_epoch_rows.groupby(['model'] + params)['val_posthoc_regret'].idxmin()
#         else:
#             idx = last_epoch_rows.groupby(['model'])['val_posthoc_regret'].idxmin()
#     else:
#         if params:
#             idx = last_epoch_rows.groupby(['model'] + params)['val_infeasibility'].idxmin()
#         else:
#             idx = last_epoch_rows.groupby(['model'])['val_infeasibility'].idxmin()
#     best = last_epoch_rows.loc[idx]
#     all_best.append(best)

# best_lrs = pd.concat(all_best, ignore_index=True)

# df_selected = []
# for _, row in best_lrs.iterrows():
#     model = row['model']
#     lr = row['lr']
#     params = model_param_map[model]
#     # Filter for this model, lr, and parameter values
#     cond = (knapsack_weights['model'] == model) & (knapsack_weights['lr'] == lr)
#     for p in params:
#         cond &= (knapsack_weights[p] == row[p])
#     df_plot = knapsack_weights[cond]
#     df_selected.append (df_plot)
# df_selected = pd.concat ( df_selected, ignore_index=True)

df_selected = knapsack_weights
### Filtering parameter set
model_filters = {
    'odece': {'use_pcgrad': True, 'ial_reduction': ial_reduction, 'lr':0.001,
                                        'infeasibility_aversion_coeff':infeasibility_aversion_coeff,
                                        'degree':6, 'noise':0.25  },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
    'sfl':{"temp": 0.1, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
}

filtered_dfs = []
for model in df_selected['model'].unique():
    model_df = df_selected[df_selected['model'] == model]
    if model in model_filters:
        for col, val in model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    filtered_dfs.append(model_df)
filtered_custom = pd.concat(filtered_dfs, ignore_index=True)
filtered_custom['model']= filtered_custom['model'].where(( filtered_custom['model']!='odece' ),
                                           filtered_custom['model'].astype(str) + filtered_custom['infeasibility_aversion_coeff'].astype(str) )
# filtered_custom = filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
#                                                      "test_regret":['mean','std'],}).reset_index()

# # Flatten MultiIndex columns
# filtered_custom.columns = [
#     col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
#     for col in filtered_custom.columns.values
# ]
combined_order = filtered_custom.model.unique().tolist()
combined_order = filtered_custom.model.unique().tolist()
print (combined_order)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
# Validation Infeasibility vs Epoch
sns.lineplot(
    data=filtered_custom,
    x='epoch',
    y='val_infeasibility',
    hue='model',
    palette=colors_dict,
    ax=axes[0],
    marker='o'
)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Infeasibility')
axes[0].set_title('Validation Infeasibility vs Epoch')
axes[0].legend()
axes[0].grid(True)

# Validation Regret vs Epoch
sns.lineplot(
    data=filtered_custom,
    x='epoch',
    y='val_regret',
    hue='model',
    palette=colors_dict,
    ax=axes[1],
    marker='o'
)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Regret')
axes[1].set_title('Validation Regret vs Epoch')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"figures/LRPlot_KPCapacity_alpha{infeasibility_aversion_coeff}_IAL_{ial_reduction}_PCGradNoFixedCost.png")

plt.close()

### Filtering parameter set
model_filters = {
    'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction, 'lr':0.005,
                                        'infeasibility_aversion_coeff':infeasibility_aversion_coeff,
                                        'degree':6, 'noise':0.25  },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
    'sfl':{"temp": 0.1, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
}

filtered_dfs = []
for model in df_selected['model'].unique():
    model_df = df_selected[df_selected['model'] == model]
    if model in model_filters:
        for col, val in model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    filtered_dfs.append(model_df)
filtered_custom = pd.concat(filtered_dfs, ignore_index=True)
filtered_custom['model']= filtered_custom['model'].where(( filtered_custom['model']!='odece' ),
                                           filtered_custom['model'].astype(str) + filtered_custom['infeasibility_aversion_coeff'].astype(str) )
# filtered_custom = filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
#                                                      "test_regret":['mean','std'],}).reset_index()

# # Flatten MultiIndex columns
# filtered_custom.columns = [
#     col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
#     for col in filtered_custom.columns.values
# ]
combined_order = filtered_custom.model.unique().tolist()
combined_order = filtered_custom.model.unique().tolist()
print (combined_order)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
# Validation Infeasibility vs Epoch
sns.lineplot(
    data=filtered_custom,
    x='epoch',
    y='val_infeasibility',
    hue='model',
    palette=colors_dict,
    ax=axes[0],
    marker='o'
)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Infeasibility')
axes[0].set_title('Validation Infeasibility vs Epoch')
axes[0].legend()
axes[0].grid(True)

# Validation Regret vs Epoch
sns.lineplot(
    data=filtered_custom,
    x='epoch',
    y='val_regret',
    hue='model',
    palette=colors_dict,
    ax=axes[1],
    marker='o'
)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Regret')
axes[1].set_title('Validation Regret vs Epoch')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"figures/LRPlot_KPCapacity_alpha{infeasibility_aversion_coeff}_IAL_{ial_reduction}_FixedAlphaNoFixedCost.png")

plt.close()

### Alloy

alloy = pd.read_csv( "../CombinedResults/ResultsAlloy.csv")

filtered = alloy[alloy['val_infeasibility'].notna()]
model_param_map = {
    'comboptnet': ['tau'],
    'sfl': ['temp'],
    'odece': ['ial_reduction', "use_pcgrad", "fix_alpha",'infeasibility_aversion_coeff'],
    # 'TwoStageIntOpt':['damping','thr','penalty'],
    'mse': []  # No extra parameter
}

# all_best = []
# ### Choosing best lr
# for model, params in model_param_map.items():
#     # Get only this model's rows
#     model_df = filtered[filtered['model'] == model]
#     # Group by params + lr
#     group_cols = ['model'] + params + ['lr']
#     # Find last epoch for each group
#     last_epochs = model_df.groupby(group_cols)['epoch'].max().reset_index()
#     # Merge to get the last valid epoch rows
#     last_epoch_rows = pd.merge(model_df, last_epochs, on=group_cols + ['epoch'])
#     # For each unique param set, pick lr with min val_infeasibility
#     if model == 'TwoStageIntOpt':
#         if params:
#             idx = last_epoch_rows.groupby(['model'] + params)['val_posthoc_regret'].idxmin()
#         else:
#             idx = last_epoch_rows.groupby(['model'])['val_posthoc_regret'].idxmin()
#     else:
#         if params:
#             idx = last_epoch_rows.groupby(['model'] + params)['val_infeasibility'].idxmin()
#         else:
#             idx = last_epoch_rows.groupby(['model'])['val_infeasibility'].idxmin()
#     best = last_epoch_rows.loc[idx]
#     all_best.append(best)

# best_lrs = pd.concat(all_best, ignore_index=True)

# df_selected = []
# for _, row in best_lrs.iterrows():
#     model = row['model']
#     lr = row['lr']
#     params = model_param_map[model]
#     # Filter for this model, lr, and parameter values
#     cond = (alloy['model'] == model) & (alloy['lr'] == lr)
#     for p in params:
#         cond &= (alloy[p] == row[p])
#     df_plot = alloy[cond]
#     df_selected.append (df_plot)
# df_selected = pd.concat ( df_selected, ignore_index=True)
models = model_param_map.keys()
df_selected = alloy [alloy['model'].isin(models)]
### Filtering parameter set
model_filters ={
    'odece': {'use_pcgrad': True, 'ial_reduction': ial_reduction, 'lr':0.002,
    'infeasibility_aversion_coeff':infeasibility_aversion_coeff },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.005},
    'sfl':{"temp": 0.5, 'lr':0.001},
    # 'TwoStageIntOpt': {'penalty': 0.5,'damping':0.01,'thr':0.1, 'lr':0.05}
}

filtered_dfs = []
for model in df_selected['model'].unique():
    model_df = df_selected[df_selected['model'] == model]
    if model in model_filters:
        for col, val in model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    filtered_dfs.append(model_df)
filtered_custom = pd.concat(filtered_dfs, ignore_index=True)
filtered_custom['model']= filtered_custom['model'].where(( filtered_custom['model']!='odece' ),
                                           filtered_custom['model'].astype(str) + filtered_custom['infeasibility_aversion_coeff'].astype(str) )
# filtered_custom = filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
#                                                      "test_regret":['mean','std'],}).reset_index()

# # Flatten MultiIndex columns
# filtered_custom.columns = [
#     col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
#     for col in filtered_custom.columns.values
# ]
combined_order = filtered_custom.model.unique().tolist()
combined_order = filtered_custom.model.unique().tolist()
print (combined_order)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
# Validation Infeasibility vs Epoch
sns.lineplot(
    data=filtered_custom,
    x='epoch',
    y='val_infeasibility',
    hue='model',
    palette=colors_dict,
    ax=axes[0],
    marker='o'
)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Infeasibility')
axes[0].set_title('Validation Infeasibility vs Epoch')
axes[0].legend()
axes[0].grid(True)

# Validation Regret vs Epoch
sns.lineplot(
    data=filtered_custom,
    x='epoch',
    y='val_regret',
    hue='model',
    palette=colors_dict,
    ax=axes[1],
    marker='o'
)
axes[1].set_ylim(0, 1)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Regret')
axes[1].set_title('Validation Regret vs Epoch')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"figures/LRPlot_Alloy_alpha{infeasibility_aversion_coeff}_IAL_{ial_reduction}_PCGrad.png")

plt.close()

### Filtering parameter set
model_filters = {
    'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction, 'lr':0.002,
    'infeasibility_aversion_coeff':infeasibility_aversion_coeff },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.005},
    'sfl':{"temp": 0.5, 'lr':0.001},
    'TwoStageIntOpt': {'penalty': 0.5,'damping':0.01,'thr':0.1, 'lr':0.05}
}

filtered_dfs = []
for model in df_selected['model'].unique():
    model_df = df_selected[df_selected['model'] == model]
    if model in model_filters:
        for col, val in model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    filtered_dfs.append(model_df)
filtered_custom = pd.concat(filtered_dfs, ignore_index=True)
filtered_custom['model']= filtered_custom['model'].where(( filtered_custom['model']!='odece' ),
                                           filtered_custom['model'].astype(str) + filtered_custom['infeasibility_aversion_coeff'].astype(str) )
# filtered_custom = filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
#                                                      "test_regret":['mean','std'],}).reset_index()

# # Flatten MultiIndex columns
# filtered_custom.columns = [
#     col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
#     for col in filtered_custom.columns.values
# ]
combined_order = filtered_custom.model.unique().tolist()
combined_order = filtered_custom.model.unique().tolist()
print (combined_order)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
# Validation Infeasibility vs Epoch
sns.lineplot(
    data=filtered_custom,
    x='epoch',
    y='val_infeasibility',
    hue='model',
    palette=colors_dict,
    ax=axes[0],
    marker='o'
)
axes[1].set_ylim(0, 1)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Infeasibility')
axes[0].set_title('Validation Infeasibility vs Epoch')
axes[0].legend()
axes[0].grid(True)

# Validation Regret vs Epoch
sns.lineplot(
    data=filtered_custom,
    x='epoch',
    y='val_regret',
    hue='model',
    palette=colors_dict,
    ax=axes[1],
    marker='o'
)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Regret')
axes[1].set_title('Validation Regret vs Epoch')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"figures/LRPlot_Alloy_alpha{infeasibility_aversion_coeff}_IAL_{ial_reduction}_FixedAlpha.png")

plt.close()
