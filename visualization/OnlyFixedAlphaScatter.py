import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec

colors_dict = {'mse':'tab:orange',  
'comboptnet': 'navy',
'sfl': 'c', 
'odece(0.2)':'m',
'odece(0.3)':'m',
'odece(0.4)':'m',
'odece(0.5)':'m',
'odece(0.6)':'m',
'odece(0.7)':'m',
'odece(0.8)':'m',
'TwoStageIntOpt' :'g'
        }

markers_dict = {'mse':'o',  
'comboptnet': 'P',
'sfl': 'X', 
'odece(0.2)':'v',
'odece(0.3)':'>',
'odece(0.4)':'1',
'odece(0.5)':'s',
'odece(0.6)':'2',
'odece(0.7)': '<',
'odece(0.8)':'^',
'TwoStageIntOpt' : 'D'
        }
ial_reduction = 'mean'#'mean'
### Next dataset when costs are not fixed
knapsack_weights = pd.read_csv( "../CombinedResults/KnapsackWeightsNoFixedCosts.csv")

filtered = knapsack_weights[knapsack_weights['val_infeasibility'].notna()]
model_param_map = {
    'comboptnet': ['tau'],
    'sfl': ['temp'],
    'odece': ['ial_reduction', "use_pcgrad", "fix_alpha",'infeasibility_aversion_coeff'],
    'TwoStageIntOpt':['damping','thr'],
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

models = model_param_map.keys()
weight_df_selected = knapsack_weights [knapsack_weights['model'].isin(models)]

### Filtering parameter set
weight_fixed_alpha_model_filters = {
    'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
    'sfl':{"temp": 0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
}
weight_fixed_alpha_filtered_dfs = []
for model in weight_df_selected['model'].unique():
    model_df = weight_df_selected[weight_df_selected['model'] == model]
    if model in weight_fixed_alpha_model_filters:
        for col, val in weight_fixed_alpha_model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    weight_fixed_alpha_filtered_dfs.append(model_df)
weight_fixed_alpha_filtered_custom = pd.concat(weight_fixed_alpha_filtered_dfs, ignore_index=True)

weight_fixed_alpha_filtered_custom['model']= weight_fixed_alpha_filtered_custom['model'].where(( weight_fixed_alpha_filtered_custom['model']!='odece' ),
                                           weight_fixed_alpha_filtered_custom['model'].astype(str) + "(" + weight_fixed_alpha_filtered_custom['infeasibility_aversion_coeff'].astype(str) + ")" )
weight_fixed_alpha_filtered_custom = weight_fixed_alpha_filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
                                                     "test_regret":['mean','std'],}).reset_index()

# Flatten MultiIndex columns
weight_fixed_alpha_filtered_custom.columns = [
    col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
    for col in weight_fixed_alpha_filtered_custom.columns.values
]

weight_combined_order = weight_fixed_alpha_filtered_custom.model.unique().tolist()


### Knapsack Capacity when costs are not fixed
knapsack_capa = pd.read_csv( "../CombinedResults/KnapsackCapacityNoFixedCosts.csv")

filtered = knapsack_capa[knapsack_capa['val_infeasibility'].notna()]
model_param_map = {
    'comboptnet': ['tau'],
    'sfl': ['temp'],
    'odece': ['ial_reduction', "use_pcgrad", "fix_alpha",'infeasibility_aversion_coeff'],
    'mse': []  # No extra parameter
}

# all_best = []

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
#     cond = (knapsack_capa['model'] == model) & (knapsack_capa['lr'] == lr)
#     for p in params:
#         cond &= (knapsack_capa[p] == row[p])
#     df_plot = knapsack_capa[cond]
#     df_selected.append (df_plot)
# df_selected = pd.concat ( df_selected, ignore_index=True)

models = model_param_map.keys()
capa_df_selected = knapsack_capa [knapsack_capa['model'].isin(models)]


capa_fixed_alpha_model_filters = {
    'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
    'sfl':{"temp": 0.1, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
    'mse': {'lr':0.05,
                                        'degree':6, 'noise':0.25  },
}
# capa_fixed_alpha_model_filters = {
#     'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction, 'lr':0.01,
#                                         'degree':6, 'noise':0.25  },
#     'comboptnet':{'loss':'l1','tau':0.1, 'lr':0.01,
#                                         'degree':6, 'noise':0.25  },
#     'sfl':{"temp": 0.1, 'lr':0.01,
#                                         'degree':6, 'noise':0.25  },
#     'mse': {'lr':0.01,
#                                         'degree':6, 'noise':0.25  },
# }
capa_fixed_alpha_filtered_dfs = []
for model in capa_df_selected['model'].unique():
    model_df = capa_df_selected[capa_df_selected['model'] == model]
    if model in capa_fixed_alpha_model_filters:
        for col, val in capa_fixed_alpha_model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    capa_fixed_alpha_filtered_dfs.append(model_df)
capa_fixed_alpha_filtered_custom = pd.concat(capa_fixed_alpha_filtered_dfs, ignore_index=True)

capa_fixed_alpha_filtered_custom['model']= capa_fixed_alpha_filtered_custom['model'].where(( capa_fixed_alpha_filtered_custom['model']!='odece' ),
                                           capa_fixed_alpha_filtered_custom['model'].astype(str) + "(" + capa_fixed_alpha_filtered_custom['infeasibility_aversion_coeff'].astype(str) + ")" )
capa_fixed_alpha_filtered_custom = capa_fixed_alpha_filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
                                                     "test_regret":['mean','std'],}).reset_index()

# Flatten MultiIndex columns
capa_fixed_alpha_filtered_custom.columns = [
    col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
    for col in capa_fixed_alpha_filtered_custom.columns.values
]

combined_order = capa_fixed_alpha_filtered_custom.model.unique().tolist()






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
alloy_df_selected = alloy [alloy['model'].isin(models)]

alloy_fixed_alpha_model_filters =  {
    'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction, 'lr':0.001},
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.005},
    'sfl':{"temp": 0.5, 'lr':0.001},
    'mse':{"lr": 0.001},
    'TwoStageIntOpt': {'penalty': 0.5,'damping':0.01,'thr':0.1, 'lr':0.05}
}

alloy_fixed_alpha_filtered_dfs = []
for model in alloy_df_selected['model'].unique():
    model_df = alloy_df_selected[alloy_df_selected['model'] == model]
    if model in alloy_fixed_alpha_model_filters:
        for col, val in alloy_fixed_alpha_model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    alloy_fixed_alpha_filtered_dfs.append(model_df)
alloy_fixed_alpha_filtered_custom = pd.concat(alloy_fixed_alpha_filtered_dfs, ignore_index=True)

alloy_fixed_alpha_filtered_custom['model']= alloy_fixed_alpha_filtered_custom['model'].where(( alloy_fixed_alpha_filtered_custom['model']!='odece' ),
                                           alloy_fixed_alpha_filtered_custom['model'].astype(str) + "(" + alloy_fixed_alpha_filtered_custom['infeasibility_aversion_coeff'].astype(str) + ")" )
alloy_fixed_alpha_filtered_custom = alloy_fixed_alpha_filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
                                                     "test_regret":['mean','std'],}).reset_index()

# Flatten MultiIndex columns
alloy_fixed_alpha_filtered_custom.columns = [
    col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
    for col in alloy_fixed_alpha_filtered_custom.columns.values
]
alloy_fixed_alpha_combined_order = alloy_fixed_alpha_filtered_custom.model.unique().tolist()


combined_order = alloy_fixed_alpha_filtered_custom.model.unique().tolist()

fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(2, 3, height_ratios=[10, 1])

# Scatter plots side by side
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])


# First subplot
for model in combined_order:
    plot_ = weight_fixed_alpha_filtered_custom[weight_fixed_alpha_filtered_custom.model==model]
    sns.scatterplot(
        data=plot_,
        x='test_infeasibility_mean',
        y='test_regret_mean',
        label=model,
        alpha=1,
        linewidth=2,
        s=200,
        color=colors_dict.get(model, 'C0'),
        marker=markers_dict.get(model, 'o'),
        ax=ax0
    )
ax0.tick_params(axis='both', which='major', labelsize=11)
ax0.set_xlabel('Test Infeasibility',fontsize=18)
ax0.set_ylabel('Test Regret',fontsize=18)
ax0.set_title('Predicting Weights in MDKP',fontsize=20)
ax0.grid(True)

for model in combined_order:
    plot_ = capa_fixed_alpha_filtered_custom[capa_fixed_alpha_filtered_custom.model==model]
    sns.scatterplot(
        data= plot_,
        x='test_infeasibility_mean',
        y='test_regret_mean',
        label= model,
        alpha=1,
        linewidth=2,
        s= 200,
        color=colors_dict.get(model, 'C0'),
        marker=markers_dict.get(model, 'o'),
        ax=ax1
    )
ax1.tick_params(axis='both', which='major', labelsize=11)
ax1.set_xlabel('Test Infeasibility',fontsize=18)
ax1.set_ylabel('Test Regret',fontsize=18)
ax1.set_title('Predicting Capacities in MDKP',fontsize=20)
ax1.legend()
ax1.grid(True)

for model in combined_order:
    plot_ = alloy_fixed_alpha_filtered_custom[alloy_fixed_alpha_filtered_custom.model==model]
    sns.scatterplot(
        data=plot_,
        x='test_infeasibility_mean',
        y='test_regret_mean',
        label=model,
        alpha=1,
        linewidth=2,
        s=200,
        color=colors_dict.get(model, 'C0'),
        marker=markers_dict.get(model, 'o'),
        ax=ax2
    )
ax2.tick_params(axis='both', which='major', labelsize=11)
ax2.set_xlabel('Test Infeasibility',fontsize=18)
ax2.set_ylabel('Test Regret',fontsize=18)
ax2.set_title('Predicting Quantities in Brass Alloy',fontsize=20)
ax2.grid(True)

ax0.set_xlim(0,1.1)
ax1.set_xlim(0, 1.1)
ax2.set_xlim(0, 1.1)


# Remove legends from both axes
if ax0.get_legend() is not None:
    ax0.get_legend().remove()
if ax1.get_legend() is not None:
    ax1.get_legend().remove()
if ax2.get_legend() is not None:
    ax2.get_legend().remove()
# Legend in the second row, spanning both columns
ax_legend = fig.add_subplot(gs[1, :])
ax_legend.axis('off')
handles, labels = [], []
for ax in [ax0, ax1, ax2]:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))
ax_legend.legend(list(by_label.values()), list(by_label.keys()), loc='center',
 ncol=5, fontsize=20)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(f"figures/FixedAlpha_IAL_{ial_reduction}_Scatterplot.png")
plt.close()