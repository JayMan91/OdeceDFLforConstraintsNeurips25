import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
### First Knapsack Weights

# knapsack_weights = pd.read_csv( "../CombinedResults/KnapsackWeightsFixedCosts.csv")

# filtered = knapsack_weights[knapsack_weights['val_infeasibility'].notna()]
# model_param_map = {
#     'comboptnet': ['tau'],
#     'sfl': ['temp'],
#     'odece': ['ial_reduction', "use_pcgrad", "fix_alpha",'infeasibility_aversion_coeff'],
#     'mse': []  # No extra parameter
# }

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


# ### Filtering parameter set
# model_filters = {
#     'odece': {'use_pcgrad': True, 'ial_reduction': ial_reduction},
#     'comboptnet':{'loss':'l1','tau':0.5},
#     'sfl':{"temp": 0.1},
# }

# filtered_dfs = []
# for model in df_selected['model'].unique():
#     model_df = df_selected[df_selected['model'] == model]
#     if model in model_filters:
#         for col, val in model_filters[model].items():
#             model_df = model_df[model_df[col] == val]
#     # If model not in model_filters, keep all rows for that model
#     filtered_dfs.append(model_df)
# filtered_custom = pd.concat(filtered_dfs, ignore_index=True)
# filtered_custom['model']= filtered_custom['model'].where(( filtered_custom['model']!='odece' ),
#                                            filtered_custom['model'].astype(str) + filtered_custom['infeasibility_aversion_coeff'].astype(str) )
# filtered_custom = filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
#                                                      "test_regret":['mean','std'],}).reset_index()

# # Flatten MultiIndex columns
# filtered_custom.columns = [
#     col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
#     for col in filtered_custom.columns.values
# ]
# combined_order = filtered_custom.model.unique().tolist()
# print (combined_order)
# plt.figure(figsize=(8, 6))
# for model in combined_order:
#     plot_ = filtered_custom[filtered_custom.model==model]
#     sns.scatterplot(
#         data= plot_,
#         x='test_infeasibility_mean',
#         y='test_regret_mean',
#         label= model,
#         alpha=0.8,
#         s= 200,
#         color=colors_dict.get(model, 'C0'),
#         marker=markers_dict.get(model, 'o'),
#     )
# plt.xlabel('Validation Infeasibility')
# plt.ylabel('Validation Regret')
# plt.title('Scatter Plot: Infeasibility vs Regret')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"figures/KPWeight_IAL_{ial_reduction}_PCGradFixedCostScatterplot.png")
# plt.close()

# model_filters = {
#     'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction},
#     'comboptnet':{'loss':'l1','tau':0.5},
#     'sfl':{"temp": 0.1}
# }

# filtered_dfs = []
# for model in df_selected['model'].unique():
#     model_df = df_selected[df_selected['model'] == model]
#     if model in model_filters:
#         for col, val in model_filters[model].items():
#             model_df = model_df[model_df[col] == val]
#     # If model not in model_filters, keep all rows for that model
#     filtered_dfs.append(model_df)
# filtered_custom = pd.concat(filtered_dfs, ignore_index=True)

# filtered_custom['model']= filtered_custom['model'].where(( filtered_custom['model']!='odece' ),
#                                            filtered_custom['model'].astype(str) + filtered_custom['infeasibility_aversion_coeff'].astype(str) )
# filtered_custom = filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
#                                                      "test_regret":['mean','std'],}).reset_index()

# # Flatten MultiIndex columns
# filtered_custom.columns = [
#     col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
#     for col in filtered_custom.columns.values
# ]
# combined_order = filtered_custom.model.unique().tolist()
# plt.figure(figsize=(8, 6))
# for model in combined_order:
#     plot_ = filtered_custom[filtered_custom.model==model]
#     sns.scatterplot(
#         data= plot_,
#         x='test_infeasibility_mean',
#         y='test_regret_mean',
#         label= model,
#         alpha=0.8,
#         s= 200,
#         color=colors_dict.get(model, 'C0'),
#         marker=markers_dict.get(model, 'o'),
#     )
# plt.xlabel('Validation Infeasibility')
# plt.ylabel('Validation Regret')
# plt.title('Scatter Plot: Infeasibility vs Regret')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"figures/KPWeight_IAL_{ial_reduction}_FixedAlphaFixedCostScatterplot.png")
# plt.close()

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
df_selected = knapsack_weights [knapsack_weights['model'].isin(models)]
### Filtering parameter set
PCGRAD_model_filters = {
    'odece': {'use_pcgrad': True, 'ial_reduction': ial_reduction,  'lr':0.01,
                                        'degree':6, 'noise':0.25  },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
    'sfl':{"temp": 0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
}

PCGRAD_filtered_dfs = []
for model in df_selected['model'].unique():
    model_df = df_selected[df_selected['model'] == model]
    if model in PCGRAD_model_filters:
        for col, val in PCGRAD_model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    PCGRAD_filtered_dfs.append(model_df)
PCGRAD_filtered_custom = pd.concat(PCGRAD_filtered_dfs, ignore_index=True)
PCGRAD_filtered_custom['model']= PCGRAD_filtered_custom['model'].where(( PCGRAD_filtered_custom['model']!='odece' ),
                                           PCGRAD_filtered_custom['model'].astype(str) + "(" + PCGRAD_filtered_custom['infeasibility_aversion_coeff'].astype(str) + ")" )
PCGRAD_filtered_custom = PCGRAD_filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
                                                     "test_regret":['mean','std'],}).reset_index()

# Flatten MultiIndex columns
PCGRAD_filtered_custom.columns = [
    col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
    for col in PCGRAD_filtered_custom.columns.values
]

### Filtering parameter set
fixed_alpha_model_filters = {
    'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
    'sfl':{"temp": 0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
}
fixed_alpha_filtered_dfs = []
for model in df_selected['model'].unique():
    model_df = df_selected[df_selected['model'] == model]
    if model in fixed_alpha_model_filters:
        for col, val in fixed_alpha_model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    fixed_alpha_filtered_dfs.append(model_df)
fixed_alpha_filtered_custom = pd.concat(fixed_alpha_filtered_dfs, ignore_index=True)

fixed_alpha_filtered_custom['model']= fixed_alpha_filtered_custom['model'].where(( fixed_alpha_filtered_custom['model']!='odece' ),
                                           fixed_alpha_filtered_custom['model'].astype(str) + "(" + fixed_alpha_filtered_custom['infeasibility_aversion_coeff'].astype(str) + ")" )
fixed_alpha_filtered_custom = fixed_alpha_filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
                                                     "test_regret":['mean','std'],}).reset_index()

# Flatten MultiIndex columns
fixed_alpha_filtered_custom.columns = [
    col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
    for col in fixed_alpha_filtered_custom.columns.values
]

combined_order = PCGRAD_filtered_custom.model.unique().tolist()
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 2, height_ratios=[10, 1])

# Scatter plots side by side
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])

# First subplot
for model in combined_order:
    plot_ = fixed_alpha_filtered_custom[fixed_alpha_filtered_custom.model==model]
    sns.scatterplot(
        data=plot_,
        x='test_infeasibility_mean',
        y='test_regret_mean',
        label=model,
        alpha=0.8,
        s=200,
        color=colors_dict.get(model, 'C0'),
        marker=markers_dict.get(model, 'o'),
        ax=ax0
    )
ax0.tick_params(axis='both', which='major', labelsize=9)
ax0.set_xlabel('Test Infeasibility',fontsize=12)
ax0.set_ylabel('Test Regret',fontsize=12)
ax0.set_title('Weighted Averaging',fontsize=15)
ax0.grid(True)

# Second subplot (replace with your second plot data if different)
for model in combined_order:
    plot_ = PCGRAD_filtered_custom[PCGRAD_filtered_custom.model==model]
    sns.scatterplot(
        data=plot_,
        x='test_infeasibility_mean',
        y='test_regret_mean',
        label=model,
        alpha=0.8,
        s=200,
        color=colors_dict.get(model, 'C0'),
        marker=markers_dict.get(model, 'o'),
        ax=ax1
    )
ax1.tick_params(axis='both', which='major', labelsize=9)
ax1.set_xlabel('Test Infeasibility',fontsize=12)
ax1.set_ylabel('Test Regret',fontsize=12)
ax1.set_title('Gradient Projection',fontsize=15)
ax1.grid(True)

# Second subplot (replace with your second plot data if different)


# Remove legends from both axes
if ax0.get_legend() is not None:
    ax0.get_legend().remove()
if ax1.get_legend() is not None:
    ax1.get_legend().remove()

# Legend in the second row, spanning both columns
ax_legend = fig.add_subplot(gs[1, :])
ax_legend.axis('off')
handles, labels = [], []
for ax in [ax0, ax1]:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))
ax_legend.legend(list(by_label.values()), list(by_label.keys()), loc='center',
 ncol=5, fontsize=15)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(f"figures/KPWeight_IAL_{ial_reduction}_NoFixedCostScatterplot.png",bbox_inches='tight')
plt.close()
# combined_order = filtered_custom.model.unique().tolist()
# plt.figure(figsize=(8, 6))
# for model in combined_order:
#     plot_ = filtered_custom[filtered_custom.model==model]
#     sns.scatterplot(
#         data= plot_,
#         x='test_infeasibility_mean',
#         y='test_regret_mean',
#         label= model,
#         alpha=0.8,
#         s= 200,
#         color=colors_dict.get(model, 'C0'),
#         marker=markers_dict.get(model, 'o'),
#     )
# plt.xlabel('Test Infeasibility')
# plt.ylabel('Test Regret')
# plt.title('Scatter Plot: Test Infeasibility vs Test Regret')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"figures/KPWeight_IAL_{ial_reduction}_FixedAlphaNoFixedCostScatterplot.png")
# plt.close()

#### Knapsack Capcity

# knapsack_capa = pd.read_csv( "../CombinedResults/KnapsackCapacityFixedCosts.csv")

# filtered = knapsack_capa[knapsack_capa['val_infeasibility'].notna()]
# model_param_map = {
#     'comboptnet': ['tau'],
#     'sfl': ['temp'],
#     'odece': ['ial_reduction', "use_pcgrad", "fix_alpha",'infeasibility_aversion_coeff'],
#     'mse': []  # No extra parameter
# }

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

# model_filters = {
#     'odece': {'use_pcgrad': True, 'ial_reduction': ial_reduction},
#     'comboptnet':{'loss':'l1','tau':0.2},
#     'sfl':{"temp": 0.2}
# }

# filtered_dfs = []
# for model in df_selected['model'].unique():
#     model_df = df_selected[df_selected['model'] == model]
#     if model in model_filters:
#         for col, val in model_filters[model].items():
#             model_df = model_df[model_df[col] == val]
#     # If model not in model_filters, keep all rows for that model
#     filtered_dfs.append(model_df)
# filtered_custom = pd.concat(filtered_dfs, ignore_index=True)

# filtered_custom['model']= filtered_custom['model'].where(( filtered_custom['model']!='odece' ),
#                                            filtered_custom['model'].astype(str) + filtered_custom['infeasibility_aversion_coeff'].astype(str) )
# filtered_custom = filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
#                                                      "test_regret":['mean','std'],}).reset_index()

# # Flatten MultiIndex columns
# filtered_custom.columns = [
#     col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
#     for col in filtered_custom.columns.values
# ]
# combined_order = filtered_custom.model.unique().tolist()

# plt.figure(figsize=(8, 6))
# for model in combined_order:
#     plot_ = filtered_custom[filtered_custom.model==model]
#     sns.scatterplot(
#         data= plot_,
#         x='test_infeasibility_mean',
#         y='test_regret_mean',
#         label= model,
#         alpha=0.8,
#         s= 200,
#         color=colors_dict.get(model, 'C0'),
#         marker=markers_dict.get(model, 'o'),
#     )
# plt.xlabel('Validation Infeasibility')
# plt.ylabel('Validation Regret')
# plt.title('Scatter Plot: Infeasibility vs Regret')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"figures/KPCapacity_IAL_{ial_reduction}_PCGradFixedCostScatterplot.png")
# plt.close()


# model_filters = {
#     'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction},
#     'comboptnet':{'loss':'l1','tau':0.1},
#     'sfl':{"temp": 0.1}
# }

# filtered_dfs = []
# for model in df_selected['model'].unique():
#     model_df = df_selected[df_selected['model'] == model]
#     if model in model_filters:
#         for col, val in model_filters[model].items():
#             model_df = model_df[model_df[col] == val]
#     # If model not in model_filters, keep all rows for that model
#     filtered_dfs.append(model_df)
# filtered_custom = pd.concat(filtered_dfs, ignore_index=True)

# filtered_custom['model']= filtered_custom['model'].where(( filtered_custom['model']!='odece' ),
#                                            filtered_custom['model'].astype(str) + filtered_custom['infeasibility_aversion_coeff'].astype(str) )
# filtered_custom = filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
#                                                      "test_regret":['mean','std'],}).reset_index()

# # Flatten MultiIndex columns
# filtered_custom.columns = [
#     col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
#     for col in filtered_custom.columns.values
# ]
# combined_order = filtered_custom.model.unique().tolist()

# plt.figure(figsize=(8, 6))
# for model in combined_order:
#     plot_ = filtered_custom[filtered_custom.model==model]
#     sns.scatterplot(
#         data= plot_,
#         x='test_infeasibility_mean',
#         y='test_regret_mean',
#         label= model,
#         alpha=0.8,
#         s= 200,
#         color=colors_dict.get(model, 'C0'),
#         marker=markers_dict.get(model, 'o'),
#     )
# plt.xlabel('Validation Infeasibility')
# plt.ylabel('Validation Regret')
# plt.title('Scatter Plot: Infeasibility vs Regret')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"figures/KPCapacity_IAL_{ial_reduction}_FixedAlphaFixedCostScatterplot.png")
# plt.close()

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
df_selected = knapsack_capa [knapsack_capa['model'].isin(models)]
### Filtering parameter set
PCGRAD_model_filters = {
    'odece': {'use_pcgrad': True, 'ial_reduction': ial_reduction, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
    'sfl':{"temp": 0.1, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
}
PCGRAD_filtered_dfs = []
for model in df_selected['model'].unique():
    model_df = df_selected[df_selected['model'] == model]
    if model in PCGRAD_model_filters:
        for col, val in PCGRAD_model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    PCGRAD_filtered_dfs.append(model_df)
PCGRAD_filtered_custom = pd.concat(PCGRAD_filtered_dfs, ignore_index=True)

PCGRAD_filtered_custom['model']= PCGRAD_filtered_custom['model'].where(( PCGRAD_filtered_custom['model']!='odece' ),
                                           PCGRAD_filtered_custom['model'].astype(str) + "(" + PCGRAD_filtered_custom['infeasibility_aversion_coeff'].astype(str) + ")" )
PCGRAD_filtered_custom = PCGRAD_filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
                                                     "test_regret":['mean','std'],}).reset_index()

# Flatten MultiIndex columns
PCGRAD_filtered_custom.columns = [
    col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
    for col in PCGRAD_filtered_custom.columns.values
]

fixed_alpha_model_filters = {
    'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
    'sfl':{"temp": 0.1, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
}

fixed_alpha_filtered_dfs = []
for model in df_selected['model'].unique():
    model_df = df_selected[df_selected['model'] == model]
    if model in fixed_alpha_model_filters:
        for col, val in fixed_alpha_model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    fixed_alpha_filtered_dfs.append(model_df)
fixed_alpha_filtered_custom = pd.concat(fixed_alpha_filtered_dfs, ignore_index=True)

fixed_alpha_filtered_custom['model']= fixed_alpha_filtered_custom['model'].where(( fixed_alpha_filtered_custom['model']!='odece' ),
                                           fixed_alpha_filtered_custom['model'].astype(str) + "(" + fixed_alpha_filtered_custom['infeasibility_aversion_coeff'].astype(str) + ")" )
fixed_alpha_filtered_custom = fixed_alpha_filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
                                                     "test_regret":['mean','std'],}).reset_index()

# Flatten MultiIndex columns
fixed_alpha_filtered_custom.columns = [
    col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
    for col in fixed_alpha_filtered_custom.columns.values
]

combined_order = PCGRAD_filtered_custom.model.unique().tolist()

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 2, height_ratios=[10, 1])

# Scatter plots side by side
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])

for model in combined_order:
    plot_ = fixed_alpha_filtered_custom[fixed_alpha_filtered_custom.model==model]
    sns.scatterplot(
        data= plot_,
        x='test_infeasibility_mean',
        y='test_regret_mean',
        label= model,
        alpha=0.8,
        s= 200,
        color=colors_dict.get(model, 'C0'),
        marker=markers_dict.get(model, 'o'),
        ax=ax0
    )
ax0.tick_params(axis='both', which='major', labelsize=9)
ax0.set_xlabel('Test Infeasibility',fontsize=12)
ax0.set_ylabel('Test Regret',fontsize=12)
ax0.set_title('Weighted Averaging',fontsize=15)
ax0.legend()
ax0.grid(True)

for model in combined_order:
    plot_ = PCGRAD_filtered_custom[PCGRAD_filtered_custom.model==model]
    sns.scatterplot(
        data= plot_,
        x='test_infeasibility_mean',
        y='test_regret_mean',
        label= model,
        alpha=0.8,
        s= 200,
        color=colors_dict.get(model, 'C0'),
        marker=markers_dict.get(model, 'o'),
    )
ax1.tick_params(axis='both', which='major', labelsize=9)
ax1.set_xlabel('Test Infeasibility',fontsize=12)
ax1.set_ylabel('Test Regret',fontsize=12)
ax1.set_title('Gradient Projection',fontsize=15)
ax1.legend()
ax1.grid(True)

# Remove legends from both axes
if ax0.get_legend() is not None:
    ax0.get_legend().remove()
if ax1.get_legend() is not None:
    ax1.get_legend().remove()
# Legend in the second row, spanning both columns
ax_legend = fig.add_subplot(gs[1, :])
ax_legend.axis('off')
handles, labels = [], []
for ax in [ax0, ax1]:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))
ax_legend.legend(list(by_label.values()), list(by_label.keys()), loc='center',
 ncol=5, fontsize=15)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(f"figures/KPCapacity_IAL_{ial_reduction}_NoFixedCostScatterplot.png",bbox_inches='tight')
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

PCGRAD_model_filters = {
    'odece': {'use_pcgrad': True, 'ial_reduction': ial_reduction, 'lr':0.001},
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.005},
    'sfl':{"temp": 0.5, 'lr':0.001},
    'TwoStageIntOpt': {'penalty': 0.5,'damping':0.01,'thr':0.1, 'lr':0.05}
}

PCGRAD_filtered_dfs = []
for model in df_selected['model'].unique():
    model_df = df_selected[df_selected['model'] == model]
    if model in PCGRAD_model_filters:
        for col, val in PCGRAD_model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    PCGRAD_filtered_dfs.append(model_df)
PCGRAD_filtered_custom = pd.concat(PCGRAD_filtered_dfs, ignore_index=True)

PCGRAD_filtered_custom['model']= PCGRAD_filtered_custom['model'].where(( PCGRAD_filtered_custom['model']!='odece' ),
                                           PCGRAD_filtered_custom['model'].astype(str) + "(" + PCGRAD_filtered_custom['infeasibility_aversion_coeff'].astype(str) + ")" )
PCGRAD_filtered_custom = PCGRAD_filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
                                                     "test_regret":['mean','std'],}).reset_index()

# Flatten MultiIndex columns
PCGRAD_filtered_custom.columns = [
    col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
    for col in PCGRAD_filtered_custom.columns.values
]
fixed_alpha_model_filters =  {
    'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction, 'lr':0.001},
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.005},
    'sfl':{"temp": 0.5, 'lr':0.001},
    'TwoStageIntOpt': {'penalty': 0.5,'damping':0.01,'thr':0.1, 'lr':0.05}
}

fixed_alpha_filtered_dfs = []
for model in df_selected['model'].unique():
    model_df = df_selected[df_selected['model'] == model]
    if model in fixed_alpha_model_filters:
        for col, val in fixed_alpha_model_filters[model].items():
            model_df = model_df[model_df[col] == val]
    # If model not in model_filters, keep all rows for that model
    fixed_alpha_filtered_dfs.append(model_df)
fixed_alpha_filtered_custom = pd.concat(fixed_alpha_filtered_dfs, ignore_index=True)

fixed_alpha_filtered_custom['model']= fixed_alpha_filtered_custom['model'].where(( fixed_alpha_filtered_custom['model']!='odece' ),
                                           fixed_alpha_filtered_custom['model'].astype(str) + "(" + fixed_alpha_filtered_custom['infeasibility_aversion_coeff'].astype(str) + ")" )
fixed_alpha_filtered_custom = fixed_alpha_filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
                                                     "test_regret":['mean','std'],}).reset_index()

# Flatten MultiIndex columns
fixed_alpha_filtered_custom.columns = [
    col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
    for col in fixed_alpha_filtered_custom.columns.values
]
fixed_alpha_combined_order = fixed_alpha_filtered_custom.model.unique().tolist()


combined_order = PCGRAD_filtered_custom.model.unique().tolist()

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 2, height_ratios=[10, 1])

# Scatter plots side by side
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])


for model in combined_order:
    plot_ = fixed_alpha_filtered_custom[fixed_alpha_filtered_custom.model==model]
    sns.scatterplot(
        data=plot_,
        x='test_infeasibility_mean',
        y='test_regret_mean',
        label=model,
        alpha=0.8,
        s=200,
        color=colors_dict.get(model, 'C0'),
        marker=markers_dict.get(model, 'o'),
        ax=ax0
    )
ax0.tick_params(axis='both', which='major', labelsize=9)
ax0.set_xlabel('Test Infeasibility',fontsize=12)
ax0.set_ylabel('Test Regret',fontsize=12)
ax0.set_title('Weighted Averaging',fontsize=15)
ax0.grid(True)

for model in combined_order:
    plot_ = PCGRAD_filtered_custom[PCGRAD_filtered_custom.model==model]
    sns.scatterplot(
        data=plot_,
        x='test_infeasibility_mean',
        y='test_regret_mean',
        label=model,
        alpha=0.8,
        s=200,
        color=colors_dict.get(model, 'C0'),
        marker=markers_dict.get(model, 'o'),
        ax=ax1
    )
ax1.tick_params(axis='both', which='major', labelsize=9)
ax1.set_xlabel('Test Infeasibility',fontsize=12)
ax1.set_ylabel('Test Regret',fontsize=12)
ax1.set_title('Gradient Projection',fontsize=15)
ax1.grid(True)

# Remove legends from both axes
if ax0.get_legend() is not None:
    ax0.get_legend().remove()
if ax1.get_legend() is not None:
    ax1.get_legend().remove()
# Legend in the second row, spanning both columns
ax_legend = fig.add_subplot(gs[1, :])
ax_legend.axis('off')
handles, labels = [], []
for ax in [ax0, ax1]:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))
ax_legend.legend(list(by_label.values()), list(by_label.keys()), loc='center',
 ncol=5, fontsize=15)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(f"figures/Alloy_IAL_{ial_reduction}_Scatterplot.png")
plt.close()