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


### Filtering parameter set
weight_fixed_alpha_model_filters = {
    'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
    'sfl':{"temp": 0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
}

models = model_param_map.keys()
weight_df_selected = knapsack_weights [knapsack_weights['model'].isin(models)]
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
                                           weight_fixed_alpha_filtered_custom['model'].astype(str) + weight_fixed_alpha_filtered_custom['infeasibility_aversion_coeff'].astype(str) )
# filtered_custom = filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],
#                                                      "test_regret":['mean','std'],}).reset_index()

# # Flatten MultiIndex columns
# filtered_custom.columns = [
#     col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
#     for col in filtered_custom.columns.values
# ]
combined_order = sorted(weight_fixed_alpha_filtered_custom.model.unique().tolist())
print(combined_order)
fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()  # Flatten to 1D for easy indexing

for idx, model in enumerate(combined_order):
    model_df = weight_fixed_alpha_filtered_custom[weight_fixed_alpha_filtered_custom['model'] == model]
    sns.lineplot(
        data=model_df,
        x='epoch',
        y='val_infeasibility',
        color=colors_dict.get(model, 'C0'),
        marker='o',
        ax=axes[idx]
    )
    axes[idx].set_title(model)
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Val Infeasibility')

# Hide any unused subplots if there are fewer than 10 models
for ax in axes[len(combined_order):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(f"figures/IndividualLR_KPWeightInfeasibility.png")
plt.close()

fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()  # Flatten to 1D for easy indexing

for idx, model in enumerate(combined_order):
    model_df = weight_fixed_alpha_filtered_custom[weight_fixed_alpha_filtered_custom['model'] == model]
    sns.lineplot(
        data=model_df,
        x='epoch',
        y='val_regret',
        color=colors_dict.get(model, 'C0'),
        marker='o',
        ax=axes[idx]
    )
    axes[idx].set_title(model)
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Val Regret')

# Hide any unused subplots if there are fewer than 10 models
for ax in axes[len(combined_order):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(f"figures/IndividualLR_KPWeightRegret.png")
plt.close()


knapsack_capa = pd.read_csv( "../CombinedResults/KnapsackCapacityNoFixedCosts.csv")

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
                                           capa_fixed_alpha_filtered_custom['model'].astype(str) + capa_fixed_alpha_filtered_custom['infeasibility_aversion_coeff'].astype(str) )

# filtered_custom = filtered_custom.groupby(['model'], dropna=False).agg ({"test_infeasibility":['mean', 'std'],

combined_order = capa_fixed_alpha_filtered_custom.model.unique().tolist()
combined_order = sorted(combined_order)
print(combined_order)
fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()  # Flatten to 1D for easy indexing

for idx, model in enumerate(combined_order):
    model_df = capa_fixed_alpha_filtered_custom[capa_fixed_alpha_filtered_custom['model'] == model]
    sns.lineplot(
        data=model_df,
        x='epoch',
        y='val_infeasibility',
        color=colors_dict.get(model, 'C0'),
        marker='o',
        ax=axes[idx]
    )
    axes[idx].set_title(model)
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Val Infeasibility')

# Hide any unused subplots if there are fewer than 10 models
for ax in axes[len(combined_order):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(f"figures/IndividualLR_KPCapaInfeasibility.png")
plt.close()

fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()  # Flatten to 1D for easy indexing

for idx, model in enumerate(combined_order):
    model_df = capa_fixed_alpha_filtered_custom[capa_fixed_alpha_filtered_custom['model'] == model]
    sns.lineplot(
        data=model_df,
        x='epoch',
        y='val_regret',
        color=colors_dict.get(model, 'C0'),
        marker='o',
        ax=axes[idx]
    )
    axes[idx].set_title(model)
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Val Regret')
    # axes[idx].set_ylim(0, 1)

# Hide any unused subplots if there are fewer than 10 models
for ax in axes[len(combined_order):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(f"figures/IndividualLR_KPCapaRegret.png")
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

models = model_param_map.keys()
alloy_df_selected = alloy [alloy['model'].isin(models)]

alloy_fixed_alpha_model_filters =  {
    'odece': {'fix_alpha': True, 'ial_reduction': ial_reduction, 'lr':0.001},
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.005},
    'sfl':{"temp": 0.5, 'lr':0.001},
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
combined_order = alloy_fixed_alpha_filtered_custom.model.unique().tolist()
combined_order = sorted(combined_order)

fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()  # Flatten to 1D for easy indexing

for idx, model in enumerate(combined_order):
    model_df = alloy_fixed_alpha_filtered_custom[alloy_fixed_alpha_filtered_custom['model'] == model]
    sns.lineplot(
        data=model_df,
        x='epoch',
        y='val_infeasibility',
        color=colors_dict.get(model, 'C0'),
        marker='o',
        ax=axes[idx]
    )
    axes[idx].set_title(model)
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Val Infeasibility')

# Hide any unused subplots if there are fewer than 10 models
for ax in axes[len(combined_order):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(f"figures/IndividualLR_AlloyInfeasibility.png")
plt.close()

fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()  # Flatten to 1D for easy indexing

for idx, model in enumerate(combined_order):
    model_df = alloy_fixed_alpha_filtered_custom[alloy_fixed_alpha_filtered_custom['model'] == model]
    sns.lineplot(
        data=model_df,
        x='epoch',
        y='val_regret',
        color=colors_dict.get(model, 'C0'),
        marker='o',
        ax=axes[idx]
    )
    axes[idx].set_title(model)
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Val Regret')
    axes[idx].set_ylim(0, 2)

# Hide any unused subplots if there are fewer than 10 models
for ax in axes[len(combined_order):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(f"figures/IndividualLR_AlloyRegret.png")
plt.close()
