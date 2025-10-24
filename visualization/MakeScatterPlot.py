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
'2sPtO' :'g'
        }

markers_dict = {'mse':'D',  
'comboptnet': 'x',
'sfl': 's', 
'odece(0.2)':'v',
'odece(0.3)':'>',
'odece(0.4)':'1',
'odece(0.5)':'o',
'odece(0.6)':'2',
'odece(0.7)': '<',
'odece(0.8)':'^',
'2sPtO' : '+',
        }

### Next dataset when costs are not fixed
knapsack_weights = pd.read_csv( "../CombinedResults/KnapsackWeightsNoFixedCosts.csv")
knapsack_weights['model'] = knapsack_weights['model'].replace('TwoStagePtO', '2sPtO')
filtered = knapsack_weights[knapsack_weights['val_infeasibility'].notna()]
model_param_map = {
    'comboptnet': ['tau'],
    'sfl': ['temp'],
    'odece': ['infeasibility_aversion_coeff'],
    '2sPtO':['damping','thr','knapsack_penalty'],
    'mse': []  # No extra parameter
}

models = model_param_map.keys()
weight_df_selected = knapsack_weights [knapsack_weights['model'].isin(models)]

### Filtering parameter set
weight_fixed_alpha_model_filters = {
    'odece': {'lr':0.05,
                                        'degree':6, 'noise':0.25  },
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
    'sfl':{"temp": 0.5, 'lr':0.05,
                                        'degree':6, 'noise':0.25  },
     '2sPtO': {'knapsack_penalty': 0.21,'damping':0.01,'thr':0.1, 'lr':0.05}
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

# During evaluation, unsolvable instances to be considered as infeasible
weight_fixed_alpha_filtered_custom['total_infeasibility'] = np.fmax(
    weight_fixed_alpha_filtered_custom ['test_unsolvable'], 
    weight_fixed_alpha_filtered_custom ['test_infeasibility']
)

weight_fixed_alpha_filtered_custom['model']= weight_fixed_alpha_filtered_custom['model'].where(( weight_fixed_alpha_filtered_custom['model']!='odece' ),
                                           weight_fixed_alpha_filtered_custom['model'].astype(str) + "(" + weight_fixed_alpha_filtered_custom['infeasibility_aversion_coeff'].astype(str) + ")" )
                   
weight_fixed_alpha_filtered_custom = weight_fixed_alpha_filtered_custom.groupby(['model'], dropna=False).agg ({"total_infeasibility":['mean', 'std'],
                                                     "test_regret":['mean','std'],}).reset_index()

# Flatten MultiIndex columns
weight_fixed_alpha_filtered_custom.columns = [
    col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
    for col in weight_fixed_alpha_filtered_custom.columns.values
]

weight_combined_order = weight_fixed_alpha_filtered_custom.model.unique().tolist()


### Knapsack Capacity when costs are not fixed
knapsack_capa = pd.read_csv( "../CombinedResults/KnapsackCapacityNoFixedCosts.csv")
knapsack_capa['model'] = knapsack_capa['model'].replace('TwoStagePtO', '2sPtO')
filtered = knapsack_capa[knapsack_capa['val_infeasibility'].notna()]
model_param_map = {
    'comboptnet': ['tau'],
    'sfl': ['temp'],
    'odece': ['infeasibility_aversion_coeff'],
    'mse': []  # No extra parameter
}


models = model_param_map.keys()
capa_df_selected = knapsack_capa [knapsack_capa['model'].isin(models)]


capa_fixed_alpha_model_filters = {
    'odece': { 'lr':0.005,
            'degree':6, 'noise':0.25  },
    'comboptnet':{'loss':'l1','tau':0.1, 'lr':0.005,
                                        'degree':6, 'noise':0.25  },
    'sfl':{"temp": 0.1, 'lr':0.01,
                                        'degree':6, 'noise':0.25  },
    'mse': {'lr':0.005,
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
capa_fixed_alpha_filtered_custom['total_infeasibility'] = np.fmax(
    capa_fixed_alpha_filtered_custom ['test_unsolvable'], 
    capa_fixed_alpha_filtered_custom ['test_infeasibility']
)
capa_fixed_alpha_filtered_custom['model']= capa_fixed_alpha_filtered_custom['model'].where(( capa_fixed_alpha_filtered_custom['model']!='odece' ),
                                           capa_fixed_alpha_filtered_custom['model'].astype(str) + "(" + capa_fixed_alpha_filtered_custom['infeasibility_aversion_coeff'].astype(str) + ")" )
capa_fixed_alpha_filtered_custom = capa_fixed_alpha_filtered_custom.groupby(['model'], dropna=False).agg ({"total_infeasibility":['mean', 'std'],
                                                     "test_regret":['mean','std'],}).reset_index()

# Flatten MultiIndex columns
capa_fixed_alpha_filtered_custom.columns = [
    col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
    for col in capa_fixed_alpha_filtered_custom.columns.values
]

combined_order = capa_fixed_alpha_filtered_custom.model.unique().tolist()

### Alloy

alloy = pd.read_csv( "../CombinedResults/ResultsAlloy.csv")
alloy['model'] = alloy['model'].replace('TwoStagePtO', '2sPtO')
filtered = alloy[alloy['val_infeasibility'].notna()]
model_param_map = {
    'comboptnet': ['tau'],
    'sfl': ['temp'],
    'odece': ['infeasibility_aversion_coeff'],
    '2sPtO':['damping','thr','penalty'],
    'mse': []  # No extra parameter
}


models = model_param_map.keys()
alloy_df_selected = alloy [alloy['model'].isin(models)]

alloy_fixed_alpha_model_filters =  {
    'odece': { 'lr':0.001},
    'comboptnet':{'loss':'l1','tau':0.5, 'lr':0.005},
    'sfl':{"temp": 0.5, 'lr':0.001},
    'mse':{"lr": 0.001},
    '2sPtO': {'penalty': 8,'damping':0.01,'thr':0.001, 'lr':0.05}
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
alloy_fixed_alpha_filtered_custom['total_infeasibility'] = np.fmax(
    alloy_fixed_alpha_filtered_custom ['test_unsolvable'], 
    alloy_fixed_alpha_filtered_custom ['test_infeasibility']
)


alloy_fixed_alpha_filtered_custom['model']= alloy_fixed_alpha_filtered_custom['model'].where(( alloy_fixed_alpha_filtered_custom['model']!='odece' ),
                                           alloy_fixed_alpha_filtered_custom['model'].astype(str) + "(" + alloy_fixed_alpha_filtered_custom['infeasibility_aversion_coeff'].astype(str) + ")" )
alloy_fixed_alpha_filtered_custom = alloy_fixed_alpha_filtered_custom.groupby(['model'], dropna=False).agg ({"total_infeasibility":['mean', 'std'],
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
        x='total_infeasibility_mean',
        y='test_regret_mean',
        label=model,
        alpha=1,  # <— transparency helps reveal overlaps
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
        x='total_infeasibility_mean',
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
        x='total_infeasibility_mean',
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
plt.savefig(f"figures/FixedAlpha_Scatterplot.png")
plt.close()


### Same Figure With Broken Axis

fig = plt.figure(figsize=(18, 6))

gs = gridspec.GridSpec(2, 3, height_ratios=[10,  1]) # Added space for broken axis marks

# Scatter plots side by side
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
# First subplot
for model in combined_order:
    plot_ = weight_fixed_alpha_filtered_custom[weight_fixed_alpha_filtered_custom.model==model]
    sns.scatterplot(
        data=plot_,
        x='total_infeasibility_mean',
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
        x='total_infeasibility_mean',
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

# --- Subplot 3: Predicting Quantities in Brass Alloy (with Broken Axis) ---
# For just the broken axis part:
# For ax2 (Alloy), create a broken axis with two y-ranges
# We'll use two subplots stacked vertically to simulate broken axis


gs_broken = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs[0, 2], height_ratios=[1, 3], hspace=0.15
)
ax2_top = fig.add_subplot(gs_broken[0])
ax2_bottom = fig.add_subplot(gs_broken[1], sharex=ax2_top)

ax2_top.tick_params(axis='x', labelbottom=False)
ax2_bottom.spines['top'].set_visible(False)
ax2_top.spines['bottom'].set_visible(False)
ax2_top.grid(True, which='major', axis='both')
ax2_bottom.grid(True)

for model in combined_order:
    plot_ = alloy_fixed_alpha_filtered_custom[alloy_fixed_alpha_filtered_custom.model==model]
    # Plot on both axes
    for ax_part in [ax2_top, ax2_bottom]:
        sns.scatterplot(
            data=plot_,
            x='total_infeasibility_mean',
            y='test_regret_mean',
            label=model,
            alpha=1,
            linewidth=2,
            s=200,
            color=colors_dict.get(model, 'C0'),
            marker=markers_dict.get(model, 'o'),
            ax=ax_part
        )
ax2_top.set_ylim(5, 10)
ax2_bottom.set_ylim(-0.01, 0.6)
ax2_bottom.set_xlabel('Test Infeasibility',fontsize=18)
ax2_bottom.set_ylabel('Test Regret',fontsize=18)
ax2_top.set_ylabel('',fontsize=18)



# Diagonal lines to indicate break
d = 0.01
# Top plot diagonal lines
ax2_top.plot((-d, +d), (-d, +d), transform=ax2_top.transAxes, color='k', clip_on=False, linewidth=1.5)
ax2_top.plot((1 - d, 1 + d), (-d, +d), transform=ax2_top.transAxes, color='k', clip_on=False, linewidth=1.5)
# Bottom plot diagonal lines
ax2_bottom.plot((-d, +d), (1 - d, 1 + d), transform=ax2_bottom.transAxes, color='k', clip_on=False, linewidth=1.5)
ax2_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax2_bottom.transAxes, color='k', clip_on=False, linewidth=1.5)


# Set labels and titles
ax2_top.set_title('Predicting Quantities in Brass Alloy',fontsize=20)

# fig.text(0.67, 0.55, 'Test Regret', va='center', rotation='vertical', fontsize=18) # Common Y-label
# ax2_top.grid(True)
# ax2_bottom.grid(True)
# ax2_top.set_yticklabels([])  # Remove y-tick labels from top
# ax2_top.set_ylabel('')       # Remove y-axis label from top
# ax2_top.yaxis.set_ticks([])

# --- X-axis limits for all plots ---
ax0.set_xlim(0, 1.1)
ax1.set_xlim(0, 1.1)
# ax2_top.set_xlim(0, 1.1)
# ax2_bottom.set_xlim(0, 1.1)


# --- Legend Handling ---
# Remove individual legends
if ax0.get_legend() is not None: ax0.get_legend().remove()
if ax1.get_legend() is not None: ax1.get_legend().remove()
if ax2_top.get_legend() is not None: ax2_top.get_legend().remove()
if ax2_bottom.get_legend() is not None: ax2_bottom.get_legend().remove()



# Legend in the second row, spanning both columns
ax_legend = fig.add_subplot(gs[1, :])
ax_legend.axis('off')
handles, labels = [], []
for ax in [ax0, ax1, ax2]:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
desired_order = [
    'mse', 'comboptnet', 'sfl','2sPtO',
    'odece(0.2)', 'odece(0.3)', 
    'odece(0.4)', 'odece(0.5)',
     'odece(0.6)', 'odece(0.7)',
      'odece(0.8)',
    
]
label_to_handle = dict(zip(labels, handles))
ordered_handles = [label_to_handle[label] for label in desired_order if label in label_to_handle]
ordered_labels = [label for label in desired_order if label in label_to_handle]

# Map to custom legend labels
custom_legend_map = {
    'mse': 'MSE',
    'comboptnet': 'CombOptNet',
    'sfl': 'SFL',
    '2sPtO': '2sPtO',
    'odece(0.2)': 'Odece (0.2)',
    'odece(0.3)': 'Odece (0.3)',
    'odece(0.4)': 'Odece (0.4)',
    'odece(0.5)': 'Odece (0.5)',
    'odece(0.6)': 'Odece (0.6)',
    'odece(0.7)': 'Odece (0.7)',
    'odece(0.8)': 'Odece (0.8)',
}
new_labels = [custom_legend_map.get(label, label) for label in ordered_labels]


ax_legend.legend(ordered_handles, new_labels, loc='center', ncol=6, fontsize=20)


plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(f"figures/BrokenAxesFixedAlpha_Scatterplot.png")
plt.close()