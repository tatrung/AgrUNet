#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 15:58:37 2025

@author: tahoangtrung
"""

import pandas as pd
import matplotlib.pyplot as plt

# Data provided by the user
categories = [
    'Water bodies', 'Built-up', 'Aquaculture', 'Rice paddy', 'Coffee',
    'Grassland', 'Orchard', 'Melauleca', 'Mangrove', 'EBF',
    'Rubber Tree', 'Crop', 'Barren', 'Coconut', 'cashew'
]

cross_entropy = [
    0.7324708166, 0.4551729312, 0.6308087285, 0.7791385777, 0.3566372251,
    0.2339214763, 0.2232671276, 0.2298636383, 0.9203830782, 0.6657766525,
    0.597004997, 0.481889304, 0.4045627029, 0.2912651347, 0.2912651347
]
dice_loss = [
    0.7597017858, 0.5086545854, 0.766789543, 0.771054495, 0.8406939874,
    0.7167742112, 0.7744053805, 0.7873064111, 0.9621118095, 0.906890998,
    0.8482607796, 0.7825739027, 0.6789365219, 0.6831916903, 0.2980829865
]
combine_ce_dice = [
    0.7496759559, 0.7401748519, 0.7598465672, 0.7533255321, 0.8408021836,
    0.7340677115, 0.7823924855, 0.8108137406, 0.9549320449, 0.8919948857,
    0.8414863225, 0.7314529598, 0.7870029674, 0.5906948715, 0.6692694955
]
combine_adjust_weight = [
    0.7407262292, 0.7100390873, 0.7664624025, 0.7631142929, 0.8484043233,
    0.7602284535, 0.7599528668, 0.8877527949, 0.9481738115, 0.8910693668,
    0.8362351968, 0.80291984, 0.8962202257, 0.6846741601, 0.6986503678
]

# Percentage data (Category: Percentage)
percentage_data = {
    'Water bodies': 4.77, 'Built-up': 3.29, 'Aquaculture': 4.17, 'Rice paddy': 4.41,
    'Coffee': 12.05, 'Grassland': 3.76, 'Orchard': 7.29, 'Melauleca': 0.56,
    'Mangrove': 12.7, 'EBF': 20.33, 'Rubber Tree': 17.37, 'Crop': 3.3,
    'Barren': 1.68, 'Coconut': 0.51, 'cashew': 3.81
}

# Create a DataFrame for the loss data
df_loss = pd.DataFrame({
    'Category': categories,
    'Cross entropy': cross_entropy,
    'Dice loss': dice_loss,
    'Combined Cross Entropy and Dice Loss': combine_ce_dice,
    'AWCLF': combine_adjust_weight
})

# Add the percentage column
df_loss['Percentage'] = df_loss['Category'].map(percentage_data)

# Sort the DataFrame by Percentage in ascending order
df_loss_sorted = df_loss.sort_values(by='Percentage', ascending=True)

# Melt the DataFrame for easier plotting (long format)
df_plot = df_loss_sorted.melt(
    id_vars=['Category', 'Percentage'],
    var_name='Loss Type',
    value_name='IoU'
)

# Convert Category to a categorical type with the desired order
category_order = df_loss_sorted['Category'].tolist()
df_plot['Category'] = pd.Categorical(
    df_plot['Category'],
    categories=category_order,
    ordered=True
)

# Set up the figure size
plt.figure(figsize=(15, 8))
bar_width = 0.2
categories_pos = range(len(category_order))

# Loss types and labels
loss_types = [
    'Cross entropy',
    'Dice loss',
    'Combined Cross Entropy and Dice Loss',
    'AWCLF'
]
colors = ['skyblue', 'salmon', 'lightgreen', 'gold']

# Plot bars for each loss type
for i, loss_type_col in enumerate(df_loss_sorted.columns[1:5]):
    loss_data = df_plot[df_plot['Loss Type'] == loss_type_col]['IoU'].tolist()
    r = [x + bar_width * i for x in categories_pos]
    
    # Label is the column name
    plt.bar(r, loss_data, color=colors[i], width=bar_width, edgecolor='grey', label=loss_type_col)

# Set a larger font size for the plot elements
plt.rcParams.update({'font.size': 12})
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)

# Add titles and labels with increased font size
plt.xlabel('Land Cover Category (Ordered by Percentage)', fontweight='bold', fontsize=14)
plt.ylabel('IoU', fontweight='bold', fontsize=14)
plt.title('IoU Values by Land Cover Category and Loss Type', fontweight='bold', fontsize=16)

# Set x-axis ticks to be the category names
plt.xticks([r + bar_width * (len(loss_types) - 1) / 2 for r in categories_pos], category_order, rotation=45, ha='right')

# Adjust legend: remove title, move to center (top center)
plt.legend(
    loc='upper left',  # Move to the center of the top edge
    fontsize=12,         # Set font size for the legend
    ncol=4,              # Display in one row (4 columns) to keep it compact and centered
    title=None           # Remove the title
)

# Adjust plot layout
plt.tight_layout()

# Save the figure, overwriting the previous one
plt.savefig('iou_grouped_bar_chart_v4.png')

print("iou_grouped_bar_chart_v4.png")