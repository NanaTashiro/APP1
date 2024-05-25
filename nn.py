import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# Load your data here
# For example:
new_merged_demo_polls_path = 'merged_demo_polls.csv'
new_combined_result_list_path = 'combined_result_list.csv'

# Define your features and target
X = new_merged_demo_polls.drop(columns=['Election Year', 'Electorate'])
Y = Y_train_model

# Prepare the data
def prepare_data(df, year):
    return df[df['Election Year'] == year].drop(columns=['Election Year', 'Electorate'])

# Prepare the data for each year
X_2017 = prepare_data(new_merged_demo_polls, 2017)
X_2020 = prepare_data(new_merged_demo_polls, 2020)
X_2023 = prepare_data(new_merged_demo_polls, 2023)
X_2024 = prepare_data(new_merged_demo_polls, 2024)

# Normalize the data
scaler = MinMaxScaler()
X_2017_normalized = scaler.fit_transform(X_2017)
X_2020_normalized = scaler.transform(X_2020)
X_2023_normalized = scaler.transform(X_2023)
X_2024_normalized = scaler.transform(X_2024)

# Define the model and parameters
param_grid = {
    'hidden_layer_sizes': [(50,), (50, 50)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'batch_size': [32, 64],
    'learning_rate': ['constant'],
    'learning_rate_init': [0.1],
    'max_iter': [200],
    'early_stopping': [True]
}

# Initial model
mlp = MLPRegressor(random_state=42)
grid = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_result = grid.fit(X_2017_normalized, Y_train_model.values)

best_params = grid_result.best_params_
best_model = grid_result.best_estimator_
best_rmse = np.sqrt(-grid_result.best_score_)

# Cross-validation
X_historical = new_merged_demo_polls[new_merged_demo_polls['Election Year'].isin([2017, 2020, 2023])].drop(columns=['Election Year', 'Electorate'])
y_historical = Y_train_model.loc[new_merged_demo_polls['Election Year'].isin([2017, 2020, 2023])]

X_historical_normalized = scaler.fit_transform(X_historical)

cv_scores = cross_val_score(best_model, X_historical_normalized, y_historical, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)
mean_rmse = np.mean(rmse_scores)

# Train the model on historical data
best_model.fit(X_historical_normalized, y_historical)

# Make predictions for each year and ensure they are non-negative
def make_predictions(model, data):
    predictions = model.predict(data)
    return np.clip(predictions, 0, None)

predictions_2017 = make_predictions(best_model, X_2017_normalized)
predictions_2020 = make_predictions(best_model, X_2020_normalized)
predictions_2023 = make_predictions(best_model, X_2023_normalized)
predictions_2024 = make_predictions(best_model, X_2024_normalized)

# Combine predictions with election year and electorates
def create_predictions_df(predictions, year, electorates):
    df = pd.DataFrame(predictions, columns=Y_train_model.columns)
    df['Election Year'] = year
    df['Electorate'] = electorates
    return df

electorates_2017 = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2017]['Electorate'].values
electorates_2020 = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2020]['Electorate'].values
electorates_2023 = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2023]['Electorate'].values
electorates_2024 = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2024]['Electorate'].values

predictions_2017_df = create_predictions_df(predictions_2017, 2017, electorates_2017)
predictions_2020_df = create_predictions_df(predictions_2020, 2020, electorates_2020)
predictions_2023_df = create_predictions_df(predictions_2023, 2023, electorates_2023)
predictions_2024_df = create_predictions_df(predictions_2024, 2024, electorates_2024)

# Combine predictions for all years into a single DataFrame
all_predictions_df = pd.concat([predictions_2017_df, predictions_2020_df, predictions_2023_df])
all_predictions_df = all_predictions_df[['Election Year', 'Electorate'] + list(Y_train_model.columns)]

# Plot comparisons and predictions
def plot_comparison(comparison_df, year):
    comparison_melted = pd.melt(comparison_df, id_vars=['Electorate', 'Feature'], value_vars=['Actual', 'Predicted'], var_name='Type', value_name='Votes')
    g = sns.FacetGrid(comparison_melted, col='Electorate', col_wrap=4, height=4, sharey=False)
    g.map_dataframe(sns.lineplot, x='Feature', y='Votes', hue='Type', marker='o')
    g.add_legend()
    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(90)
    g.set_titles("{col_name}")
    g.set_axis_labels("Party", "Votes")
    plt.subplots_adjust(top=0.92)
    g.fig.suptitle(f'Comparison of Actual and Predicted Votes by Party for Each Electorate in {year} in Auckland Region')
    plt.show()

def plot_predictions_2024(predictions_df):
    subset_features = ['National Party Vote', 'Labour Party Vote', 'Green Party Vote', 'New Zealand First Party Vote', 'ACT New Zealand Vote', 'Others Vote']
    party_colors = {
        'ACT New Zealand Vote': 'yellow',
        'Green Party Vote': 'green',
        'Labour Party Vote': 'red',
        'National Party Vote': 'blue',
        'New Zealand First Party Vote': 'black',
        'Others Vote': 'grey'
    }
    predictions_melted = pd.melt(predictions_df, id_vars=['Electorate'], value_vars=subset_features, var_name='Party', value_name='Votes')
    g = sns.FacetGrid(predictions_melted, col='Electorate', col_wrap=4, height=4, sharey=False, hue='Party', palette=party_colors)
    g.map_dataframe(sns.barplot, x='Party', y='Votes', palette=party_colors)
    for ax in g.axes.flatten():
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', label_type='edge', fontsize=8, padding=3)
    g.add_legend()
    for ax in g.axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel("Party", fontsize=12)
        ax.set_ylabel("Votes", fontsize=12)
    g.set_titles("{col_name}", size=13)
    g.set_axis_labels("Party", "Votes")
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.4)
    g.fig.suptitle('Predicted Votes by Party for Each Electorate in Auckland Region in 2024', fontsize=16)
    plt.show()

# Streamlit app
st.title("Neural Network Election Prediction Performance")

st.header("Initial Model Performance")
st.write(f"Best parameters: {best_params}")
st.write(f"Best RMSE: {best_rmse}")

st.header("Cross-Validation Performance")
st.write(f"Cross-validation RMSE scores: {rmse_scores}")
st.write(f"Mean cross-validation RMSE: {mean_rmse}")

st.header("Comparison of Actual and Predicted Votes")
# Add your code to display the comparison plots for 2017, 2020, and 2023

st.header("Predictions for 2024")
plot_predictions_2024(predictions_2024_df)
