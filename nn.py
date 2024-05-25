import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# Load datasets
new_merged_demo_polls_path = 'merged_demo_polls.csv'
new_combined_result_list_path = 'combined_result_list.csv'

new_merged_demo_polls = pd.read_csv(new_merged_demo_polls_path)
new_combined_result_list = pd.read_csv(new_combined_result_list_path)

# Prepare the training set (2017, 2020, 2023) and the prediction set (2024)
combined_data_train = pd.concat([new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2017],
                                 new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2020],
                                 new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2023]])

combined_targets_train = pd.concat([new_combined_result_list[new_combined_result_list['Election Year'] == 2017],
                                    new_combined_result_list[new_combined_result_list['Election Year'] == 2020],
                                    new_combined_result_list[new_combined_result_list['Election Year'] == 2023]])

# Prepare the feature set for 2024 prediction
prediction_data = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2024]

# Splitting the data into features (X) and targets (Y)
X_train = combined_data_train.drop(columns=['Election Year', 'Electorate'])
Y_train = combined_targets_train.drop(columns=['Election Year', 'Electorate'])
X_test = prediction_data.drop(columns=['Election Year', 'Electorate'])

# Normalize the data
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

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
grid_result = grid.fit(X_train_normalized, Y_train.values)

best_params = grid_result.best_params_
best_model = grid_result.best_estimator_
best_rmse = np.sqrt(-grid_result.best_score_)

# Cross-validation
cv_scores = cross_val_score(best_model, X_train_normalized, Y_train.values, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)
mean_rmse = np.mean(rmse_scores)

# Train the model on historical data
best_model.fit(X_train_normalized, Y_train.values)

# Make predictions for each year and ensure they are non-negative
def make_predictions(model, data):
    predictions = model.predict(data)
    return np.clip(predictions, 0, None)

predictions_2024 = make_predictions(best_model, X_test_normalized)

# Combine predictions with election year and electorates
def create_predictions_df(predictions, year, electorates):
    df = pd.DataFrame(predictions, columns=Y_train.columns)
    df['Election Year'] = year
    df['Electorate'] = electorates
    return df

electorates_2024 = prediction_data['Electorate'].values
predictions_2024_df = create_predictions_df(predictions_2024, 2024, electorates_2024)

# Streamlit app
st.title("Neural Network Election Prediction Performance")

st.header("Initial Model Performance")
st.write(f"Best parameters: {best_params}")
st.write(f"Best RMSE: {best_rmse}")

st.header("Cross-Validation Performance")
st.write(f"Cross-validation RMSE scores: {rmse_scores}")
st.write(f"Mean cross-validation RMSE: {mean_rmse}")

st.header("Predictions for 2024")
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

plot_predictions_2024(predictions_2024_df)
