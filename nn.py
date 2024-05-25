import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# Subset of features (party votes)
subset_features = ['National Party Vote', 'Labour Party Vote', 'Green Party Vote', 'New Zealand First Party Vote', 'ACT New Zealand Vote', 'Others Vote']

# Defining party colors
party_colors = {
    'ACT New Zealand Vote': 'yellow',
    'Green Party Vote': 'green',
    'Labour Party Vote': 'red',
    'National Party Vote': 'blue',
    'New Zealand First Party Vote': 'black',
    'Others Vote': 'grey'
}

# Set the title of the Streamlit app
st.title("Neural Network Model of Predicted Election Results of Party List Vote in Percentage of All Electorates in Auckland")

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

# Normalize and standardize data
min_max_scaler = MinMaxScaler()
X_train_normalized = min_max_scaler.fit_transform(X_train)
X_test_normalized = min_max_scaler.transform(X_test)

standard_scaler = StandardScaler()
X_train_standardized = standard_scaler.fit_transform(X_train)
X_test_standardized = standard_scaler.transform(X_test)

# Define the neural network model with the best parameters
best_model_neural = MLPRegressor(
    activation='relu',
    alpha=0.0001,
    batch_size=64,
    early_stopping=True,
    hidden_layer_sizes=(50,),
    learning_rate='constant',
    learning_rate_init=0.1,
    max_iter=200,
    solver='adam',
    random_state=42
)

# Function to perform cross-validation and return the mean RMSE
def cross_val_rmse(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores.mean()

# Cross-validate RMSE values for neural network
normalized_rmse_cv = cross_val_rmse(best_model_neural, X_train_normalized, Y_train)
standardized_rmse_cv = cross_val_rmse(best_model_neural, X_train_standardized, Y_train)

st.write(f"Cross-Validated RMSE for Neural Network (normalized): {normalized_rmse_cv}")
st.write(f"Cross-Validated RMSE for Neural Network (standardized): {standardized_rmse_cv}")

# Train the final model on the entire training data (normalized)
best_model_neural.fit(X_train_normalized, Y_train)

# Predictions for 2024
final_predictions_2024 = best_model_neural.predict(X_test_normalized)
final_predictions_2024_df = pd.DataFrame(final_predictions_2024, columns=Y_train.columns)
final_predictions_2024_df['Election Year'] = 2024
final_predictions_2024_df['Electorate'] = prediction_data['Electorate'].values

# Save the 2024 predictions to CSV
final_predictions_2024_df.to_csv('final_neural_predictions_2024.csv', index=False)

# Prepare data for past predictions
X_2017 = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2017].drop(columns=['Election Year', 'Electorate'])
X_2020 = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2020].drop(columns=['Election Year', 'Electorate'])
X_2023 = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2023].drop(columns=['Election Year', 'Electorate'])

# Normalize the data
X_2017_normalized = min_max_scaler.transform(X_2017)
X_2020_normalized = min_max_scaler.transform(X_2020)
X_2023_normalized = min_max_scaler.transform(X_2023)

# Make predictions for each year
predictions_2017 = best_model_neural.predict(X_2017_normalized)
predictions_2020 = best_model_neural.predict(X_2020_normalized)
predictions_2023 = best_model_neural.predict(X_2023_normalized)

# Combine predictions with election year and electorates
predictions_2017_df = pd.DataFrame(predictions_2017, columns=Y_train.columns)
predictions_2017_df['Election Year'] = 2017
predictions_2017_df['Electorate'] = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2017]['Electorate'].values

predictions_2020_df = pd.DataFrame(predictions_2020, columns=Y_train.columns)
predictions_2020_df['Election Year'] = 2020
predictions_2020_df['Electorate'] = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2020]['Electorate'].values

predictions_2023_df = pd.DataFrame(predictions_2023, columns=Y_train.columns)
predictions_2023_df['Election Year'] = 2023
predictions_2023_df['Electorate'] = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2023]['Electorate'].values

# Combine predictions for all years into a single DataFrame
all_predictions_df = pd.concat([predictions_2017_df, predictions_2020_df, predictions_2023_df])

# Reorder columns to place 'Election Year' and 'Electorate' at the front
all_predictions_df = all_predictions_df[['Election Year', 'Electorate'] + list(Y_train.columns)]

# Save combined predictions to CSV
all_predictions_df.to_csv('combined_neural_predictions.csv', index=False)

# Define the electorate mapping based on inspection
electorate_mapping = {
    # Add mappings here based on discrepancies found in inspection
}

# Normalize and map electorate names
def normalize_and_map_electorate_names(df, mapping):
    df['Electorate'] = df['Electorate'].str.lower().str.strip().replace(mapping)
    return df

new_combined_result_list = normalize_and_map_electorate_names(new_combined_result_list, electorate_mapping)
all_predictions_df = normalize_and_map_electorate_names(all_predictions_df, electorate_mapping)

# Function to create comparison DataFrame for a single year
def create_comparison_df(year, electorate):
    actual_df = new_combined_result_list[(new_combined_result_list['Election Year'] == year) & (new_combined_result_list['Electorate'] == electorate)]
    predicted_df = all_predictions_df[(all_predictions_df['Election Year'] == year) & (all_predictions_df['Electorate'] == electorate)]
    
    comparison_df = pd.DataFrame()
    for feature in subset_features:
        actual_values = actual_df[feature].values
        predicted_values = predicted_df[feature].values
        
        temp_df = pd.DataFrame({
            'Feature': feature,
            'Actual': actual_values,
            'Predicted': predicted_values
        })
        comparison_df = pd.concat([comparison_df, temp_df], ignore_index=True)
    return comparison_df

# Function to plot comparison for a single year
def plot_comparison(comparison_df, year, electorate):
    if comparison_df.empty:
        st.write(f"No common electorates for {year}. Skipping plot.")
        return
    
    fig, ax = plt.subplots()
    comparison_df = comparison_df.melt(id_vars=['Feature'], value_vars=['Actual', 'Predicted'], var_name='Type', value_name='Votes')
    sns.lineplot(data=comparison_df, x='Feature', y='Votes', hue='Type', marker='o', ax=ax)
    ax.set_title(f'Comparison of Actual and Predicted Votes by Party for {electorate} in {year}')
    ax.set_xlabel('Party')
    ax.set_ylabel('Votes')
    ax.set_xticks(range(len(subset_features)))
    ax.set_xticklabels(subset_features, rotation=45, ha='right')
    ax.legend()
    st.pyplot(fig)

# User inputs for comparison
st.header("Compare Actual and Predicted Votes")
year = st.selectbox("Select Year", [2017, 2020, 2023])
electorate = st.selectbox("Select Electorate", new_combined_result_list['Electorate'].unique())

comparison_df = create_comparison_df(year, electorate)
plot_comparison(comparison_df, year, electorate)

# Function to plot 2024 predictions with color by party name as bar charts and add results number
def plot_predictions_2024(predictions_df, electorate):
    predictions = predictions_df[predictions_df['Electorate'] == electorate]
    if predictions.empty:
        st.write(f"No data available for {electorate} in 2024.")
        return
    
    predictions = predictions.melt(id_vars=['Electorate'], value_vars=subset_features, var_name='Party', value_name='Votes')
    
    fig, ax = plt.subplots()
    sns.barplot(data=predictions, x='Party', y='Votes', palette=party_colors, ax=ax)
    
    # Add vote counts on top of each bar
    for index, row in predictions.iterrows():
        ax.text(index, row['Votes'], f'{row["Votes"]:.2f}%', color='black', ha="center")
    
    ax.set_title(f'Predicted Votes for {electorate} in 2024')
    ax.set_xlabel('Party')
    ax.set_ylabel('Votes (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

# User input for 2024 predictions
st.header("Predictions for 2024")
electorate_2024 = st.selectbox("Select Electorate for 2024", prediction_data['Electorate'].unique())
plot_predictions_2024(final_predictions_2024_df, electorate_2024)


