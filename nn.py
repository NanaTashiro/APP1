import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
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

# Define the model with the best parameters
best_model = MLPRegressor(
    activation='relu',
    alpha=0.0001,
    hidden_layer_sizes=(50, 50),
    learning_rate='constant',
    learning_rate_init=0.1,
    max_iter=500,
    solver='adam',
    batch_size=64,
    early_stopping=True,
    random_state=42
)

# Perform cross-validation
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

# Combine predictions for all years into a single DataFrame
electorates_2017 = combined_data_train[combined_data_train['Election Year'] == 2017]['Electorate'].values
electorates_2020 = combined_data_train[combined_data_train['Election Year'] == 2020]['Electorate'].values
electorates_2023 = combined_data_train[combined_data_train['Election Year'] == 2023]['Electorate'].values

predictions_2017 = make_predictions(best_model, scaler.transform(combined_data_train[combined_data_train['Election Year'] == 2017].drop(columns=['Election Year', 'Electorate'])))
predictions_2020 = make_predictions(best_model, scaler.transform(combined_data_train[combined_data_train['Election Year'] == 2020].drop(columns=['Election Year', 'Electorate'])))
predictions_2023 = make_predictions(best_model, scaler.transform(combined_data_train[combined_data_train['Election Year'] == 2023].drop(columns=['Election Year', 'Electorate'])))

predictions_2017_df = create_predictions_df(predictions_2017, 2017, electorates_2017)
predictions_2020_df = create_predictions_df(predictions_2020, 2020, electorates_2020)
predictions_2023_df = create_predictions_df(predictions_2023, 2023, electorates_2023)

all_predictions_df = pd.concat([predictions_2017_df, predictions_2020_df, predictions_2023_df])
all_predictions_df = all_predictions_df[['Election Year', 'Electorate'] + list(Y_train.columns)]

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

# Streamlit app
st.title("Neural Network Election Prediction Performance")

st.header("Initial Model Performance")
st.write(f"Best parameters: {{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50), 'learning_rate': 'constant', 'max_iter': 500, 'solver': 'adam'}}")

st.header("Cross-Validation Performance")
st.write(f"Cross-validation RMSE scores: {rmse_scores}")
st.write(f"Mean cross-validation RMSE: {mean_rmse}")

# User inputs for comparison
st.header("Compare Actual and Predicted Votes")
year = st.selectbox("Select Year", [2017, 2020, 2023])
electorate = st.selectbox("Select Electorate", new_combined_result_list['Electorate'].unique())

comparison_df = create_comparison_df(year, electorate)
plot_comparison(comparison_df, year, electorate)

# User input for 2024 predictions
st.header("Predictions for 2024")
electorate_2024 = st.selectbox("Select Electorate for 2024", prediction_data['Electorate'].unique())
plot_predictions_2024(predictions_2024_df, electorate_2024)


