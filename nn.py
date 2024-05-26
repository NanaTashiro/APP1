import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
final_neural_predictions_2024 = pd.read_csv('final_neural_predictions_2024.csv')
combined_neural_predictions = pd.read_csv('combined_neural_predictions.csv')
final_neural_predictions1_2024 = pd.read_csv('final_neural_predictions1_2024.csv')

# Parameters and RMSE
st.title("Neural Network Model of Predicted Election Results of Party List Vote in Percentage of All Electorates in Auckland")

st.subheader("Best Parameters from Neural Network Tuning")
st.write("""
- Activation: 'relu'
- Alpha: 0.0001
- Batch Size: 64
- Early Stopping: True
- Hidden Layer Sizes: (50,)
- Learning Rate: 'constant'
- Learning Rate Init: 0.1
- Max Iter: 200
- Solver: 'adam'
- RMSE: 9.284181455443612
""")

st.subheader("Prediction for 2023 Election")
st.dataframe(final_neural_predictions_2024)

st.write("""
Based on the distribution of Labour Party vote predictions for 2024, it appears that the majority of the predictions are clustered at negative. This result is concerning, especially considering the Labour Party's historical significance and strong presence in New Zealand politics.
""")

st.subheader("Cross-validate the Model on a Subset of Historical Data")
st.write("""
Cross-validation RMSE scores: [10.58263428, 7.5126416, 9.74489136, 8.80247001, 8.84035454]
""")
st.write("""
Mean cross-validation RMSE: 9.09659835907756
""")

st.subheader("Combined Neural Prediction for 2017, 2020, 2023")
st.dataframe(combined_neural_predictions)

st.subheader("Final Neural Prediction after Cross-validation")
st.dataframe(final_neural_predictions1_2024)

# Normalize and map electorate names
def normalize_and_map_electorate_names(df, mapping):
    df['Electorate'] = df['Electorate'].str.lower().str.strip().replace(mapping)
    return df

electorate_mapping = {
    # Add mappings here if necessary
}

new_combined_result_list = normalize_and_map_electorate_names(combined_neural_predictions, electorate_mapping)
all_predictions_df = normalize_and_map_electorate_names(final_neural_predictions_2024, electorate_mapping)

# Function to create comparison DataFrame for a single year
def create_comparison_df(year, electorate):
    actual_df = new_combined_result_list[(new_combined_result_list['Election Year'] == year) & (new_combined_result_list['Electorate'] == electorate)]
    predicted_df = all_predictions_df[(all_predictions_df['Election Year'] == year) & (all_predictions_df['Electorate'] == electorate)]
    
    comparison_df = pd.DataFrame()
    for feature in ['ACT New Zealand Vote', 'Green Party Vote', 'Labour Party Vote', 'National Party Vote', 'New Zealand First Party Vote', 'Others Vote']:
        actual_values = actual_df[feature].values
        predicted_values = predicted_df[feature].values
        
        # Handle mismatched lengths by aligning data
        min_length = min(len(actual_values), len(predicted_values))
        actual_values = actual_values[:min_length]
        predicted_values = predicted_values[:min_length]
        
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
    ax.set_xticks(range(len(['ACT New Zealand Vote', 'Green Party Vote', 'Labour Party Vote', 'National Party Vote', 'New Zealand First Party Vote', 'Others Vote'])))
    ax.set_xticklabels(['ACT', 'Green', 'Labour', 'National', 'NZ First', 'Others'], rotation=45, ha='right')
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
    
    predictions = predictions.melt(id_vars=['Electorate'], value_vars=['ACT New Zealand Vote', 'Green Party Vote', 'Labour Party Vote', 'National Party Vote', 'New Zealand First Party Vote', 'Others Vote'], var_name='Party', value_name='Votes')
    
    fig, ax = plt.subplots()
    sns.barplot(data=predictions, x='Party', y='Votes', palette='viridis', ax=ax)
    
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
electorate_2024 = st.selectbox("Select Electorate for 2024", final_neural_predictions_2024['Electorate'].unique())
plot_predictions_2024(final_neural_predictions_2024, electorate_2024)

