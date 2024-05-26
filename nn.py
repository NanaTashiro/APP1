import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
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
st.title("Neural Network Model for Predicted Election Results of Party List Vote in Percentage of All Electorates in Auckland")

# Load datasets
@st.cache
def load_data(demo_polls_path, result_list_path, combined_predictions_path, final_predictions_2024_path):
    try:
        demo_polls = pd.read_csv(demo_polls_path)
        result_list = pd.read_csv(result_list_path)
        combined_predictions = pd.read_csv(combined_predictions_path)
        final_predictions_2024 = pd.read_csv(final_predictions_2024_path)
        return demo_polls, result_list, combined_predictions, final_predictions_2024
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

new_merged_demo_polls_path = 'merged_demo_polls.csv'
new_combined_result_list_path = 'combined_result_list.csv'
combined_predictions_path = 'combined_neural_predictions.csv'
final_predictions_2024_path = 'final_neural_predictions_2024.csv'

new_merged_demo_polls, new_combined_result_list, combined_predictions, final_predictions_2024 = load_data(
    new_merged_demo_polls_path, new_combined_result_list_path, combined_predictions_path, final_predictions_2024_path)

if new_merged_demo_polls is None or new_combined_result_list is None or combined_predictions is None or final_predictions_2024 is None:
    st.stop()

# Function to create comparison DataFrame
def create_comparison_df(actual_df, predicted_df, year):
    comparison_df = pd.DataFrame()
    for feature in subset_features:
        actual_values = actual_df[feature].values
        predicted_values = predicted_df[feature].values
        temp_df = pd.DataFrame({
            'Electorate': actual_df['Electorate'],
            'Feature': feature,
            'Actual': actual_values,
            'Predicted': predicted_values,
            'Year': year
        })
        comparison_df = pd.concat([comparison_df, temp_df], ignore_index=True)
    return comparison_df

# Comparison for each year
st.header("Compare Actual and Predicted Votes")
year = st.selectbox("Select Year", [2017, 2020, 2023])
electorate = st.selectbox("Select Electorate", new_combined_result_list['Electorate'].unique())

actual_year = new_combined_result_list[new_combined_result_list['Election Year'] == year].sort_values(by='Electorate')
predictions_year = combined_predictions[combined_predictions['Election Year'] == year].sort_values(by='Electorate')

comparison_df = create_comparison_df(actual_year, predictions_year, year)

# Plot comparison
def plot_comparison(comparison_df, year):
    if comparison_df.empty:
        st.write(f"No common electorates for {year}. Skipping plot.")
        return
    
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
    st.pyplot(g.fig)

plot_comparison(comparison_df, year)

# Predictions for 2024
st.header("Predictions for 2024")
electorate_2024 = st.selectbox("Select Electorate for 2024", final_predictions_2024['Electorate'].unique())

# Function to plot 2024 predictions
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

plot_predictions_2024(final_predictions_2024, electorate_2024)


