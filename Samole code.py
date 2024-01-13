# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:07:45 2023

@author: USER
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis

# Define file paths
population_data_path = 'Dataset/population.csv'
pivoted_data_path = 'Dataset/PivotedDataset.csv'
cleaned_data_path = 'Dataset/CleanedDataset.csv'

# Function to load dataset


def load_dataset(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    dataset = pd.read_csv(file_path)
    return dataset


# Define indicator codes, indicator map, and country names
indicator_codes = ['Country Name', 'Country Code', 'Year', 'SP.POP.TOTL.MA.ZS', 'SP.POP.TOTL.FE.ZS', 'SP.POP.GROW',
                   'SP.POP.DPND', 'SP.DYN.CDRT.IN', 'SP.DYN.CBRT.IN', 'SP.POP.1564.TO.ZS', 'SH.DTH.IMRT']

indicator_map = {
    "SP.POP.TOTL.MA.ZS": "Male population ",
    "SP.POP.TOTL.FE.ZS": "Female population",
    "SP.POP.GROW": "Population growth",
    "SP.POP.DPND": "Dependency age ratio",
    "SP.DYN.CDRT.IN": "Death rate",
    "SP.DYN.CBRT.IN": "Birth rate",
    "SP.POP.1564.TO.ZS": "Working age population",
    "SH.DTH.IMRT": "Number of infant deaths"
}

country_names = {
    "CAN": "Canada",
    "DEU": "Germany",
    "ARG": "Argentina",
    "JPN": "Japan",
    "SWE": "Sweden",
    "UGA": "Uganda"
}


def load_and_melt_dataset(file_path):
    """
    Load the dataset from a CSV file and melt it.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Melted dataset with columns 'Country Name', 'Country Code', 'Indicator Name',
      'Indicator Code', 'Year', and 'Value'.
    """
    dataset = load_dataset(file_path)
    melted_df = dataset.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                             var_name='Year', value_name='Value')
    return melted_df


# Example usage
melted_population_df = load_and_melt_dataset(population_data_path)


def pivot_and_filter_dataset(melted_df, indicator_codes):
    """
    Pivot the melted dataset, fill NaN values with 0, and filter columns based on indicator codes.

    Parameters:
    - melted_df (pd.DataFrame): Melted dataset.
    - indicator_codes (list): List of indicator codes to keep.

    Returns:
    - pd.DataFrame: Pivoted, filled, and filtered dataset.
    """
    # Pivot the table to have Indicator Codes as columns
    pivoted_df = melted_df.pivot_table(index=['Country Name', 'Country Code', 'Year'], columns='Indicator Code',
                                       values='Value').reset_index()

    # Fill NaN values with 0
    pivoted_df.fillna(0, inplace=True)

    # Filter columns based on indicator codes
    filtered_columns = [
        col for col in pivoted_df.columns if col in indicator_codes]
    filtered_df = pivoted_df[filtered_columns]

    return filtered_df


# Example usage
melted_population_df = load_and_melt_dataset(population_data_path)
filtered_df = pivot_and_filter_dataset(melted_population_df, indicator_codes)
filtered_df.to_csv(cleaned_data_path, index=False)


# Clean the transformed dataset
# Fill missing values with the mean of the column
cleaned_df = filtered_df.fillna(filtered_df.mean(numeric_only=True))
cleaned_df.to_csv(cleaned_data_path)


# Create DataFrames for specific countries
df_argentina = cleaned_df[cleaned_df["Country Name"] == "Argentina"]
df_germany = cleaned_df[cleaned_df["Country Name"] == "Germany"]
df_uganda = cleaned_df[cleaned_df["Country Name"] == "Uganda"]


# Apply statistical methods on the cleaned dataset
statistical_df = cleaned_df.drop(['Year', 'Country Name'], axis='columns')
print(statistical_df.describe())

stats_skew = skew(df_argentina["SP.POP.GROW"])
print(stats_skew)

stats_kurtosis = kurtosis(df_argentina["SP.POP.GROW"])
print(stats_kurtosis)


# Correlation Matrix and Heat map for Argentina
correlation_matrix_argentina = df_argentina.corr(numeric_only=True).rename(
    columns=indicator_map).rename(index=indicator_map)
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix_argentina,
            annot=True, fmt=".1g", vmax=1, vmin=0)
plt.title('Correlation Matrix for Argentina')
plt.show()

# Correlation Matrix and Heat map for Germany
correlation_matrix_germany = df_germany.corr(numeric_only=True).rename(
    columns=indicator_map).rename(index=indicator_map)
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix_germany, annot=True, fmt=".1g", vmax=1, vmin=0)
plt.title('Correlation Matrix for Germany')
plt.show()

# Correlation Matrix and Heat map for Uganda
correlation_matrix_uganda = df_uganda.corr(numeric_only=True).rename(
    columns=indicator_map).rename(index=indicator_map)
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix_uganda, annot=True, fmt=".1g", vmax=1, vmin=0)
plt.title('Correlation Matrix for Uganda')
plt.show()


# Bar graphs
# Filtering for specific countries and years
df_cleaned = pd.read_csv(cleaned_data_path)

filtered_population = df_cleaned[
    ((df_cleaned['Country Name'] == 'Argentina') | (df_cleaned['Country Name'] == 'Germany') |
     (df_cleaned['Country Name'] == 'Japan') | (df_cleaned['Country Name'] == 'Canada') | (df_cleaned['Country Name'] == 'Uganda')) &
    ((df_cleaned['Year'] == 1975) | (df_cleaned['Year'] == 1985) |
     (df_cleaned['Year'] == 1995) | (df_cleaned['Year'] == 2005))]

filtered_population = filtered_population[[
    "Country Name", "Year", "SP.DYN.CBRT.IN", "SH.DTH.IMRT"]]


# Pivot the data for the first bar graph about birth rate per year
pivoted_birth_rate_df = filtered_population.pivot(
    index='Country Name', columns='Year', values='SP.DYN.CBRT.IN').reset_index()

# Plotting the bar graph for birth rate
ax = pivoted_birth_rate_df.plot(kind='bar', x='Country Name', y=[
                                1975, 1985, 1995, 2005], rot=30, legend=True)
plt.xticks(rotation=30, horizontalalignment="center")

# Adding labels and title for birth rate
ax.set_ylabel('Birth Rate (crude per 1,000 people)')
ax.set_xlabel('Country')
plt.title('Birth Rate Per 1,000 People')

# Show the plot
plt.show()

# Bar graph for infant death
pivoted_infant_death_df = filtered_population.pivot(
    index='Country Name', columns='Year', values='SH.DTH.IMRT').reset_index()

# Plotting the bar graph for infant death
ax = pivoted_infant_death_df.plot(kind='bar', x='Country Name', y=[
                                  1975, 1985, 1995, 2005], rot=30, legend=True)
plt.xticks(rotation=0, horizontalalignment="center")

# Adding labels and title for infant death
ax.set_xlabel('Country')
plt.title('Infant Death Rate')

# Show the plot
plt.show()

# Line graphs for working age people and dependency age people
# Filter data for specific countries
age_argentina = df_cleaned[df_cleaned['Country Name'] == "Argentina"]
age_germany = df_cleaned[df_cleaned['Country Name'] == "Germany"]
age_japan = df_cleaned[df_cleaned['Country Name'] == "Japan"]
age_sweden = df_cleaned[df_cleaned['Country Name'] == "Canada"]
age_uganda = df_cleaned[df_cleaned['Country Name'] == "Uganda"]

# Plotting the first line graph Dependency age
plt.plot(age_argentina["Year"],
         age_argentina["SP.POP.DPND"], label='Argentina')
plt.plot(age_germany["Year"], age_germany["SP.POP.DPND"], label='Germany')
plt.plot(age_japan["Year"], age_japan["SP.POP.DPND"], label='Japan')
plt.plot(age_sweden["Year"], age_sweden["SP.POP.DPND"], label='Canada')
plt.plot(age_uganda["Year"], age_uganda["SP.POP.DPND"], label='Uganda')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Population of dependency age group ')
plt.title('Dependency Age Group Comparison')

# Adding legend
plt.legend()

# Display the plot
plt.show()

# Second Line graph for Population ages 15-64 (% of total population)
plt.plot(age_argentina["Year"],
         age_argentina["SP.POP.1564.TO.ZS"], label='Argentina')
plt.plot(age_germany["Year"],
         age_germany["SP.POP.1564.TO.ZS"], label='Germany')
plt.plot(age_japan["Year"], age_japan["SP.POP.1564.TO.ZS"], label='Japan')
plt.plot(age_sweden["Year"], age_sweden["SP.POP.1564.TO.ZS"], label='Canada')
plt.plot(age_uganda["Year"], age_uganda["SP.POP.1564.TO.ZS"], label='Uganda')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Population of 15-64 Age Group (% of Total Population)')
plt.title('Population of Working-Age People')

# Adding legend
plt.legend()

# Display the plot
plt.show()
