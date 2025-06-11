## Omri Naftali
    
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import parallel_coordinates
import numpy as np
from prophet import Prophet
import colorcet as cc

# Loading academic datasets (per year)
directory = 'raw_data' # Directory containing the Excel files
dfs = []

# 1 - Loop through each year and read the corresponding Excel file
for year in range(2010, 2024):
    file_path = os.path.join(directory, f'{year}.xlsx')
    df = pd.read_excel(file_path)
    df['Year'] = year  # Add a new column for the year
    dfs.append(df)

# Combine all dataframes into one
academic_data = pd.concat(dfs, ignore_index=True)

# Loading democratic dataset:
democratic_data = pd.read_csv(f'{directory}\\democracy-index-eiu.csv')

# Remove row for year earlier then 2010 from democratic_data
democratic_data = democratic_data[democratic_data['Year'] >= 2010]

# Examine the datasets
print("academic data:\n", academic_data.info())
print(academic_data.describe())
print("democratic data:\n", democratic_data.info())
print(democratic_data.describe())
      
# Count how many nan values
print("academic data:\n", academic_data.isnull().sum())
print("democratic data:\n", democratic_data.isnull().sum())

# 2 - Remove rows with null values from democratic_data dataset
democratic_data = democratic_data.dropna(subset=['Code'])

# 3 - Get all entities values (continents name):
unique_regions_values = democratic_data['Entity'].unique()
print(unique_regions_values)

# 4 - Adjusting the names of the continents and regions
academic_data.loc[academic_data['Region'].str.contains('Europe', na=False), 'Region'] = 'Europe'
academic_data['Region'].replace('Latin America', "South America", inplace=True)
academic_data['Region'].replace('Northern America', "North America", inplace=True)
academic_data['Region'].replace('Asiatic Region', "Asia", inplace=True)
academic_data['Region'].replace('Africa/Middle East', "Africa", inplace=True)
academic_data['Region'].replace('Middle East', "Asia", inplace=True)
academic_data['Region'].replace('Pacific Region', "Oceania", inplace=True)

# Remove "Rank" column from academic_data
academic_data.drop('Rank', axis=1, inplace=True)

# Remove "Code" column from democratic_data
democratic_data.drop('Code', axis=1, inplace=True)

# 5 - Renaming the 'entity' column in democratic_data to 'Country' for consistency
democratic_data.rename(columns={'Entity': 'Country'}, inplace=True)

# 6 - Merging the countries datasets on 'Country' and 'Year'
countries_merged_dataset = pd.merge(academic_data, democratic_data, on=['Country', 'Year'], how='outer')

# 7 - Count how many nan values
print(countries_merged_dataset.isnull().sum())

# 8 - Group by 'Country' and remove countries where all 'Democracy score' values are NaN
countries_without_any_democracy_score = countries_merged_dataset[countries_merged_dataset['Democracy score'].isna()].groupby('Country').filter(lambda x: len(x) == len(countries_merged_dataset[countries_merged_dataset['Country'] == x.name])).index
countries_merged_dataset = countries_merged_dataset.drop(countries_without_any_democracy_score)

# 9 - Remove rows with null values from dataset
countries_merged_dataset = countries_merged_dataset[countries_merged_dataset['Year'] != 2023]

# 10 - Group by 'Country' and remove countries where all academic score values are NaN
countries_without_any_academic_score = countries_merged_dataset[countries_merged_dataset['H index'].isna()].groupby('Country').filter(lambda x: len(x) == len(countries_merged_dataset[countries_merged_dataset['Country'] == x.name])).index
countries_merged_dataset = countries_merged_dataset.drop(countries_without_any_academic_score)

# ----- Visualization ----- #

# 1 - Democratic score histogram across different regions
plt.figure(figsize=(14, 8))
sns.histplot(data=countries_merged_dataset, x='Democracy score', hue='Region', multiple='stack', palette="Set2")
plt.title('Distribution of Democracy Scores Across Different Regions', fontsize=20, fontweight='bold')
plt.xlabel('Democracy Score')
plt.ylabel('Count')
plt.show()

# 2 - Box plot distribution of democracy scores across different regions
plt.figure(figsize=(14, 7))
sns.boxplot(data=countries_merged_dataset, x='Region', y='Democracy score', palette="Set3")
sns.stripplot(data=countries_merged_dataset, x='Region', y='Democracy score', hue='Region', dodge=True, palette="Set2", jitter=True, alpha=0.6)
plt.title('Democracy Score Distribution by Region with Countries Differentiated', fontsize=20, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# 3 - Academic scores box plot across different regions
academic_metrics = ['Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index'] 

# Plotting the distribution of academic variables across different regions
plt.figure(figsize=(14, 10))
# Loop through each academic column and create a histogram for each, grouped by region
for i, metric in enumerate(academic_metrics, 1):
    plt.subplot(2, 3, i)
    sns.histplot(countries_merged_dataset, x=metric, hue='Region', multiple='stack', bins=20, palette='Set2')
    plt.title(f'Distribution of {metric} by Region', fontsize=13, fontweight='bold')
    plt.xlabel(metric)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4 - Plotting the distribution of academic variables across different regions - Box plot
plt.figure(figsize=(18, 12))
# Loop through each academic column and create a boxplot for each
for i, metric in enumerate(academic_metrics, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='Region', y=metric, data=countries_merged_dataset, palette="Set3", dodge=True)
    plt.title(f'Distribution of {metric} by Region', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45)
    plt.xlabel('Region')
    plt.ylabel(metric)
plt.tight_layout()
plt.show()
    
# 5 - Create box plot for each academic index by region
for metric in academic_metrics:
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Region', y=metric, data=countries_merged_dataset, palette="Set3")
    sns.stripplot(data=countries_merged_dataset, x='Region', y=metric, hue='Region', dodge=True, palette="Set2", jitter=True, alpha=0.6)
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.title(f'Box Plot of {metric} by Region', fontsize=20, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

# 6 - Bar chart - count the number of countries in each region
plt.figure(figsize=(14, 8))
# Assign different colors for each region
region_counts = countries_merged_dataset.groupby('Region')['Country'].nunique()
region_counts.plot(kind='bar', color=plt.cm.Set3(range(len(region_counts))), legend=False)
plt.title('Number of Countries by Region', fontsize=20, fontweight='bold')
plt.xlabel('Region')
plt.ylabel('Number of Countries')
plt.xticks(rotation=45)
plt.grid(axis='y')

# 7 - Heat-map for correlation matrix
metrics = ['Democracy score', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index']
# Calculate the correlation matrix
spearman_correlation_matrix = countries_merged_dataset[metrics].corr(method='spearman')
plt.figure(figsize=(14, 10))
sns.heatmap(spearman_correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Spearman Correlation Matrix', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show()

# 8 - Lineplot to show democracy score per regions over the years
ax = sns.lineplot(data=countries_merged_dataset, x='Year', y='Democracy score', hue='Region')
plt.xlabel('Year')
plt.ylabel('Democracy score')
plt.title('Democracy score per regions over the years\n', fontsize=20, fontweight='bold')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1,1))
plt.show()

# Create a pivot table to aggregate data by year and region for all academic metrics
pivot_data_all_metrics = countries_merged_dataset.pivot_table(index='Year', columns='Region', values=academic_metrics, aggfunc='mean')

# 9 - Line plot - aggregated data for each region over the years for all academic metrics 
plt.figure(figsize=(20, 24))

titles = ['Average Number of Documents', 'Average Number of Citable Documents', 
          'Average Number of Citations', 'Average Number of Self-Citations', 
          'Average Citations per Document', 'Average H Index']

# Loop to create subplots for each metric
for i, metric in enumerate(academic_metrics, 1):
    plt.subplot(3, 2, i)
    for region in pivot_data_all_metrics[metric].columns:
        plt.plot(pivot_data_all_metrics.index, pivot_data_all_metrics[metric][region], label=region)
    plt.title(f'{titles[i-1]} by Year and Region', fontsize=16, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel(metric)
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle('Trends of Various Academic Indicators Over Years by Region', fontsize=40, fontweight='bold')

# 10 - Scatter plots with trend lines (Relationship between Democracy Score and H Index) for each region
regions = countries_merged_dataset['Region'].unique()
custom_palette = sns.color_palette(cc.glasbey, n_colors=50)

for region in regions:
    regional_data = countries_merged_dataset[countries_merged_dataset['Region'] == region]
    num_countries = len(regional_data['Country'].unique())
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot with trend line
    sns.scatterplot(x='Democracy score', y='H index', hue='Country', data=regional_data, palette=custom_palette, s=100, alpha=0.7)
    sns.regplot(x='Democracy score', y='H index', data=regional_data, scatter=False, color='gray', ci=None, line_kws={"linestyle": "--"})
    
    # Adding titles and labels
    plt.title(f'Relationship between Democracy Score and H Index in {region}', fontsize=20, fontweight='bold')
    plt.xlabel('Democracy Score')
    plt.ylabel('H Index')
    
    # Conditionally adjust the legend to have two columns if more than 20 countries
    if num_countries > 20:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    else:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

# 11 - Time-series for academic and democratic index over the years:
plt.figure(figsize=(14, 10))
# Loop through each metric and create a subplot
for i, metric in enumerate(metrics, 1):
    plt.subplot(len(metrics, ) // 2 + len(metrics, ) % 2, 2, i)  # Create a subplot in a grid with 2 columns
    plt.plot(countries_merged_dataset['Year'], countries_merged_dataset[metric])
    plt.title(f'Time Series of {metric}', fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel(metric)
plt.tight_layout()
plt.show()

# 12 - Regional Disparities in Academic Performance and Democracy Scores
# Aggregate the data by region (taking the mean for simplicity)
df_region = countries_merged_dataset.groupby(['Region', 'Year'])[metrics].mean().reset_index()
df_region_log = df_region.copy()
for metric in metrics:
    df_region_log[metric] = np.log1p(df_region[metric])  # log1p is log(1 + x) to handle zero values

# Plot the parallel coordinates with the log-scaled data
plt.figure(figsize=(14, 8))
parallel_coordinates(df_region_log[['Region'] + metrics], 'Region', color=plt.cm.tab10.colors)
plt.title('Parallel Coordinates Plot of Democracy and Academic Metrics by Region (Log Scale)', fontsize=14, fontweight='bold')
plt.ylabel('Log-Scaled Value')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Perform the ADF test for each index
for metric in metrics:
    result = adfuller(countries_merged_dataset[metric].dropna())
    print(f'ADF Statistic for {metric}: {result[0]}')
    print(f'p-value for {metric}: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    print('-----------------------------------')

# --- Prediction model --- #

# Prophet model to forecast the democracy and academic scores for each country from 2023 to 2030
# Reset index and convert 'Year' to datetime
countries_merged_dataset.reset_index(inplace=True)
countries_merged_dataset['Year'] = pd.to_datetime(countries_merged_dataset['Year'], format='%Y')

# Initialize an empty list to store results
results = []

# Get the list of unique countries
countries = countries_merged_dataset['Country'].unique()

# Loop over each country separately
for country in countries:
    country_data = countries_merged_dataset[countries_merged_dataset['Country'] == country] 
    
    # Loop over each metric separately
    for metric in metrics:
        # Preparing the data for Prophet
        df_prophet = country_data[['Year', metric]].rename(columns={'Year': 'ds', metric: 'y'})
        
        # Initialize Prophet model
        model = Prophet()
        model.fit(df_prophet)
        
        # Create future dates and forecast
        future = model.make_future_dataframe(periods=9, freq='Y')
        future = future[future['ds'].dt.year >= 2023]
        forecast = model.predict(future)
        
        # Set negative forecast values to zero
        forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0, x))
        
        # Add the country name, metric name, and year to the forecast data
        forecast['Country'] = country
        forecast['Metric'] = metric
        forecast['Year'] = forecast['ds'].dt.year
        
        # Select relevant columns and rename them
        future_forecast = forecast[['Country', 'Year', 'Metric', 'yhat']]
        future_forecast = future_forecast.rename(columns={'yhat': metric})
        
        # Apply appropriate formatting
        if metric != 'Democracy score' and metric != 'Citations per document':
            future_forecast[metric] = future_forecast[metric].round().astype(int)
        
        # Append the results to the list
        results.append(future_forecast)

# Concatenate all the results into a single DataFrame
final_forecast = pd.concat(results, ignore_index=True)

# Pivot the DataFrame so each metric has its own column
final_forecast_pivot = final_forecast.pivot_table(index=['Country', 'Year'], columns='Metric', values=metrics).reset_index()

# Flatten the MultiIndex columns after pivot
final_forecast_pivot.columns = [col[0] if col[1] == '' else col[1] for col in final_forecast_pivot.columns]

# Merge the Region information into the forecast
final_forecast_pivot = pd.merge(final_forecast_pivot, countries_merged_dataset[['Country', 'Region']].drop_duplicates(), on='Country', how='left')

# Reorder the columns according to your requested order
final_forecast_pivot = final_forecast_pivot[['Country', 'Region', 'Year', 'Documents', 'Citable documents', 
                                             'Citations', 'Self-citations', 'Citations per document', 
                                             'H index', 'Democracy score']]

# Save the DataFrame to a CSV file
final_forecast_pivot.to_csv('country_future_forecasts.csv', index=False)

# 13 - Plotting forecasts alongside history data

# Combine historical and forecast data
countries_merged_dataset['Year'] = countries_merged_dataset['Year'].apply(lambda x: x.year)

# Ensure all values in the 'Year' column are of type int
final_forecast_pivot['Year'] = final_forecast_pivot['Year'].astype(int)

# Combine historical data with future data (2023 to 2030)
combined_data = pd.concat([countries_merged_dataset, final_forecast_pivot], ignore_index=True)

# Select only numeric columns, excluding 'Year'
numeric_columns = combined_data.select_dtypes(include=['number']).columns
numeric_columns = numeric_columns.drop('Year')  # Exclude 'Year' to avoid duplications

# Calculate averages by region and year
region_forecast = combined_data.groupby(['Region', 'Year'])[numeric_columns].mean().reset_index()

# Create plots with 2 plots per row
fig, axes = plt.subplots(len(metrics) // 2 + len(metrics) % 2, 2, figsize=(14, len(metrics) * 2.5))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Loop over each metric to create a plot
for i, metric in enumerate(metrics):
    ax = axes[i]
    for region in regions:
        region_data = region_forecast[region_forecast['Region'] == region]
        ax.plot(region_data['Year'], region_data[metric], marker='o', label=region)
    
    ax.set_title(f'Time Series of {metric}', fontsize=16, fontweight='bold')
    ax.set_ylabel(metric)
    ax.grid(True)
    
    # Ensure the years are shown on the x-axis
    ax.set_xticks(region_data['Year'].unique())
    ax.set_xticklabels(region_data['Year'].unique(), rotation=45)
    ax.set_xlabel('Year')  # Add label to the x-axis

# Remove any empty subplots if the number of metrics is odd
if len(metrics) % 2 != 0:
    fig.delaxes(axes[-1])

# Add legend to the right side of the last single plot
if len(metrics) % 2 != 0:
    axes[-2].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)

# Adjust the layout of the plots
plt.tight_layout()
plt.show()