# Democracy Index Forecasting 🗳️📊

This project analyzes the relationship between economic indicators and the Democracy Index across various countries over time, and builds a predictive model for forecasting future democracy scores.

## 🧠 Project Overview

We combined real-world data from 2010–2023 on:
- GDP per capita
- Inflation
- Unemployment
- Democracy Index (EIU)

Using this data, we applied statistical analysis and linear regression to identify patterns and predict the 2024 democracy index values for a wide range of countries.

## 📁 Key Files

- `Final_Project.py` – Main Python script that handles data cleaning, visualization, and model building
- `raw_data/` – Contains Excel files with economic data per year
- `democracy-index-eiu.csv` – Source of official democracy index data
- `country_future_forecasts.csv` – Final forecast results for 2024

## 🧪 Technologies Used

- Python
- Pandas
- Matplotlib, Seaborn
- scikit-learn
- Excel (data source)

## ▶️ How to Run

1. Install required packages:
```bash
pip install pandas matplotlib seaborn scikit-learn openpyxl
```

2. Run the script:
```bash
python Final_Project.py
```

## 📈 Output

- Summary of economic-democratic relationships
- Correlation heatmaps and visual plots
- A CSV file with predicted democracy scores for 2024

---

> Final project in Data Science Practicum  
> By: Eliora Stone & Omri Naftali
