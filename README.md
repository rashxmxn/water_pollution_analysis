# 💧 Water Pollution Monitoring Dashboard

Comprehensive web-based dashboard for monitoring water quality across multiple sampling points using Streamlit.

## 📋 Overview

This interactive dashboard provides real-time visualization and analysis of water pollution data from four monitoring points along a water body. The system tracks heavy metal concentrations, water quality classifications, and discharge rates from 2020 to 2023.

### Monitoring Points
- **T1** (formerly Yer 2)
- **T2** (formerly Yer 3)
- **T3** (formerly Yer 4)
- **T4** (formerly Yer 5)

## 🚀 Features

### 1. Overview (Обзор)
- Water quality class visualization by year and location
- Interactive bar charts and heatmaps
- Summary statistics for all monitoring points
- Visual classification system (Class I-V)

### 2. Heavy Metals Analysis (Тяжелые металлы)
- **Metals tracked**: Manganese (Mn), Zinc (Zn), Copper (Cu), Cadmium (Cd)
- Time series charts showing concentration trends
- Distribution analysis with box plots
- Heatmaps of average concentrations
- Statistical summaries (mean, median, min, max, std)
- Interactive filtering by metal type

### 3. Water Discharge (Расход воды)
- Annual average discharge trends
- Year-over-year comparisons
- Detailed monthly data tables
- Statistical metrics and visualizations

### 4. Monitoring Points Comparison (Сравнение точек)
- Radar charts comparing all locations
- Side-by-side metal concentration comparisons
- Water quality class trends
- Comprehensive summary tables

### 5. Trends Analysis (Тренды)
- Long-term trend analysis (2020-2023)
- Year-over-year change calculations
- Combined indicator analysis
- Detailed statistical breakdowns

## 📊 Data Files

The dashboard uses three Excel files:

1. **Данные по ТМ.xlsx** - Heavy metals concentration data
   - Monthly measurements for Mn, Zn, Cu, Cd
   - Data organized by year and monitoring point
   
2. **Индекс.xlsx** - Water quality index
   - Annual water quality classifications (Class I-V)
   - Data for each monitoring point
   
3. **расход.xlsx** - Water discharge data
   - Monthly and annual flow measurements
   - Historical data from 2014 onwards

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the repository**
   ```bash
   cd water_pol
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

1. **Ensure you're in the project directory with data files**
   ```bash
   cd C:\Users\G1415-01\Desktop\Mukhamed\projects\water_pol
   ```

2. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to this URL

## 📦 Dependencies

```
streamlit==1.31.0
pandas==2.2.0
plotly==5.18.0
openpyxl==3.1.2
numpy==1.26.3
```

## 🎨 Design Features

- **Language**: Russian interface text with English chart labels
- **Theme**: Forced white background (works on any system theme)
- **Responsive**: Wide layout optimized for desktop viewing
- **Interactive**: All charts are interactive with hover details, zoom, and pan
- **User-friendly**: Sidebar navigation with clear sections

## 📈 Data Analysis Capabilities

### Statistical Analysis
- Mean, median, min, max, standard deviation
- Year-over-year change calculations
- Trend identification
- Distribution analysis

### Visualizations
- Line charts for time series
- Bar charts for comparisons
- Box plots for distributions
- Heatmaps for spatial-temporal analysis
- Radar charts for multi-dimensional comparisons
- Scatter plots for relationships

## 🔍 Water Quality Classification

| Class | Description (Russian) | Description (English) |
|-------|----------------------|----------------------|
| I | Очень чистая | Very Clean |
| II | Чистая | Clean |
| III | Умеренно загрязненная | Moderately Polluted |
| IV | Загрязненная | Polluted |
| V | Грязная | Dirty |

## 📝 Usage Tips

1. **Navigation**: Use the sidebar to switch between different analysis sections
2. **Filtering**: Select specific metals or years to focus your analysis
3. **Interactivity**: Hover over charts for detailed values
4. **Export**: Most charts have a camera icon for downloading as PNG
5. **Data Tables**: Scroll through tables to see detailed numerical data

## 🐛 Troubleshooting

### Common Issues

**Application won't start:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that Excel files are in the same directory as `app.py`
- Verify Python version: `python --version` (should be 3.8+)

**Data loading errors:**
- Verify Excel file names match exactly: `Данные по ТМ.xlsx`, `Индекс.xlsx`, `расход.xlsx`
- Check file permissions
- Ensure files are not open in Excel

**Charts not displaying:**
- Clear browser cache
- Try a different browser
- Check JavaScript is enabled

## 📁 Project Structure

```
water_pol/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── Данные по ТМ.xlsx        # Heavy metals data
├── Индекс.xlsx              # Water quality index
└── расход.xlsx              # Water discharge data
```

## 🔮 Future Enhancements

Potential improvements for future versions:
- Export functionality for reports
- Predictive modeling for future trends
- Alert system for threshold violations
- Map integration for monitoring point locations
- Multi-language support
- Data upload interface for new measurements
- PDF report generation

## 👥 Contact & Support

For questions about the data or dashboard functionality, please contact the project maintainer.

## 📄 License

This project is for internal use for water quality monitoring purposes.

---

**Last Updated**: February 2026  
**Data Period**: 2020-2023  
**Version**: 1.0.0
