import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Force light theme
st.set_page_config(
    page_title="Мониторинг загрязнения воды",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to force white background
st.markdown("""
    <style>
    .stApp {
        background-color: white;
        color: black;
    }
    .stSidebar {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_heavy_metals_data():
    """Load and process heavy metals data"""
    df = pd.read_excel('Данные по ТМ.xlsx', sheet_name='Sheet1')
    
    # Process the data to create a structured format
    data_list = []
    
    for idx, row in df.iterrows():
        first_col = str(row.iloc[0])
        
        # Check if this is a year-month header
        if '-' in first_col and any(year in first_col for year in ['2020', '2021', '2022', '2023', '2024', '2025']):
            current_period = first_col
            year = int(current_period.split('-')[0])
            month_ru = current_period.split('-')[1]
            
            # Month mapping
            month_map = {
                'Январь': 'January', 'Февраль': 'February', 'Март': 'March',
                'Апрель': 'April', 'Май': 'May', 'Июнь': 'June',
                'Июль': 'July', 'Август': 'August', 'Сентябрь': 'September',
                'Октябрь': 'October', 'Ноябрь': 'November', 'Декабрь': 'December'
            }
            month_en = month_map.get(month_ru, month_ru)
        elif first_col in ['Mn', 'Zn', 'Cu', 'Cd']:
            # This is a data row
            metal = first_col
            
            # Extract values for each year (columns are grouped by year)
            for col_idx, col in enumerate(df.columns[1:]):
                value = row.iloc[col_idx + 1]
                if pd.notna(value):
                    # Determine which T point and which year
                    col_name = str(col)
                    
                    # Parse column name to get T point
                    if 'T1' in col_name:
                        t_point = 'T1'
                    elif 'T2' in col_name:
                        t_point = 'T2'
                    elif 'T3' in col_name:
                        t_point = 'T3'
                    elif 'T4' in col_name:
                        t_point = 'T4'
                    else:
                        continue
                    
                    # Determine year from column name
                    if '2021-' in col_name or (col_idx >= 0 and col_idx < 4):
                        data_year = 2021
                    elif '2022-' in col_name or (col_idx >= 4 and col_idx < 9):
                        data_year = 2022
                    elif '2023-' in col_name or (col_idx >= 9 and col_idx < 14):
                        data_year = 2023
                    elif '2024-' in col_name or (col_idx >= 14 and col_idx < 19):
                        data_year = 2024
                    elif '2025-' in col_name or (col_idx >= 19):
                        data_year = 2025
                    else:
                        continue
                    
                    data_list.append({
                        'Year': data_year,
                        'Month': month_en if 'current_period' in locals() else 'Unknown',
                        'Period': f"{data_year}-{month_en if 'current_period' in locals() else 'Unknown'}",
                        'Metal': metal,
                        'Location': t_point,
                        'Value': float(value)
                    })
    
    return pd.DataFrame(data_list)


@st.cache_data
def load_index_data():
    """Load water quality index data"""
    df = pd.read_excel('Индекс.xlsx')
    
    # Clean the data - keep only the first 5 rows with actual data
    df_clean = df.iloc[:5].copy()
    
    # Rename columns
    df_clean.columns = ['Year', 'T1', 'T2', 'T3', 'T4']
    
    # Convert year to int
    df_clean['Year'] = df_clean['Year'].astype(int)
    
    return df_clean


@st.cache_data
def load_discharge_data():
    """Load water discharge data"""
    df = pd.read_excel('расход.xlsx')
    
    # Extract years from columns (skip first column 'By year')
    years = [col for col in df.columns if col != 'By year']
    
    # First row contains averages
    averages = df.iloc[0, 1:].values
    
    discharge_data = pd.DataFrame({
        'Year': years,
        'Average_Discharge': averages
    })
    
    # Convert year to int and average to float
    discharge_data['Year'] = discharge_data['Year'].astype(int)
    discharge_data['Average_Discharge'] = pd.to_numeric(discharge_data['Average_Discharge'], errors='coerce')
    
    return discharge_data, df


# Load data
try:
    metals_df = load_heavy_metals_data()
    index_df = load_index_data()
    discharge_df, discharge_raw = load_discharge_data()
except Exception as e:
    st.error(f"Ошибка при загрузке данных: {e}")
    st.stop()


# Sidebar
st.sidebar.title("💧 Навигация")
page = st.sidebar.radio(
    "Выберите раздел:",
    ["📊 Обзор", "🔬 Тяжелые металлы", "💦 Расход воды", "📈 Сравнение точек", "📉 Тренды"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **О системе мониторинга:**
    
    Точки мониторинга:
    - **T1** (бывший Yer 2)
    - **T2** (бывший Yer 3)
    - **T3** (бывший Yer 4)
    - **T4** (бывший Yer 5)
    
    Период: 2020-2023
""")


# Main content
if page == "📊 Обзор":
    st.title("📊 Обзор качества воды")
    st.markdown("### Общая информация о состоянии водных ресурсов")
    
    # Water quality classes
    st.markdown("## Классы качества воды по годам")
    
    class_desc = {
        1: "I - Очень чистая",
        2: "II - Чистая",
        3: "III - Умеренно загрязненная",
        4: "IV - Загрязненная",
        5: "V - Грязная"
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Filter for 2020-2023
        index_filtered = index_df[index_df['Year'].between(2020, 2023)]
        
        # Create bar chart
        fig = go.Figure()
        
        for location in ['T1', 'T2', 'T3', 'T4']:
            fig.add_trace(go.Bar(
                name=location,
                x=index_filtered['Year'],
                y=index_filtered[location],
                text=index_filtered[location],
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Water Quality Class by Year and Location",
            xaxis_title="Year",
            yaxis_title="Water Quality Class",
            barmode='group',
            height=400,
            yaxis=dict(range=[0, 6], dtick=1),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Классификация")
        for class_num, desc in class_desc.items():
            st.markdown(f"**{desc}**")
        
        st.markdown("---")
        st.metric("Всего точек мониторинга", "4")
        st.metric("Период наблюдений", "2020-2023")
    
    # Heatmap of quality classes
    st.markdown("### Тепловая карта качества воды")
    
    index_pivot = index_filtered.set_index('Year')[['T1', 'T2', 'T3', 'T4']]
    
    fig = px.imshow(
        index_pivot.T,
        labels=dict(x="Year", y="Location", color="Class"),
        x=index_pivot.index,
        y=['T1', 'T2', 'T3', 'T4'],
        color_continuous_scale='RdYlGn_r',
        aspect="auto",
        title="Water Quality Class Heatmap"
    )
    
    fig.update_layout(
        height=300,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("### Статистика по точкам мониторинга (2020-2023)")
    
    cols = st.columns(4)
    
    for idx, location in enumerate(['T1', 'T2', 'T3', 'T4']):
        with cols[idx]:
            avg_class = index_filtered[location].mean()
            current_class = index_filtered[index_filtered['Year'] == 2023][location].values[0]
            
            st.metric(
                label=f"📍 {location}",
                value=f"Класс {int(current_class)}",
                delta=f"Средний: {avg_class:.1f}"
            )


elif page == "🔬 Тяжелые металлы":
    st.title("🔬 Анализ тяжелых металлов")
    st.markdown("### Концентрация тяжелых металлов в воде (мг/л)")
    
    # Filter for 2020-2023
    metals_filtered = metals_df[metals_df['Year'].between(2020, 2023)]
    
    # Metal selector
    selected_metal = st.selectbox(
        "Выберите металл:",
        ['All', 'Mn', 'Zn', 'Cu', 'Cd'],
        format_func=lambda x: {
            'All': 'Все металлы',
            'Mn': 'Марганец (Mn)',
            'Zn': 'Цинк (Zn)',
            'Cu': 'Медь (Cu)',
            'Cd': 'Кадмий (Cd)'
        }[x]
    )
    
    # Time series chart
    st.markdown("## Временные ряды концентраций")
    
    if selected_metal == 'All':
        # Create subplots for all metals
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Manganese (Mn)', 'Zinc (Zn)', 'Copper (Cu)', 'Cadmium (Cd)')
        )
        
        metals = ['Mn', 'Zn', 'Cu', 'Cd']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metal, (row, col) in zip(metals, positions):
            metal_data = metals_filtered[metals_filtered['Metal'] == metal]
            
            for location in ['T1', 'T2', 'T3', 'T4']:
                loc_data = metal_data[metal_data['Location'] == location]
                
                fig.add_trace(
                    go.Scatter(
                        x=loc_data['Period'],
                        y=loc_data['Value'],
                        name=location,
                        mode='lines+markers',
                        showlegend=(row == 1 and col == 1)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=700,
            title_text="Heavy Metals Concentration Over Time (mg/L)",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Single metal detailed view
        metal_data = metals_filtered[metals_filtered['Metal'] == selected_metal]
        
        fig = go.Figure()
        
        for location in ['T1', 'T2', 'T3', 'T4']:
            loc_data = metal_data[metal_data['Location'] == location]
            
            fig.add_trace(go.Scatter(
                x=loc_data['Period'],
                y=loc_data['Value'],
                name=location,
                mode='lines+markers',
                line=dict(width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=f"{selected_metal} Concentration Over Time",
            xaxis_title="Period",
            yaxis_title="Concentration (mg/L)",
            height=500,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Box plots for distribution
    st.markdown("## Распределение концентраций металлов")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot by location
        if selected_metal == 'All':
            plot_data = metals_filtered
        else:
            plot_data = metals_filtered[metals_filtered['Metal'] == selected_metal]
        
        fig = px.box(
            plot_data,
            x='Location',
            y='Value',
            color='Location',
            title="Distribution by Location",
            labels={'Value': 'Concentration (mg/L)', 'Location': 'Monitoring Point'}
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot by year
        fig = px.box(
            plot_data,
            x='Year',
            y='Value',
            color='Location',
            title="Distribution by Year",
            labels={'Value': 'Concentration (mg/L)', 'Year': 'Year'}
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown("## Статистическая сводка")
    
    if selected_metal == 'All':
        summary_data = metals_filtered.groupby(['Location', 'Metal'])['Value'].agg([
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Std', 'std')
        ]).round(4)
    else:
        summary_data = metals_filtered[metals_filtered['Metal'] == selected_metal].groupby('Location')['Value'].agg([
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Std', 'std')
        ]).round(4)
    
    st.dataframe(summary_data, use_container_width=True)
    
    # Heatmap of average concentrations
    st.markdown("## Средние концентрации по металлам и точкам")
    
    heatmap_data = metals_filtered.groupby(['Metal', 'Location'])['Value'].mean().unstack()
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Location", y="Metal", color="Concentration"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='Reds',
        aspect="auto",
        title="Average Metal Concentration Heatmap (mg/L)"
    )
    
    fig.update_layout(
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


elif page == "💦 Расход воды":
    st.title("💦 Анализ расхода воды")
    st.markdown("### Средний годовой расход воды (м³/с)")
    
    # Filter for 2020-2023
    discharge_filtered = discharge_df[discharge_df['Year'].between(2020, 2023)]
    
    # Line chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=discharge_filtered['Year'],
            y=discharge_filtered['Average_Discharge'],
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=12),
            name='Average Discharge'
        ))
        
        fig.update_layout(
            title="Average Annual Water Discharge",
            xaxis_title="Year",
            yaxis_title="Discharge (m³/s)",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Статистика")
        
        avg_discharge = discharge_filtered['Average_Discharge'].mean()
        max_discharge = discharge_filtered['Average_Discharge'].max()
        min_discharge = discharge_filtered['Average_Discharge'].min()
        
        st.metric("Средний расход", f"{avg_discharge:.2f} m³/s")
        st.metric("Максимум", f"{max_discharge:.2f} m³/s")
        st.metric("Минимум", f"{min_discharge:.2f} m³/s")
    
    # Bar chart comparison
    st.markdown("### Сравнение по годам")
    
    fig = px.bar(
        discharge_filtered,
        x='Year',
        y='Average_Discharge',
        title="Annual Water Discharge Comparison",
        labels={'Average_Discharge': 'Discharge (m³/s)', 'Year': 'Year'},
        color='Average_Discharge',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed monthly data table
    st.markdown("### Детальные данные по годам")
    
    # Show the raw data for 2020-2023
    years_to_show = [2020, 2021, 2022, 2023]
    detailed_data = discharge_raw[['By year'] + years_to_show]
    detailed_data.columns = ['Month/Day'] + [str(year) for year in years_to_show]
    
    st.dataframe(detailed_data.head(35), use_container_width=True, height=400)


elif page == "📈 Сравнение точек":
    st.title("📈 Сравнение точек мониторинга")
    st.markdown("### Сопоставление показателей между T1, T2, T3, T4")
    
    # Filter for 2020-2023
    metals_filtered = metals_df[metals_df['Year'].between(2020, 2023)]
    index_filtered = index_df[index_df['Year'].between(2020, 2023)]
    
    # Radar chart for average metal concentrations
    st.markdown("## Средние концентрации металлов по точкам")
    
    avg_by_location = metals_filtered.groupby(['Location', 'Metal'])['Value'].mean().unstack()
    
    fig = go.Figure()
    
    for location in ['T1', 'T2', 'T3', 'T4']:
        fig.add_trace(go.Scatterpolar(
            r=avg_by_location.loc[location].values,
            theta=avg_by_location.columns,
            fill='toself',
            name=location
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        showlegend=True,
        title="Average Metal Concentrations - Radar Chart",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Grouped bar chart for metals
    st.markdown("## Сравнение концентраций металлов")
    
    selected_year = st.selectbox(
        "Выберите год:",
        [2020, 2021, 2022, 2023]
    )
    
    year_data = metals_filtered[metals_filtered['Year'] == selected_year]
    avg_by_metal_location = year_data.groupby(['Metal', 'Location'])['Value'].mean().reset_index()
    
    fig = px.bar(
        avg_by_metal_location,
        x='Metal',
        y='Value',
        color='Location',
        barmode='group',
        title=f"Metal Concentrations Comparison - {selected_year}",
        labels={'Value': 'Concentration (mg/L)', 'Metal': 'Heavy Metal'}
    )
    
    fig.update_layout(
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Water quality class comparison
    st.markdown("## Классы качества воды")
    
    fig = go.Figure()
    
    for location in ['T1', 'T2', 'T3', 'T4']:
        fig.add_trace(go.Scatter(
            x=index_filtered['Year'],
            y=index_filtered[location],
            name=location,
            mode='lines+markers',
            line=dict(width=3),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title="Water Quality Class Trends by Location",
        xaxis_title="Year",
        yaxis_title="Water Quality Class",
        height=400,
        yaxis=dict(range=[0, 6], dtick=1),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Table comparison
    st.markdown("## Сводная таблица по точкам")
    
    summary_by_location = metals_filtered.groupby('Location').agg({
        'Value': ['mean', 'median', 'min', 'max']
    }).round(4)
    
    summary_by_location.columns = ['Mean Concentration', 'Median', 'Min', 'Max']
    
    # Add water quality info
    latest_year = index_filtered[index_filtered['Year'] == 2023]
    quality_classes = latest_year.set_index('Year')[['T1', 'T2', 'T3', 'T4']].T
    quality_classes.columns = ['Quality Class 2023']
    quality_classes.index.name = 'Location'
    
    combined_summary = pd.concat([summary_by_location, quality_classes], axis=1)
    
    st.dataframe(combined_summary, use_container_width=True)


elif page == "📉 Тренды":
    st.title("📉 Анализ трендов")
    st.markdown("### Долгосрочные тренды загрязнения воды (2020-2023)")
    
    # Filter for 2020-2023
    metals_filtered = metals_df[metals_df['Year'].between(2020, 2023)]
    index_filtered = index_df[index_df['Year'].between(2020, 2023)]
    discharge_filtered = discharge_df[discharge_df['Year'].between(2020, 2023)]
    
    # Overall trend analysis
    st.markdown("## Общие тренды по металлам")
    
    yearly_avg = metals_filtered.groupby(['Year', 'Metal'])['Value'].mean().reset_index()
    
    fig = px.line(
        yearly_avg,
        x='Year',
        y='Value',
        color='Metal',
        markers=True,
        title="Average Metal Concentration Trends (2020-2023)",
        labels={'Value': 'Average Concentration (mg/L)', 'Year': 'Year'}
    )
    
    fig.update_layout(
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Location-specific trends
    st.markdown("## Тренды по точкам мониторинга")
    
    selected_location = st.selectbox(
        "Выберите точку:",
        ['T1', 'T2', 'T3', 'T4']
    )
    
    location_data = metals_filtered[metals_filtered['Location'] == selected_location]
    yearly_location = location_data.groupby(['Year', 'Metal'])['Value'].mean().reset_index()
    
    fig = px.line(
        yearly_location,
        x='Year',
        y='Value',
        color='Metal',
        markers=True,
        title=f"Metal Concentration Trends - {selected_location}",
        labels={'Value': 'Average Concentration (mg/L)', 'Year': 'Year'}
    )
    
    fig.update_layout(
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Combined indicators
    st.markdown("## Комплексный анализ показателей")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Water quality trend
        fig = go.Figure()
        
        for location in ['T1', 'T2', 'T3', 'T4']:
            fig.add_trace(go.Scatter(
                x=index_filtered['Year'],
                y=index_filtered[location],
                name=location,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="Water Quality Class Trends",
            xaxis_title="Year",
            yaxis_title="Quality Class",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Discharge trend
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=discharge_filtered['Year'],
            y=discharge_filtered['Average_Discharge'],
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=12),
            name='Water Discharge'
        ))
        
        fig.update_layout(
            title="Water Discharge Trend",
            xaxis_title="Year",
            yaxis_title="Discharge (m³/s)",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Trend statistics
    st.markdown("## Статистика изменений")
    
    # Calculate year-over-year changes
    metals_2020 = metals_filtered[metals_filtered['Year'] == 2020]['Value'].mean()
    metals_2023 = metals_filtered[metals_filtered['Year'] == 2023]['Value'].mean()
    metals_change = ((metals_2023 - metals_2020) / metals_2020) * 100
    
    discharge_2020 = discharge_filtered[discharge_filtered['Year'] == 2020]['Average_Discharge'].values[0]
    discharge_2023 = discharge_filtered[discharge_filtered['Year'] == 2023]['Average_Discharge'].values[0]
    discharge_change = ((discharge_2023 - discharge_2020) / discharge_2020) * 100
    
    cols = st.columns(3)
    
    with cols[0]:
        st.metric(
            label="Изменение концентрации металлов",
            value=f"{metals_2023:.4f} мг/л",
            delta=f"{metals_change:+.1f}% с 2020"
        )
    
    with cols[1]:
        st.metric(
            label="Изменение расхода воды",
            value=f"{discharge_2023:.2f} м³/с",
            delta=f"{discharge_change:+.1f}% с 2020"
        )
    
    with cols[2]:
        avg_quality_2020 = index_filtered[index_filtered['Year'] == 2020][['T1', 'T2', 'T3', 'T4']].mean().mean()
        avg_quality_2023 = index_filtered[index_filtered['Year'] == 2023][['T1', 'T2', 'T3', 'T4']].mean().mean()
        quality_change = avg_quality_2023 - avg_quality_2020
        
        st.metric(
            label="Изменение класса качества",
            value=f"{avg_quality_2023:.1f}",
            delta=f"{quality_change:+.1f} с 2020"
        )
    
    # Detailed statistics table
    st.markdown("### Детальная статистика по годам")
    
    detailed_stats = metals_filtered.groupby('Year')['Value'].agg([
        ('Count', 'count'),
        ('Mean', 'mean'),
        ('Median', 'median'),
        ('Std', 'std'),
        ('Min', 'min'),
        ('Max', 'max')
    ]).round(4)
    
    st.dataframe(detailed_stats, use_container_width=True)


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Система мониторинга качества воды | 2020-2023</p>
        <p>Данные обновлены: Февраль 2026</p>
    </div>
    """, unsafe_allow_html=True)
