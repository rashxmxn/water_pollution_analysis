import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(
    page_title="Мониторинг загрязнения воды",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    df = pd.read_excel('data/Данные по ТМ.xlsx', sheet_name='Sheet1')
    
    data_list = []
    
    for idx, row in df.iterrows():
        first_col = str(row.iloc[0])
        
        if '-' in first_col and any(year in first_col for year in ['2020', '2021', '2022', '2023', '2024', '2025']):
            current_period = first_col
            year = int(current_period.split('-')[0])
            month_ru = current_period.split('-')[1]
            
            month_map = {
                'Январь': 'January', 'Февраль': 'February', 'Март': 'March',
                'Апрель': 'April', 'Май': 'May', 'Июнь': 'June',
                'Июль': 'July', 'Август': 'August', 'Сентябрь': 'September',
                'Октябрь': 'October', 'Ноябрь': 'November', 'Декабрь': 'December'
            }
            month_en = month_map.get(month_ru, month_ru)
        elif first_col in ['Mn', 'Zn', 'Cu', 'Cd']:
            metal = first_col
            for col_idx, col in enumerate(df.columns[1:]):
                value = row.iloc[col_idx + 1]
                if pd.notna(value):
                    col_name = str(col)
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

                    # Fallback to column positions when year headers are inconsistent.
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
    df = pd.read_excel('data/Индекс.xlsx')
    
    df_clean = df.iloc[:5].copy()
    df_clean.columns = ['Year', 'T1', 'T2', 'T3', 'T4']
    df_clean['Year'] = df_clean['Year'].astype(int)
    
    return df_clean


@st.cache_data
def load_discharge_data():
    """Load water discharge data"""
    df = pd.read_excel('data/расход.xlsx')
    
    # First row contains averages; use the remaining rows as observations.
    df_data = df.iloc[1:].copy()
    years = [col for col in df.columns if col != 'By year']
    averages = []
    for year in years:
        year_data = pd.to_numeric(df_data[year], errors='coerce')
        avg = year_data.mean()
        averages.append(avg)
    
    discharge_data = pd.DataFrame({
        'Year': years,
        'Average_Discharge': averages
    })
    
    discharge_data['Year'] = discharge_data['Year'].astype(int)
    
    return discharge_data, df_data


try:
    metals_df = load_heavy_metals_data()
    index_df = load_index_data()
    discharge_df, discharge_raw = load_discharge_data()
except Exception as e:
    st.error(f"Ошибка при загрузке данных: {e}")
    st.stop()


st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите раздел:",
    ["Обзор", "Тяжелые металлы", "Расход воды", "Сравнение точек", "Тренды", "Выводы"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **О системе мониторинга:**
    
    Точки мониторинга:
    - **T1** (бывший Yer 2)
    - **T2** (бывший Yer 3)
    - **T3** (бывший Yer 4)
    - **T4** (бывший Yer 5)

""")


if page == "Обзор":
    st.title("Обзор качества воды")
    st.markdown("### Панель управления мониторингом водных ресурсов")
    
    index_filtered = index_df[index_df['Year'].between(2020, 2023)]
    metals_filtered = metals_df[metals_df['Year'].between(2020, 2023)]
    discharge_filtered = discharge_df[discharge_df['Year'].between(2020, 2023)]
    
    st.markdown("## Ключевые показатели")
    
    data_2020 = index_filtered[index_filtered['Year'] == 2020]
    data_2023 = index_filtered[index_filtered['Year'] == 2023]
    
    if not data_2020.empty and not data_2023.empty:
        quality_2020 = data_2020[['T1', 'T2', 'T3', 'T4']].mean().mean()
        quality_2023 = data_2023[['T1', 'T2', 'T3', 'T4']].mean().mean()
        quality_change = quality_2023 - quality_2020
    else:
        quality_2023 = index_filtered[['T1', 'T2', 'T3', 'T4']].mean().mean()
        quality_change = 0
    
    metals_2023 = metals_filtered[metals_filtered['Year'] == 2023]
    avg_metals_2023 = metals_2023['Value'].mean() if not metals_2023.empty else metals_filtered['Value'].mean()
    
    location_avg_quality = {loc: index_filtered[loc].mean() for loc in ['T1', 'T2', 'T3', 'T4']}
    best_location = min(location_avg_quality, key=location_avg_quality.get)
    worst_location = max(location_avg_quality, key=location_avg_quality.get)
    
    total_measurements = len(metals_filtered)
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric(
            label="Средний класс качества 2023",
            value=f"{quality_2023:.1f}",
            delta=f"{quality_change:+.1f} с 2020" if quality_change != 0 else None,
            delta_color="inverse"
        )
    
    with cols[1]:
        st.metric(
            label="Лучшая точка мониторинга",
            value=best_location,
            delta=f"Класс {location_avg_quality[best_location]:.1f}"
        )
    
    with cols[2]:
        st.metric(
            label="Средняя концентрация металлов",
            value=f"{avg_metals_2023:.4f} мг/л",
            delta="2023"
        )
    
    with cols[3]:
        st.metric(
            label="Всего измерений",
            value=f"{total_measurements}",
            delta="2020-2023"
        )
    
    st.markdown("---")
    
    st.markdown("## Точки мониторинга")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        location_summary = []
        for loc in ['T1', 'T2', 'T3', 'T4']:
            avg_quality = location_avg_quality[loc]
            
            if not data_2023.empty:
                current_quality = data_2023[loc].values[0]
            else:
                current_quality = avg_quality
            
            loc_metals = metals_filtered[metals_filtered['Location'] == loc]
            avg_metals = loc_metals['Value'].mean() if not loc_metals.empty else 0
            
            if avg_quality <= 2:
                status = "Хорошо"
            elif avg_quality <= 3:
                status = "Умеренно"
            else:
                status = "Требует внимания"
            
            location_summary.append({
                'Статус': status,
                'Точка': loc,
                'Текущий класс (2023)': int(current_quality),
                'Средний класс': f"{avg_quality:.1f}",
                'Ср. концентрация металлов': f"{avg_metals:.4f}"
            })
        
        summary_df = pd.DataFrame(location_summary)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with col2:
        fig = go.Figure()
        
        colors_map = {'T1': '#3498db', 'T2': '#2ecc71', 'T3': '#e74c3c', 'T4': '#f39c12'}
        
        for location in ['T1', 'T2', 'T3', 'T4']:
            avg_val = location_avg_quality[location]
            fig.add_trace(go.Bar(
                x=[location],
                y=[avg_val],
                name=location,
                marker_color=colors_map[location],
                text=f"{avg_val:.1f}",
                textposition='auto',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Средний класс качества по точкам (2020-2023)",
            yaxis_title="Класс качества",
            height=300,
            yaxis=dict(range=[0, 5]),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("## Динамика изменений по годам")
    
    class_desc = {
        1: "I - Очень чистая",
        2: "II - Чистая",
        3: "III - Умеренно загрязненная",
        4: "IV - Загрязненная",
        5: "V - Грязная"
    }
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = go.Figure()
        
        for location in ['T1', 'T2', 'T3', 'T4']:
            fig.add_trace(go.Scatter(
                name=location,
                x=index_filtered['Year'],
                y=index_filtered[location],
                mode='lines+markers',
                line=dict(width=3),
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title="Тренды класса качества воды",
            xaxis_title="Год",
            yaxis_title="Класс качества",
            height=400,
            yaxis=dict(range=[0, 6], dtick=1),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Классификация")
        for class_num, desc in class_desc.items():
            st.markdown(f"**{class_num}.** {desc}")
        
        st.markdown("---")
        st.markdown("### Информация")
        st.markdown(f"**Период:** 2020-2023")
        st.markdown(f"**Точек:** 4")
        st.markdown(f"**Металлов:** 4 (Mn, Zn, Cu, Cd)")
    
    st.markdown("## Тепловая карта качества воды")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        index_pivot = index_filtered.set_index('Year')[['T1', 'T2', 'T3', 'T4']]
        
        fig = px.imshow(
            index_pivot.T,
            labels=dict(x="Год", y="Точка мониторинга", color="Класс"),
            x=index_pivot.index,
            y=['T1', 'T2', 'T3', 'T4'],
            color_continuous_scale='RdYlGn_r',
            aspect="auto",
            title="Карта качества воды по годам и точкам"
        )
        
        fig.update_layout(
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Интерпретация")
        st.markdown("""
        **Цветовая шкала:**
        - **Зеленый**: Хорошее качество (1-2)
        - **Желтый**: Умеренное (3)
        - **Оранжевый**: Загрязненное (4)
        - **Красный**: Сильно загрязненное (5)
        
        Темные области указывают на ухудшение качества воды.
        """)
    
    st.markdown("## Критические наблюдения")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        worst_year_location = []
        for loc in ['T1', 'T2', 'T3', 'T4']:
            for _, row in index_filtered.iterrows():
                if row[loc] >= 4:
                    worst_year_location.append((row['Year'], loc, row[loc]))
        
        if worst_year_location:
            st.warning(f"**Обнаружено загрязнений класса IV+:** {len(worst_year_location)}")
            for year, loc, cls in worst_year_location[:3]:
                st.markdown(f"- {loc} ({year}): Класс {int(cls)}")
        else:
            st.success("Критических загрязнений не обнаружено")
    
    with col2:
        high_metals = metals_filtered[metals_filtered['Value'] > metals_filtered['Value'].quantile(0.9)]
        n_high = len(high_metals)
        
        if n_high > 0:
            st.info(f"**Повышенные концентрации металлов:** {n_high} случаев")
            top_metal = high_metals.groupby('Metal')['Value'].count().idxmax()
            st.markdown(f"- Чаще всего: **{top_metal}**")
        else:
            st.success("Концентрации в норме")
    
    with col3:
        improving_locations = []
        worsening_locations = []
        
        if not data_2020.empty and not data_2023.empty:
            for loc in ['T1', 'T2', 'T3', 'T4']:
                q_2020 = data_2020[loc].values[0]
                q_2023 = data_2023[loc].values[0]
                change = q_2023 - q_2020
                
                if change < -0.5:
                    improving_locations.append(loc)
                elif change > 0.5:
                    worsening_locations.append(loc)
        
        if improving_locations:
            st.success(f"**Улучшение:** {', '.join(improving_locations)}")
        if worsening_locations:
            st.error(f"**Ухудшение:** {', '.join(worsening_locations)}")
        if not improving_locations and not worsening_locations:
            st.info("**Стабильное состояние**")
    
    st.markdown("## Сравнение периодов")
    
    comparison_data = []
    available_years = sorted(index_filtered['Year'].unique())
    
    if len(available_years) >= 2:
        first_year = available_years[0]
        last_year = available_years[-1]
        
        data_first = index_filtered[index_filtered['Year'] == first_year]
        data_last = index_filtered[index_filtered['Year'] == last_year]
        
        if not data_first.empty and not data_last.empty:
            for loc in ['T1', 'T2', 'T3', 'T4']:
                q_first = data_first[loc].values[0]
                q_last = data_last[loc].values[0]
                
                comparison_data.append({
                    'Точка': loc,
                    f'Класс {first_year}': int(q_first),
                    f'Класс {last_year}': int(q_last),
                    'Изменение': int(q_last) - int(q_first),
                    'Тренд': 'Улучшение' if q_last < q_first else 'Ухудшение' if q_last > q_first else 'Стабильно'
                })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name=str(first_year),
                x=comparison_df['Точка'],
                y=comparison_df[f'Класс {first_year}'],
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name=str(last_year),
                x=comparison_df['Точка'],
                y=comparison_df[f'Класс {last_year}'],
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title=f"Сравнение качества воды: {first_year} vs {last_year}",
                yaxis_title="Класс качества",
                barmode='group',
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    else:
        st.info("Недостаточно данных для сравнения периодов.")


elif page == "Тяжелые металлы":
    st.title("Анализ тяжелых металлов")
    st.markdown("### Концентрация тяжелых металлов в воде (мг/л)")
    
    metals_filtered = metals_df[metals_df['Year'].between(2020, 2023)]
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
    
    st.markdown("## Временные ряды концентраций")
    
    if selected_metal == 'All':
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
    
    st.markdown("## Распределение концентраций металлов")
    
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    st.markdown("## 3D визуализация концентраций")
    selected_location_3d = st.selectbox(
        "Выберите точку мониторинга для 3D визуализации:",
        ['T1', 'T2', 'T3', 'T4']
    )
    metals_avg = metals_filtered.groupby(['Year', 'Metal', 'Location'])['Value'].mean().reset_index()
    metal_map = {'Mn': 1, 'Zn': 2, 'Cu': 3, 'Cd': 4}
    metals_avg['Metal_Numeric'] = metals_avg['Metal'].map(metal_map)
    loc_data = metals_avg[metals_avg['Location'] == selected_location_3d]
    
    colors = {'T1': 'blue', 'T2': 'green', 'T3': 'red', 'T4': 'orange'}
    
    fig = go.Figure()
    
    for metal in ['Mn', 'Zn', 'Cu', 'Cd']:
        metal_data = loc_data[loc_data['Metal'] == metal].sort_values('Year')
        
        fig.add_trace(go.Scatter3d(
            x=metal_data['Year'],
            y=metal_data['Metal_Numeric'],
            z=metal_data['Value'],
            mode='lines+markers',
            name=metal,
            marker=dict(
                size=10,
                color=colors[selected_location_3d],
                opacity=0.9,
                line=dict(color='white', width=2)
            ),
            line=dict(width=4, color=colors[selected_location_3d]),
            text=[f"{m}<br>Year: {y}<br>Avg: {v:.4f} mg/L" 
                  for m, y, v in zip(metal_data['Metal'], metal_data['Year'], metal_data['Value'])],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    
    fig.update_layout(
        title=f"3D Visualization: Average Metal Concentrations - {selected_location_3d}",
        scene=dict(
            xaxis=dict(title='Year', gridcolor='lightgray', dtick=1),
            yaxis=dict(
                title='Metal Type',
                tickmode='array',
                tickvals=[1, 2, 3, 4],
                ticktext=['Mn', 'Zn', 'Cu', 'Cd'],
                gridcolor='lightgray'
            ),
            zaxis=dict(title='Average Concentration (mg/L)', gridcolor='lightgray'),
            bgcolor='white'
        ),
        height=600,
        showlegend=True,
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


elif page == "Расход воды":
    st.title("Анализ расхода воды")
    st.markdown("### Средний годовой расход воды (м³/с)")
    
    discharge_filtered = discharge_df[discharge_df['Year'] >= 2014]
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
    
    st.markdown("### 3D визуализация расхода воды")
    
    monthly_data_list = []
    all_years = [col for col in discharge_raw.columns if isinstance(col, int) and col >= 2014]
    
    for idx, row in discharge_raw.iterrows():
        for year in all_years:
            value = row[year]
            if pd.notna(value):
                if isinstance(value, str):
                    value = value.replace(',', '.')
                try:
                    discharge_value = float(value)
                    monthly_data_list.append({
                        'Year': year,
                        'Month': idx + 1,
                        'Discharge': discharge_value
                    })
                except (ValueError, TypeError):
                    continue
    
    monthly_discharge_df = pd.DataFrame(monthly_data_list)
    
    if not monthly_discharge_df.empty:
        pivot_data = monthly_discharge_df.pivot(index='Month', columns='Year', values='Discharge')
        
        fig = go.Figure(data=[go.Surface(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Blues',
            colorbar=dict(title="m³/s")
        )])
        
        fig.update_layout(
            title="3D Surface: Discharge Patterns",
            scene=dict(
                xaxis=dict(title='Year'),
                yaxis=dict(title='Month', dtick=1),
                zaxis=dict(title='Discharge (m³/s)'),
                bgcolor='white'
            ),
            height=600,
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Тепловая карта месячных значений")
    
    if not monthly_discharge_df.empty:
        pivot_heatmap = monthly_discharge_df.pivot(index='Month', columns='Year', values='Discharge')
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = px.imshow(
            pivot_heatmap,
            labels=dict(x="Year", y="Month", color="Discharge (m³/s)"),
            x=pivot_heatmap.columns,
            y=[month_names[int(m)-1] if int(m) <= 12 else str(int(m)) for m in pivot_heatmap.index],
            color_continuous_scale='RdYlBu_r',
            aspect="auto",
            title="Monthly Discharge Heatmap"
        )
        
        fig.update_layout(
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Сезонный анализ")
    
    if not monthly_discharge_df.empty:
        season_map = {
            1: 'Winter', 2: 'Winter', 3: 'Spring',
            4: 'Spring', 5: 'Spring', 6: 'Summer',
            7: 'Summer', 8: 'Summer', 9: 'Autumn',
            10: 'Autumn', 11: 'Autumn', 12: 'Winter'
        }
        
        monthly_discharge_df['Season'] = monthly_discharge_df['Month'].map(season_map)
        seasonal_data = monthly_discharge_df.groupby(['Year', 'Season'])['Discharge'].mean().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                seasonal_data,
                x='Year',
                y='Discharge',
                color='Season',
                markers=True,
                title="Seasonal Average Discharge Trends",
                labels={'Discharge': 'Discharge (m³/s)', 'Year': 'Year'},
                color_discrete_map={
                    'Winter': '#3498db',
                    'Spring': '#2ecc71',
                    'Summer': '#f39c12',
                    'Autumn': '#e74c3c'
                }
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            season_avg = monthly_discharge_df.groupby('Season')['Discharge'].mean().reset_index()
            season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
            season_avg['Season'] = pd.Categorical(season_avg['Season'], categories=season_order, ordered=True)
            season_avg = season_avg.sort_values('Season')
            
            fig = px.bar(
                season_avg,
                x='Season',
                y='Discharge',
                title="Average Discharge by Season",
                labels={'Discharge': 'Discharge (m³/s)'},
                color='Discharge',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Распределение и вариабельность")
    
    if not monthly_discharge_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                monthly_discharge_df,
                x='Year',
                y='Discharge',
                title="Discharge Distribution by Year",
                labels={'Discharge': 'Discharge (m³/s)', 'Year': 'Year'}
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.violin(
                monthly_discharge_df,
                y='Discharge',
                x='Year',
                title="Discharge Variability (Violin Plot)",
                labels={'Discharge': 'Discharge (m³/s)', 'Year': 'Year'},
                color='Year',
                color_discrete_sequence=px.colors.sequential.Blues
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Диапазон изменчивости и тренд")
    
    if not monthly_discharge_df.empty:
        yearly_stats = monthly_discharge_df.groupby('Year')['Discharge'].agg(['mean', 'min', 'max', 'std']).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=yearly_stats['Year'],
            y=yearly_stats['max'],
            fill=None,
            mode='lines',
            line=dict(color='rgba(52, 152, 219, 0.3)'),
            name='Maximum',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=yearly_stats['Year'],
            y=yearly_stats['min'],
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(52, 152, 219, 0.3)'),
            name='Minimum',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=yearly_stats['Year'],
            y=yearly_stats['mean'],
            mode='lines+markers',
            line=dict(color='rgb(41, 128, 185)', width=3),
            marker=dict(size=10),
            name='Average',
            showlegend=True
        ))
        
        z = np.polyfit(yearly_stats['Year'], yearly_stats['mean'], 2)
        p = np.poly1d(z)
        trend_line = p(yearly_stats['Year'])
        
        fig.add_trace(go.Scatter(
            x=yearly_stats['Year'],
            y=trend_line,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Trend (Polynomial)',
            showlegend=True
        ))
        
        fig.update_layout(
            title="Discharge Range and Trend Analysis",
            xaxis_title="Year",
            yaxis_title="Discharge (m³/s)",
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Анализ аномалий")
    
    if not monthly_discharge_df.empty:
        overall_mean = monthly_discharge_df['Discharge'].mean()
        overall_std = monthly_discharge_df['Discharge'].std()
        
        monthly_discharge_df['Anomaly'] = (monthly_discharge_df['Discharge'] - overall_mean) / overall_std
        monthly_discharge_df['Is_Anomaly'] = monthly_discharge_df['Anomaly'].abs() > 2
        
        fig = go.Figure()
        
        normal_data = monthly_discharge_df[~monthly_discharge_df['Is_Anomaly']]
        anomaly_data = monthly_discharge_df[monthly_discharge_df['Is_Anomaly']]
        
        fig.add_trace(go.Scatter(
            x=normal_data['Year'] + (normal_data['Month'] - 1) / 12,
            y=normal_data['Discharge'],
            mode='markers',
            marker=dict(size=6, color='lightblue'),
            name='Normal',
            text=[f"Year: {y}<br>Month: {m}<br>Discharge: {d:.2f} m³/s" 
                  for y, m, d in zip(normal_data['Year'], normal_data['Month'], normal_data['Discharge'])],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        if not anomaly_data.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_data['Year'] + (anomaly_data['Month'] - 1) / 12,
                y=anomaly_data['Discharge'],
                mode='markers',
                marker=dict(size=12, color='red', symbol='x'),
                name='Anomaly (>2σ)',
                text=[f"Year: {y}<br>Month: {m}<br>Discharge: {d:.2f} m³/s<br>Anomaly Score: {a:.2f}σ" 
                      for y, m, d, a in zip(anomaly_data['Year'], anomaly_data['Month'], 
                                            anomaly_data['Discharge'], anomaly_data['Anomaly'])],
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
        
        fig.add_hline(y=overall_mean, line_dash="dash", line_color="green", 
                      annotation_text=f"Mean: {overall_mean:.2f}")
        fig.add_hline(y=overall_mean + 2*overall_std, line_dash="dot", line_color="orange",
                      annotation_text="+2σ")
        fig.add_hline(y=overall_mean - 2*overall_std, line_dash="dot", line_color="orange",
                      annotation_text="-2σ")
        
        fig.update_layout(
            title="Discharge Anomaly Detection",
            xaxis_title="Year",
            yaxis_title="Discharge (m³/s)",
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Средний расход", f"{overall_mean:.2f} m³/s")
        with col2:
            st.metric("Стандартное отклонение", f"{overall_std:.2f} m³/s")
        with col3:
            n_anomalies = anomaly_data.shape[0]
            st.metric("Количество аномалий", f"{n_anomalies}")
    
    st.markdown("### Детальные данные по годам")
    all_years = [col for col in discharge_raw.columns if isinstance(col, int) and col >= 2014]
    detailed_data = discharge_raw[all_years].copy()
    detailed_data.columns = [str(year) for year in all_years]
    
    st.dataframe(detailed_data, use_container_width=True, height=400)


elif page == "Сравнение точек":
    st.title("Сравнение точек мониторинга")
    st.markdown("### Сопоставление показателей между T1, T2, T3, T4")
    
    metals_filtered = metals_df[metals_df['Year'].between(2020, 2023)]
    index_filtered = index_df[index_df['Year'].between(2020, 2023)]
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
    
    st.markdown("## Сравнение концентраций металлов")
    
    selected_year = st.selectbox(
        "Выберите год:",
        [2021, 2022, 2023]
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
    
    st.markdown("## Параллельные координаты - многомерное сравнение")
    
    parallel_data = metals_filtered.groupby(['Location', 'Metal'])['Value'].mean().reset_index()
    parallel_pivot = parallel_data.pivot(index='Location', columns='Metal', values='Value').reset_index()
    
    for location in ['T1', 'T2', 'T3', 'T4']:
        avg_quality_class = index_filtered[location].mean()
        parallel_pivot.loc[parallel_pivot['Location'] == location, 'Avg_Quality_Class'] = avg_quality_class
    
    location_colors = {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3}
    parallel_pivot['Location_Num'] = parallel_pivot['Location'].map(location_colors)
    
    dimensions = []
    for col in ['Mn', 'Zn', 'Cu', 'Cd']:
        if col in parallel_pivot.columns:
            dimensions.append(dict(
                label=col + ' (mg/L)',
                values=parallel_pivot[col]
            ))
    
    if 'Avg_Quality_Class' in parallel_pivot.columns:
        dimensions.append(dict(
            label='Quality Class',
            values=parallel_pivot['Avg_Quality_Class']
        ))
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=parallel_pivot['Location_Num'],
            colorscale=[[0, '#3498db'], [0.33, '#2ecc71'], [0.66, '#e74c3c'], [1, '#f39c12']],
            showscale=True,
            cmin=0,
            cmax=3,
            colorbar=dict(
                title="Location",
                tickvals=[0, 1, 2, 3],
                ticktext=['T1', 'T2', 'T3', 'T4']
            )
        ),
        dimensions=dimensions
    ))
    
    fig.update_layout(
        title="Multi-dimensional Comparison: Metals and Quality",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("## Корреляция между точками мониторинга")
    
    col1, col2 = st.columns(2)
    
    with col1:
        correlation_data = {}
        for metal in ['Mn', 'Zn', 'Cu', 'Cd']:
            metal_data = metals_filtered[metals_filtered['Metal'] == metal]
            metal_pivot = metal_data.pivot_table(index=['Year', 'Period'], columns='Location', values='Value')
            correlation_data[metal] = metal_pivot.corr()
        
        overall_corr = sum(correlation_data.values()) / len(correlation_data)
        
        fig = px.imshow(
            overall_corr,
            labels=dict(x="Location", y="Location", color="Correlation"),
            x=overall_corr.columns,
            y=overall_corr.index,
            color_continuous_scale='RdBu',
            aspect="auto",
            title="Average Correlation Between Locations",
            zmin=-1,
            zmax=1
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        corr_text = """
        **Интерпретация корреляции:**
        
        - **> 0.7**: Сильная положительная корреляция
        - **0.3 - 0.7**: Умеренная корреляция
        - **-0.3 - 0.3**: Слабая или нет корреляции
        - **< -0.3**: Отрицательная корреляция
        
        Высокая корреляция между точками указывает на общие источники загрязнения или схожие гидрологические условия.
        """
        st.markdown(corr_text)
        
        st.markdown("**Средняя корреляция по парам:**")
        for i, loc1 in enumerate(['T1', 'T2', 'T3', 'T4']):
            for loc2 in ['T1', 'T2', 'T3', 'T4'][i+1:]:
                corr_val = overall_corr.loc[loc1, loc2]
                st.metric(f"{loc1} ↔ {loc2}", f"{corr_val:.3f}")
    
    st.markdown("## 3D сравнительная визуализация")
    
    metals_3d = metals_filtered.groupby(['Location', 'Year', 'Metal'])['Value'].mean().reset_index()
    
    total_metal_by_loc_year = metals_3d.groupby(['Location', 'Year'])['Value'].sum().reset_index()
    total_metal_by_loc_year.columns = ['Location', 'Year', 'Total_Metals']
    
    quality_by_loc_year = index_filtered.melt(id_vars=['Year'], var_name='Location', value_name='Quality_Class')
    
    data_3d = total_metal_by_loc_year.merge(quality_by_loc_year, on=['Location', 'Year'])
    
    location_colors_map = {'T1': '#3498db', 'T2': '#2ecc71', 'T3': '#e74c3c', 'T4': '#f39c12'}
    
    fig = go.Figure()
    
    for location in ['T1', 'T2', 'T3', 'T4']:
        loc_data = data_3d[data_3d['Location'] == location]
        
        fig.add_trace(go.Scatter3d(
            x=loc_data['Year'],
            y=loc_data['Total_Metals'],
            z=loc_data['Quality_Class'],
            mode='markers+lines',
            name=location,
            marker=dict(
                size=12,
                color=location_colors_map[location],
                line=dict(color='white', width=2)
            ),
            line=dict(color=location_colors_map[location], width=4),
            text=[f"{loc}<br>Year: {y}<br>Total Metals: {tm:.4f} mg/L<br>Quality Class: {qc}" 
                  for loc, y, tm, qc in zip(loc_data['Location'], loc_data['Year'], 
                                            loc_data['Total_Metals'], loc_data['Quality_Class'])],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    
    fig.update_layout(
        title="3D Location Comparison: Year × Total Metals × Quality Class",
        scene=dict(
            xaxis=dict(title='Year', gridcolor='lightgray', dtick=1),
            yaxis=dict(title='Total Metal Concentration (mg/L)', gridcolor='lightgray'),
            zaxis=dict(title='Water Quality Class', gridcolor='lightgray', dtick=1),
            bgcolor='white'
        ),
        height=600,
        showlegend=True,
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("## Сравнительный рейтинг точек")
    
    ranking_scores = {}
    
    for location in ['T1', 'T2', 'T3', 'T4']:
        loc_metals = metals_filtered[metals_filtered['Location'] == location]['Value'].mean()
        loc_quality = index_filtered[location].mean()
        
        ranking_scores[location] = {
            'Avg_Metal_Concentration': loc_metals,
            'Avg_Quality_Class': loc_quality,
            'Metal_Rank': 0,
            'Quality_Rank': 0
        }
    
    sorted_by_metals = sorted(ranking_scores.items(), key=lambda x: x[1]['Avg_Metal_Concentration'])
    for rank, (loc, _) in enumerate(sorted_by_metals, 1):
        ranking_scores[loc]['Metal_Rank'] = rank
    
    sorted_by_quality = sorted(ranking_scores.items(), key=lambda x: x[1]['Avg_Quality_Class'])
    for rank, (loc, _) in enumerate(sorted_by_quality, 1):
        ranking_scores[loc]['Quality_Rank'] = rank
    
    for loc in ranking_scores:
        ranking_scores[loc]['Overall_Score'] = (ranking_scores[loc]['Metal_Rank'] + 
                                                 ranking_scores[loc]['Quality_Rank']) / 2
    
    ranking_df = pd.DataFrame(ranking_scores).T
    ranking_df = ranking_df.sort_values('Overall_Score')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=ranking_df.index,
            x=ranking_df['Metal_Rank'],
            name='Metal Concentration Rank',
            orientation='h',
            marker=dict(color='lightcoral')
        ))
        
        fig.add_trace(go.Bar(
            y=ranking_df.index,
            x=ranking_df['Quality_Rank'],
            name='Quality Class Rank',
            orientation='h',
            marker=dict(color='lightblue')
        ))
        
        fig.update_layout(
            title="Location Ranking (1=Best, 4=Worst)",
            xaxis_title="Rank",
            yaxis_title="Location",
            barmode='group',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    
    st.markdown("## Статистическое сравнение")
    
    stats_comparison = []
    
    for location in ['T1', 'T2', 'T3', 'T4']:
        loc_data = metals_filtered[metals_filtered['Location'] == location]['Value']
        
        stats_comparison.append({
            'Location': location,
            'Mean': loc_data.mean(),
            'Median': loc_data.median(),
            'Std Dev': loc_data.std(),
            'Min': loc_data.min(),
            'Max': loc_data.max(),
            'CV (%)': (loc_data.std() / loc_data.mean() * 100) if loc_data.mean() != 0 else 0,
            'Sample Size': len(loc_data)
        })
    
    stats_df = pd.DataFrame(stats_comparison).set_index('Location')
    stats_df = stats_df.round(4)
    
    st.dataframe(stats_df, use_container_width=True)
    
    st.info("""
    **Коэффициент вариации (CV)** показывает относительную изменчивость данных:
    - CV < 15%: Низкая вариабельность (стабильное качество)
    - CV 15-30%: Умеренная вариабельность
    - CV > 30%: Высокая вариабельность (нестабильное качество)
    """)


elif page == "Тренды":
    st.title("Анализ трендов")
    st.markdown("### Долгосрочные тренды загрязнения воды (2020-2023)")
    
    metals_filtered = metals_df[metals_df['Year'].between(2020, 2023)]
    index_filtered = index_df[index_df['Year'].between(2020, 2023)]
    discharge_filtered = discharge_df[discharge_df['Year'].between(2020, 2023)]
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
    
    st.markdown("## Комплексный анализ показателей")
    
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    st.markdown("## Статистика изменений")
    metals_2020 = metals_filtered[metals_filtered['Year'] == 2020]['Value'].mean()
    metals_2023 = metals_filtered[metals_filtered['Year'] == 2023]['Value'].mean()
    
    if pd.notna(metals_2020) and pd.notna(metals_2023) and metals_2020 != 0:
        metals_change = ((metals_2023 - metals_2020) / metals_2020) * 100
    else:
        metals_change = 0
    
    discharge_2020_data = discharge_filtered[discharge_filtered['Year'] == 2020]['Average_Discharge']
    discharge_2023_data = discharge_filtered[discharge_filtered['Year'] == 2023]['Average_Discharge']
    
    if len(discharge_2020_data) > 0 and len(discharge_2023_data) > 0:
        discharge_2020 = discharge_2020_data.values[0]
        discharge_2023 = discharge_2023_data.values[0]
        if pd.notna(discharge_2020) and pd.notna(discharge_2023) and discharge_2020 != 0:
            discharge_change = ((discharge_2023 - discharge_2020) / discharge_2020) * 100
        else:
            discharge_change = 0
    else:
        discharge_2020 = 0
        discharge_2023 = 0
        discharge_change = 0
    
    cols = st.columns(3)
    
    with cols[0]:
        st.metric(
            label="Изменение концентрации металлов",
            value=f"{metals_2023:.4f} мг/л" if pd.notna(metals_2023) else "N/A",
            delta=f"{metals_change:+.1f}% с 2020" if metals_change != 0 else None
        )
    
    with cols[1]:
        st.metric(
            label="Изменение расхода воды",
            value=f"{discharge_2023:.2f} м³/с" if pd.notna(discharge_2023) else "N/A",
            delta=f"{discharge_change:+.1f}% с 2020" if discharge_change != 0 else None
        )
    
    with cols[2]:
        avg_quality_2020 = index_filtered[index_filtered['Year'] == 2020][['T1', 'T2', 'T3', 'T4']].mean().mean()
        avg_quality_2023 = index_filtered[index_filtered['Year'] == 2023][['T1', 'T2', 'T3', 'T4']].mean().mean()
        
        if pd.notna(avg_quality_2020) and pd.notna(avg_quality_2023):
            quality_change = avg_quality_2023 - avg_quality_2020
        else:
            quality_change = 0
        
        st.metric(
            label="Изменение класса качества",
            value=f"{avg_quality_2023:.1f}" if pd.notna(avg_quality_2023) else "N/A",
            delta=f"{quality_change:+.1f} с 2020" if quality_change != 0 else None
        )
    
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


elif page == "Выводы":
    st.title("Выводы и рекомендации")
    st.markdown("### Анализ состояния качества воды (2020-2023)")
    
    metals_filtered = metals_df[metals_df['Year'].between(2020, 2023)]
    index_filtered = index_df[index_df['Year'].between(2020, 2023)]
    discharge_filtered = discharge_df[discharge_df['Year'].between(2020, 2023)]
    st.markdown("## Общие выводы")
    
    st.markdown("""
    На основе комплексного анализа данных мониторинга четырех точек наблюдения за период 2020-2023 годов можно сделать следующие выводы:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Качество воды")
        
        avg_quality = index_filtered[['T1', 'T2', 'T3', 'T4']].mean()
        best_location = avg_quality.idxmin()
        worst_location = avg_quality.idxmax()
        
        st.markdown(f"""
        **Основные показатели:**
        - Лучшее качество воды: **{best_location}** (средний класс {avg_quality[best_location]:.1f})
        - Наибольшее загрязнение: **{worst_location}** (средний класс {avg_quality[worst_location]:.1f})
        - Средний класс качества: **{avg_quality.mean():.1f}**
        
        Качество воды на всех точках мониторинга находится в диапазоне от чистой до умеренно загрязненной.
        """)
    
    with col2:
        st.markdown("### Тяжелые металлы")
        
        avg_metals = metals_filtered.groupby('Metal')['Value'].mean().sort_values(ascending=False)
        
        st.markdown("""
        **Средние концентрации (мг/л):**
        """)
        
        for metal, value in avg_metals.items():
            metal_name = {'Mn': 'Марганец', 'Zn': 'Цинк', 'Cu': 'Медь', 'Cd': 'Кадмий'}[metal]
            st.markdown(f"- **{metal_name} ({metal})**: {value:.4f}")
        
        st.markdown("""
        
        Наблюдается варьирование концентраций тяжелых металлов по точкам мониторинга и временным периодам.
        """)
    
    st.markdown("## Динамика изменений")
    
    col1, col2, col3 = st.columns(3)
    
    quality_2020 = index_filtered[index_filtered['Year'] == 2020][['T1', 'T2', 'T3', 'T4']].mean().mean()
    quality_2023 = index_filtered[index_filtered['Year'] == 2023][['T1', 'T2', 'T3', 'T4']].mean().mean()
    
    if pd.notna(quality_2020) and pd.notna(quality_2023):
        quality_trend = quality_2023 - quality_2020
    else:
        quality_trend = 0
    
    with col1:
        st.metric(
            "Изменение качества воды",
            f"{quality_2023:.1f}" if pd.notna(quality_2023) else "N/A",
            f"{quality_trend:+.1f}" if quality_trend != 0 else None,
            help="Средний класс качества воды (2020 → 2023)"
        )
    
    metals_2020 = metals_filtered[metals_filtered['Year'] == 2020]['Value'].mean()
    metals_2023 = metals_filtered[metals_filtered['Year'] == 2023]['Value'].mean()
    
    if pd.notna(metals_2020) and pd.notna(metals_2023) and metals_2020 != 0:
        metals_trend = ((metals_2023 - metals_2020) / metals_2020) * 100
    else:
        metals_trend = 0
    
    with col2:
        st.metric(
            "Концентрация металлов",
            f"{metals_2023:.4f} мг/л" if pd.notna(metals_2023) else "N/A",
            f"{metals_trend:+.1f}%" if metals_trend != 0 else None,
            help="Средняя концентрация (2020 → 2023)"
        )
    
    discharge_2020_data = discharge_filtered[discharge_filtered['Year'] == 2020]['Average_Discharge']
    discharge_2023_data = discharge_filtered[discharge_filtered['Year'] == 2023]['Average_Discharge']
    
    if len(discharge_2020_data) > 0 and len(discharge_2023_data) > 0:
        discharge_2020 = discharge_2020_data.values[0]
        discharge_2023 = discharge_2023_data.values[0]
        if pd.notna(discharge_2020) and pd.notna(discharge_2023) and discharge_2020 != 0:
            discharge_trend = ((discharge_2023 - discharge_2020) / discharge_2020) * 100
        else:
            discharge_trend = 0
    else:
        discharge_2023 = 0
        discharge_trend = 0
    
    with col3:
        st.metric(
            "Расход воды",
            f"{discharge_2023:.1f} м³/с" if pd.notna(discharge_2023) else "N/A",
            f"{discharge_trend:+.1f}%" if discharge_trend != 0 else None,
            help="Средний годовой расход (2020 → 2023)"
        )
    
    st.markdown("## Рекомендации")
    
    st.markdown("""
    На основе проведенного анализа рекомендуется:
    
    1. **Продолжение мониторинга**
       - Поддерживать регулярный мониторинг качества воды на всех точках
       - Увеличить частоту пробоотбора в точках с наибольшим загрязнением
       - Расширить спектр контролируемых параметров
    
    2. **Меры по улучшению качества воды**
       - Выявить и устранить источники загрязнения
       - Разработать план мероприятий по снижению концентрации тяжелых металлов
       - Внедрить системы очистки воды при необходимости
    
    3. **Анализ и прогнозирование**
       - Использовать собранные данные для прогнозирования трендов
       - Разработать систему раннего оповещения о превышении пороговых значений
       - Проводить регулярный анализ данных для оценки эффективности принимаемых мер
    
    4. **Информирование**
       - Обеспечить прозрачность данных мониторинга
       - Регулярно публиковать отчеты о состоянии качества воды
       - Информировать население о состоянии водных ресурсов
    """)
    
    st.markdown("## Заключение")
    
    st.info("""
    **Общее состояние:** Система мониторинга показывает, что качество воды в исследуемых точках 
    требует постоянного контроля и принятия мер по предотвращению дальнейшего ухудшения экологической ситуации.
    
    **Позитивные аспекты:** Систематический сбор данных позволяет отслеживать динамику изменений и своевременно 
    реагировать на негативные тенденции.
    
    **Требует внимания:** Необходимо усилить контроль за концентрацией тяжелых металлов и разработать 
    комплексные меры по улучшению ситуации.
    """)


st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Система мониторинга качества воды</p>
        <p>Данные обновлены: Февраль 2026</p>
    </div>
    """, unsafe_allow_html=True)
