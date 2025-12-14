import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_manager import DataManager

class GrowthSection:
    def __init__(self, df):
        self.df = df

    def render(self):
        st.header("📈 Growth Tracking")
        st.write("Standards developed using WHO Multicentre Growth Reference Study.")
        
        tab1, tab2, tab3 = st.tabs(["Height", "Weight", "Head Circumference"])

        with tab1:
            st.write('https://www.who.int/tools/child-growth-standards/standards/length-height-for-age')
            self._render_chart('Height', "Height", 'Start Location', 'cm', 
                               'data/height-girls-percentiles-who.csv', "Baby's Height vs Age", "Height (cm)")
        with tab2:
            st.write('https://www.who.int/tools/child-growth-standards/standards/weight-for-age')
            self._render_chart('Weight', "Weight", 'Start Condition', 'kg', 
                               'data/weight-girls-percentiles-who.csv', "Baby's Weight vs Age", "Weight (kg)")
        with tab3:
            st.write('https://www.who.int/tools/child-growth-standards/standards/head-circumference-for-age')
            self._render_chart('Head', "Head Circumference", 'End Condition', 'cm', 
                               'data/head-girls-percentiles-who.csv', "Baby's Head Circumference vs Age", "Head Circumference (cm)")

    def _render_chart(self, measure_name, value_column, source_column, unit, csv_path, title, yaxis_title, marker_symbol='diamond', marker_color='#FF69B4'):
        df = DataManager.filter_by_category(self.df, 'Growth')
        df['Date'] = df['Start'] 
        
        # Limpieza
        df[measure_name] = df[source_column].astype(str).str.replace(unit, '', regex=False)
        df[measure_name] = pd.to_numeric(df[measure_name], errors='coerce')
        
        df = df.dropna(subset=['Date', measure_name])
        
        if df.empty:
            st.info(f"No data available for {measure_name}")
            return

        birth_date = df['Date'].min()
        df['Months'] = ((df['Date'] - birth_date).dt.days / 30.44).round(1)

        # --- CÁLCULO DE RANGOS PARA CENTRAR LA GRÁFICA ---
        # 1. Rango Eje X (Edad): Desde 0 hasta la edad actual + 1 mes de margen
        max_age_months = df['Months'].max()
        x_range_max = max(max_age_months + 1, 1) # Asegurar al menos 1 mes de visualización
        
        # 2. Rango Eje Y (Valor): Min/Max del bebé + padding
        min_val = df[measure_name].min()
        max_val = df[measure_name].max()
        
        # Padding del 10% o al menos 1 unidad si solo hay un dato
        val_padding = (max_val - min_val) * 0.2 if max_val != min_val else max_val * 0.05
        if val_padding == 0: val_padding = 1
        
        y_range = [min_val - val_padding, max_val + val_padding]
        # ----------------------------------------------------

        percentiles_df = DataManager.load_growth_percentiles(csv_path)

        fig = go.Figure()

        # Datos del bebé
        fig.add_trace(go.Scatter(
            x=df['Months'], 
            y=df[measure_name],
            mode='lines+markers',
            name=f"Baby's {value_column}",
            line=dict(color=marker_color, width=2),
            marker=dict(
                symbol=marker_symbol,
                size=12,
                color=marker_color,
                line=dict(width=2, color='black')
            ),
            hovertemplate=f'Date: %{{customdata[0]}}<br>{value_column}: %{{y:.1f}} {unit}<br>',
            customdata=df['Date'].dt.strftime('%Y-%m-%d').to_frame().values
        ))

        # Percentiles
        if not percentiles_df.empty:
            percentiles = ['P1', 'P5', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99']
            for p in percentiles:
                if p in percentiles_df.columns:
                    line_style = dict(dash='dot', width=1, color='gray') if p != 'P50' else dict(dash='solid', width=2, color='darkgray')
                    fig.add_trace(go.Scatter(
                        x=percentiles_df['Months'], 
                        y=percentiles_df[p],
                        mode='lines',
                        name=f'{p} Percentile',
                        line=line_style,
                        hoverinfo='skip'
                    ))

        fig.update_layout(
            title=title, 
            xaxis_title="Age (months)", 
            yaxis_title=yaxis_title, 
            height=500,
            hovermode="x unified",
            # APLICAMOS LOS RANGOS CALCULADOS
            xaxis=dict(range=[0, x_range_max]),
            yaxis=dict(range=y_range)
        )
        
        st.plotly_chart(fig, use_container_width=True)