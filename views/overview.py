import streamlit as st
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
from data_manager import DataManager

class OverviewSection:
    def __init__(self, df):
        self.df = df

    def render(self):
        st.header("📊 Overview")
        tab1, tab2 = st.tabs(["📅 Daily Overview", "🌍 Global Metrics (All Time)"])

        with tab1:            
            self._render_daily_view()
        with tab2:
            self._render_global_metrics()

    def _render_global_metrics(self):
        def fmt_num(val):
            return f"{val:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")

        def fmt_int(val):
            return f"{val:,.0f}".replace(",", ".")
        
        sleep_df = DataManager.filter_by_category(self.df, 'Sleep')
        waking_df = DataManager.filter_by_category(self.df, 'Night waking')
        
        total_sleep_hrs = 0
        night_sleep_hrs = 0
        nap_sleep_hrs = 0
        total_naps_count = 0
        
        if not sleep_df.empty:
            is_nap = sleep_df.astype(str).apply(lambda x: x.str.contains('Nap', case=False)).any(axis=1)
            total_naps_count = is_nap.sum()            
            valid_sleep = sleep_df.dropna(subset=['Start', 'End']).copy()
            if not valid_sleep.empty:
                # Recalcular is_nap para filas válidas
                is_nap_valid = valid_sleep.astype(str).apply(lambda x: x.str.contains('Nap', case=False)).any(axis=1)
                valid_sleep['Duration'] = (valid_sleep['End'] - valid_sleep['Start']).dt.total_seconds() / 3600
                
                nap_sleep_hrs = valid_sleep[is_nap_valid]['Duration'].sum()
                night_sleep_hrs = valid_sleep[~is_nap_valid]['Duration'].sum()
                total_sleep_hrs = nap_sleep_hrs + night_sleep_hrs

        total_wakings_count = len(waking_df)


        feed_df = DataManager.filter_by_category(self.df, 'Feed')
        is_bottle = feed_df.astype(str).apply(lambda x: x.str.contains('Bottle|Formula', case=False)).any(axis=1)        
        breast_df = feed_df[~is_bottle].copy()
        total_breast_hrs = 0
        if not breast_df.empty:
            valid_breast = breast_df.dropna(subset=['Start', 'End'])
            total_breast_hrs = (valid_breast['End'] - valid_breast['Start']).dt.total_seconds().sum() / 3600

        bottle_df = feed_df[is_bottle].copy()
        total_bottles = len(bottle_df)
        total_formula_vol = 0
        
        if not bottle_df.empty:
            def extract_ml(row):
                for val in row.values:
                    val_str = str(val).lower()
                    if 'ml' in val_str:
                        match = re.search(r'(\d+)\s*ml', val_str)
                        if match: return float(match.group(1))
                return 0
            total_formula_vol = bottle_df.apply(extract_ml, axis=1).sum()


        diaper_df = DataManager.filter_by_category(self.df, 'Diaper')
        total_diapers = len(diaper_df)
        total_pee = 0
        total_poo = 0
        
        if not diaper_df.empty:
            text_data = diaper_df.astype(str).agg(' '.join, axis=1).str.lower()
            total_pee = text_data.str.contains('pee|mojado', na=False).sum()
            total_poo = text_data.str.contains('poo|sucio', na=False).sum()

        
        st.caption(f"📊 Global stats based on {len(self.df)} records.")
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.markdown("### 💤 Sleep & Rest")
                st.metric(
                    "Total Sleep Time", 
                    f"{fmt_num(total_sleep_hrs)} h", 
                    help=f"🌙 Night: {fmt_num(night_sleep_hrs)}h\n☀️ Nap: {fmt_num(nap_sleep_hrs)}h",
                )
                st.metric("Total Naps", fmt_int(total_naps_count))
                st.metric("Night Wakings", fmt_int(total_wakings_count))

        with c2:
            with st.container(border=True):
                st.markdown("### 🍼 Feeding")
                st.metric("Breastfeeding Time", f"{fmt_num(total_breast_hrs)} h")
                st.metric("Bottles Given", fmt_int(total_bottles))
                
                if total_formula_vol > 1000:
                    vol_str = f"{fmt_num(total_formula_vol/1000)} L"
                else:
                    vol_str = f"{fmt_int(total_formula_vol)} ml"
                    
                st.metric("Formula Consumed", vol_str)

        with c3:
            with st.container(border=True):
                st.markdown("### 🩲 Diapers")
                st.metric("Total Diapers", fmt_int(total_diapers))
                st.metric("Pee (Wet)", fmt_int(total_pee))
                st.metric("Poo (Dirty)", fmt_int(total_poo))


    def _render_daily_view(self):
        if not self.df.empty and 'Start' in self.df.columns:
            last_updated = self.df["Start"].max()
        else:
            last_updated = datetime.datetime.now()

        col_date, _ = st.columns([1, 2])
        with col_date:
            custom_date = st.date_input(
                "Select date",
                value=last_updated,
                max_value=datetime.datetime.now(),
                min_value=datetime.date(2024, 1, 1)
            )

        mask = self.df["Start"].dt.date == pd.to_datetime(custom_date).date()
        daily_df = self.df[mask].copy()

        col1, col2 = st.columns([2, 1], border=True)
        
        with col1:
            metrics = self._polar_chart(daily_df)
        
        woke_up, bed_time, diapers, feeds, naps, night_wakes = metrics

        with col2:
            st.subheader(f"📅 {custom_date.strftime('%d %b %Y')}")
            st.divider()
            st.write(f"**☀️ Woke Up:** {woke_up}")
            st.write(f"**🛌 Bed Time:** {bed_time}")
            st.divider()
            st.write(f"**🤱 Feeds:** {feeds}")
            st.write(f"**🩲 Diapers:** {diapers}")
            st.divider()
            st.write(f"**💤 Naps:** {naps}")
            st.write(f"**⚡ Night Wakings:** {night_wakes}")

    def _polar_chart(self, data_frame):
        single_events = ['Woke up', 'Bed time', 'Bed Time', 'Diaper']
        segment_events = ['Sleep', 'Night waking', 'Feed']

        def to_decimal(t):
            return t.hour + t.minute / 60

        def hora_a_angulo(hora_decimal):
            return (90 - (hora_decimal % 24) * 15) % 360

        eventos = {}      
        colores = {
            "Woke up": "gold", "Bed time": "orange", "Bed Time": "orange",
            "Sleep": "royalblue", "Nap": "lightblue", "Night waking": "red",
            "Diaper": "pink", "Feed": "lightgreen"
        }    

        fig = go.Figure()
        woke_up_time = '-'
        bed_time = '-'
        diaper_count_total = 0

        # SINGLE EVENTS
        for event in single_events:        
            event_df = data_frame[data_frame['Type'].str.lower() == event.lower()]
            if event_df.empty: continue
            event_df = event_df.dropna(subset=['Start'])
            
            diaper_counter = 1
            for _, row in event_df.iterrows():
                key_name = event.capitalize()
                key = key_name if len(event_df) <= 1 else f"{key_name} {diaper_counter}" 
                
                if "Diaper" in key_name:
                    colores[key] = "pink"
                    diaper_count_total += 1
                elif "Woke" in key_name: colores[key] = "gold"
                elif "Bed" in key_name: colores[key] = "orange"
                
                time_val = row['Start'].time()
                if "Woke" in key_name: woke_up_time = time_val.strftime('%H:%M')
                if "Bed" in key_name: bed_time = time_val.strftime('%H:%M')
                
                eventos[key] = to_decimal(time_val)
                diaper_counter += 1 

        # SEGMENTED EVENTS  
        feed_counter = 0
        nap_counter = 0
        night_waking_counter = 0
        first_night_sleep_time = None

        for event in segment_events:
            event_df = DataManager.filter_by_category(data_frame, event)
            if event_df.empty: continue
            event_df = event_df.dropna(subset=['Start', 'End'])

            if event == "Sleep":
                nap_df = event_df[event_df['Notes'] == "Nap"]
                nap_counter = len(nap_df)
                night_df = event_df[event_df['Notes'] != "Nap"] 
                night_waking_counter = len(night_df)
                
                if not night_df.empty:
                    first_night_sleep = night_df.sort_values(by='Start').iloc[0]
                    first_night_sleep_time = first_night_sleep['Start'].time().strftime('%H:%M')

                sleep_variants = [("Nap", nap_df, "lightblue"), ("Night Sleep", night_df, "royalblue")]
            else:
                sleep_variants = [(event, event_df, colores.get(event, "red"))]

            for name, df_sub, color in sleep_variants:
                if df_sub.empty: continue                   
                if name == "Feed": feed_counter = len(df_sub)

                all_angles = []
                all_radii = []

                for _, row in df_sub.iterrows():
                    start = row['Start'].time()
                    end = row['End'].time()
                    start_h = to_decimal(start)
                    end_h = to_decimal(end)
                    if end_h < start_h: end_h += 24

                    hours_range = np.linspace(start_h, end_h, 50)
                    angles = [hora_a_angulo(h % 24) for h in hours_range]
                    radii = [1] * len(angles)
                    all_angles.extend(angles + [None])
                    all_radii.extend(radii + [None])

                fig.add_trace(go.Scatterpolar(
                    r=all_radii, theta=all_angles, mode='lines',
                    line=dict(color=color, width=10), name=name, showlegend=True
                ))

        # Fallback Bed Time
        if bed_time == '-' and first_night_sleep_time:
            bed_time = f"{first_night_sleep_time}*"

        for evento, hora in eventos.items():
            angulo = hora_a_angulo(hora)
            base_name = evento.split()[0]
            color_final = 'grey'
            if 'Diaper' in base_name: color_final = 'pink'
            elif 'Woke' in base_name: color_final = 'gold'
            elif 'Bed' in base_name: color_final = 'orange'
            
            fig.add_trace(go.Scatterpolar(
                r=[1], theta=[angulo], mode='markers+text',
                marker=dict(color=color_final, size=30),
                text=[evento], textposition='top center', name=evento, showlegend=False
            ))

        fig.update_layout(
            height=600,
            polar=dict(
                radialaxis=dict(visible=False),
                angularaxis=dict(direction='counterclockwise', rotation=180,
                    tickmode='array', tickvals=[hora_a_angulo(h) for h in range(0, 24)],
                    ticktext=[f"{h%24}:00" for h in range(0, 24)])
            ),
            showlegend=True, margin=dict(t=20, b=20, l=0, r=0)
        )

        st.plotly_chart(fig, use_container_width=True)
        return woke_up_time, bed_time, diaper_count_total, feed_counter, nap_counter, night_waking_counter