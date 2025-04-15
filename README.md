# 👶 Baby Tracker Dashboard

This web application is a comprehensive visualization dashboard built with **Streamlit** to explore and analyze baby-related data from a CSV log. It helps caregivers understand patterns in **sleep**, **diapers**, and **feeding** habits over time.

## 📊 Features

### 1. Sleep Heatmap
- **Hourly heatmap** that visualizes sleep/wake states across multiple days.
- Each row represents a date; each column an hourly time slot.
- Colors:
  - 🟦 `Royal Blue`: Asleep
  - 🔷 `Light Blue`: Awake
- Helps identify trends in nighttime awakenings and sleep windows.

---

### 2. Diaper Analysis
- **Stacked bar chart** showing the number of `Pee` and `Poo` events per day.
- Color-coded:
  - 🟨 `Gold`: Pee
  - 🟫 `SaddleBrown`: Poo
- Filters by date range.
- **Summary statistics table**:
  - Max, Min, and Average counts per day for both categories.
- **Trend line chart** showing how the average daily frequency of pee/poo events evolves over time.
- **Average time between diaper changes** is calculated and visualized with:
  - Daily intervals.
  - Moving average line to highlight trends.

---

### 3. Breastfeeding Tracker
- **Stacked bar chart** showing feeding sessions categorized by:
  - 🟣 `Right Breast`
  - 🔴 `Left Breast`
- Only includes entries where `Start Location = Breast`.
- Summary stats:
  - Average daily sessions for left and right.
- **Total breastfeeding time chart**:
  - Total hours spent breastfeeding per day.
  - Includes a **trend line** showing moving average over time.
- **Interval analysis**:
  - Average time between breastfeeding sessions (in hours).
  - **Line chart** with moving average to visualize how intervals evolve day-to-day.

---

## 📆 Date Filtering
All visualizations support **start date** and **end date** filtering to analyze specific time windows.

---

## 📁 Data Input
- The app reads from a structured CSV file where each row contains events like sleep, feeding, and diaper changes.

---

## 🛠 Tech Stack
- **Python**
- **Streamlit**
- **Pandas**
- **Plotly**

---

## 🚀 Deployment
https://baby-tracker-dashboard.streamlit.app/

---

## 💡 Future Ideas
- Add formula or bottle feeding tracking.
- Weekly or monthly summaries.
- Export filtered views to PDF/CSV.

---

Feel free to contribute or customize it further for your baby's needs! 👶📈
