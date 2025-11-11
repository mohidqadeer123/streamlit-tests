import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np


#streamlit setup
st.set_page_config(page_title="Music & Mental Health", layout="wide")
st.title("Music & Mental Health Survey Analysis (Interactive Dashboard)")

#load dataset
file_path = "https://raw.githubusercontent.com/mohidqadeer123/streamlit-tests/refs/heads/main/Dataset.csv"
df = pd.read_csv(file_path)

#identify key columns
health_cols = [c for c in df.columns if any(x in c for x in ["Anxiety", "Depression", "Insomnia", "OCD"])]
genre_cols = [c for c in df.columns if c.startswith("Frequency [")]
bpm_col = "BPM" if "BPM" in df.columns else None  # adjust if your BPM column name differs

#clean and prepare data
df_clean = df.dropna(subset=health_cols + ["Hours per day", "Exploratory", "Music effects"])
df_clean[genre_cols] = df_clean[genre_cols].apply(pd.to_numeric, errors="coerce")
df_clean[health_cols] = df_clean[health_cols].apply(pd.to_numeric, errors="coerce")

df_clean["Variety"] = (df_clean[genre_cols] > 0).sum(axis=1)
df_clean["Avg_health"] = df_clean[health_cols].mean(axis=1)

#sidebar filters
st.sidebar.header("🧭 Filter Data")

#hours per day filter
min_hours, max_hours = int(df_clean["Hours per day"].min()), int(df_clean["Hours per day"].max())
hours_range = st.sidebar.slider("🎚 Hours Listening per Day", min_hours, max_hours, (min_hours, max_hours))

#avg mental health filter
min_health, max_health = float(df_clean["Avg_health"].min()), float(df_clean["Avg_health"].max())
health_range = st.sidebar.slider("🧠 Average Mental Health Score", min_health, max_health, (min_health, max_health))

#bpm filter (only if it exists)
if bpm_col and bpm_col in df_clean.columns:
    bpm_min = int(df_clean[bpm_col].min())
    bpm_max = int(min(df_clean[bpm_col].max(), 250))  # cap BPM at 250
    bpm_range = st.sidebar.slider("🎵 BPM (Beats Per Minute)", bpm_min, bpm_max, (bpm_min, bpm_max))
else:
    bpm_range = None

# --- Apply Filters ---
filtered_df = df_clean[
    (df_clean["Hours per day"].between(hours_range[0], hours_range[1])) &
    (df_clean["Avg_health"].between(health_range[0], health_range[1]))
].copy()

if bpm_range and bpm_col:
    filtered_df = filtered_df[filtered_df[bpm_col].between(bpm_range[0], bpm_range[1])]

#formatting
col1, col2 = st.columns(2)

# 1. Hours Listening vs Mental Health
with col1:
    st.subheader("Hours Listening vs Mental Health")
    fig1 = px.scatter(
        filtered_df,
        x="Hours per day",
        y="Avg_health",
        trendline="ols",
        opacity=0.6,
        color_discrete_sequence=["#1f77b4"],
        labels={"Hours per day": "Hours Listening per Day", "Avg_health": "Average Mental Health Score"},
        title="Does listening longer affect mental health?"
    )
    fig1.update_traces(marker=dict(size=7))
    st.plotly_chart(fig1, use_container_width=True)

# 2. Exploratory vs Reported Music Effects
with col2:
    st.subheader("Exploring New Genres vs Reported Effects")
    fig2 = px.histogram(
        filtered_df,
        x="Music effects",
        color="Exploratory",
        barmode="group",
        text_auto=True,
        title="Exploring new genres/artists vs reported effects on mental health",
        labels={"Music effects": "Reported Effect of Music", "count": "Number of Respondents"}
    )
    fig2.update_layout(xaxis_tickangle=20)
    st.plotly_chart(fig2, use_container_width=True)

# 3. BPM vs Mental Health (Scatter + Box Plot)
st.subheader("🎚 Relationship Between BPM and Mental Health")

if bpm_col and bpm_col in filtered_df.columns:
    #scatter plot
    st.markdown("#### 🔹 Scatter: Does faster music correlate with better or worse mental health?")
    fig3 = px.scatter(
        filtered_df,
        x=bpm_col,
        y="Avg_health",
        color="Exploratory",
        trendline="ols",
        opacity=0.7,
        title="BPM (Beats Per Minute) vs Average Mental Health",
        labels={
            bpm_col: "Beats Per Minute (Preferred Tempo)",
            "Avg_health": "Average Mental Health Score",
            "Exploratory": "Explores New Genres"
        },
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig3.update_traces(marker=dict(size=8))
    st.plotly_chart(fig3, use_container_width=True)

    #bpm bins from slowest to fastest
    st.markdown("#### 🔹 Box Plot: Mental Health Across BPM Ranges")
    num_bins = 5
    try:
        #quantile-based bins (limit to 250 BPM)
        bins = np.quantile(filtered_df[bpm_col].dropna(), np.linspace(0, 1, num_bins + 1))
        bins = np.clip(bins, None, 250)
        bins = np.unique(bins)  # remove duplicates if small data variation
        labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)]

        #cut and label by BPM ranges
        filtered_df["BPM_Range"] = pd.cut(filtered_df[bpm_col], bins=bins, labels=labels, include_lowest=True)

        #ensure order from slowest to fastest
        filtered_df["BPM_Range"] = pd.Categorical(filtered_df["BPM_Range"], categories=labels, ordered=True)

        #box plot
        fig4 = px.box(
            filtered_df.sort_values("BPM_Range"),
            x="BPM_Range",
            y="Avg_health",
            color="BPM_Range",
            title="Mental Health Scores Across BPM Ranges (Slow → Fast)",
            labels={"BPM_Range": "Tempo Range (BPM)", "Avg_health": "Average Mental Health Score"},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ Could not compute BPM bins: {e}")

else:
    st.info("⚠️ BPM data not found in this dataset.")

