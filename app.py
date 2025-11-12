import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


#streamlit setup
st.set_page_config(page_title="Music & Mental Health", layout="wide")
st.title("Music & Mental Health Survey Analysis (Interactive Dashboard)")

#load dataset
file_path = "https://raw.githubusercontent.com/mohidqadeer123/streamlit-tests/main/Dataset.csv"
df = pd.read_csv(file_path)

#identify key columns
health_cols = [c for c in df.columns if any(x in c for x in ["Anxiety", "Depression", "Insomnia", "OCD"])]
genre_cols = [c for c in df.columns if c.startswith("Frequency [")]
bpm_col = "BPM" if "BPM" in df.columns else None  # adjust if your BPM column name differs

# Frequency map for listening type
freq_map = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Very frequently": 3
}

genre_freq_cols = [col for col in df.columns if col.startswith("Frequency")]
df[genre_freq_cols] = df[genre_freq_cols].replace(freq_map)
df["active_genre_count"] = (df[genre_freq_cols] >= 2).sum(axis=1)
df["listening_type"] = df["active_genre_count"].apply(lambda x: "Single" if x == 1 else "Multiple")

#clean and prepare data
df_clean = df.dropna(subset=health_cols + ["Hours per day", "Exploratory", "Music effects"])
df_clean[genre_cols] = df_clean[genre_cols].apply(pd.to_numeric, errors="coerce")
df_clean[health_cols] = df_clean[health_cols].apply(pd.to_numeric, errors="coerce")

# add 'listening type' to df_clean
if "listening_type" in df.columns:
    df_clean["listening_type"] = df.loc[df_clean.index, "listening_type"]

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

# Average Mental Health by Fav Genre
if not filtered_df.empty:
    genre_subset = filtered_df[["Fav genre"] + health_cols].dropna()
    if not genre_subset.empty:
        genre_means = genre_subset.groupby("Fav genre")[health_cols].mean().reset_index()
        genre_means["avg_score"] = genre_means[health_cols].mean(axis=1)
        genre_means = genre_means.sort_values("avg_score")
        
        # Bar Plot
        st.subheader("🎚 Relationship of Average mental health with Favourite Genre and Listening style")
        st.markdown("### 📊 : Which music genre seems to be the best to fight depression?")
        fig5 = px.bar(genre_means, 
                  x="Fav genre", 
                  y=health_cols, 
                  barmode="group",
                title="Average Mental Health Scores vs Fav Genre",
                labels={"value": "Average Mental Score", "Fav genre": "Music Genre"},
                color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig5.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig5, use_container_width=True)
    else:
        print("⚠️ No genre data available after filtering.")


# Average Mental Health vs Listening Type
if "listening_type" in filtered_df.columns:
    subset = filtered_df[["listening_type"] + health_cols].dropna()
    if not subset.empty:
        mh_melted = subset.melt(
                id_vars="listening_type",
                value_vars=health_cols,
                var_name="Condition", value_name="Score"
        )
        # Whisker Plot
        st.markdown("### 📊 : Do people who spend more time listening to a single favorite genre report different mental health outcomes compared to those who spread their time across multiple genres? ")
        fig6 = px.box(mh_melted,
                x="listening_type", 
                y="Score", color="Condition",
                title="Mental Health Outcomes: Single vs Multi-Genre Listeners",
                labels={"listening_type": "Listening Style"}
        )
        st.plotly_chart(fig6, use_container_width=True)
    else:
        print("⚠️ No data for listening type comparison after filtering.")
else:
    print("⚠️ 'listening_type' not found in filtered dataset.")

# Age groups
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0, 25, 40, 60, 100],
    labels=['18-25', '26-40', '41-60', '60+'],
    include_lowest=True
)

mh_cols = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

#slider
hours_range = st.slider("🎧 Select Hours of Music Listening per Day", 0.0, 10.0, (0.0, 10.0))
filtered_df = df[(df["Hours per day"] >= hours_range[0]) & (df["Hours per day"] <= hours_range[1])]
filtered_df = df_clean[
    df_clean["Hours per day"].between(hours_range[0], hours_range[1])
].copy()
if not filtered_df.empty:
    age_group_summary = filtered_df.groupby('Age_Group')[mh_cols].mean()
    age_group_summary['Avg_Hours'] = filtered_df.groupby('Age_Group')["Hours per day"].mean()

    # Heatmap
    st.subheader("🧠 Average Mental Health Scores by Age Group and Listening Hours")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(age_group_summary, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Average Mental Health Scores by Age Group (Filtered by Hours per Day)")
    ax.set_ylabel("Age Group")
    st.pyplot(fig)
else:
    st.warning("⚠️ No data available for the selected hours range.")

    
        




