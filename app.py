import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import json
import advanced_analytics as adv
from scipy.spatial.distance import pdist          # если их нет
from scipy.cluster.hierarchy import linkage
import plotly.figure_factory as ff

st.set_page_config(
    page_title="ADR Analytics Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Глобальные стили
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
[data-testid="stMetricLabel"] { font-size: 0.75rem; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ---- Загрузка данных ----
@st.cache_data
def load_adr():
    path = "processed_adr_data.csv"
    if not os.path.exists(path):
        st.error("Файл processed_adr_data.csv не найден. Запустите parser.py.")
        st.stop()
    return pd.read_csv(path)

@st.cache_data
def load_signals():
    path = "signal_stats.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

df_all = load_adr()
df_signals = load_signals()

# ---- Заголовок ----
st.title(" ADR Signal Analysis Dashboard (синтетические данные)")
st.caption(
    f"Данные: drug_info.json + patient_info.json  |  "
    f"{len(df_all):,} ADR строк  ·  {df_all['report_id'].nunique():,} уникальных отчётов  ·  "
    f"{df_all['drug_name'].nunique():,} препаратов  ·  {df_all['reaction'].nunique():,} уникальных реакций"
)

# ---- Вкладки ----
tab_overview, tab_drug, tab_signal, tab_congenital, tab_advanced, tab_3d, tab_prognosis = st.tabs([
    " Overview",
    " Drug Explorer",
    " Signal Detection",
    " Congenital Diseases",
    " Advanced Analytics",
    " 3D Analytics",
    " Prognosis & Drug Compare",   # новая вкладка
])
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Обзор датасета")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Всего ADR строк", f"{len(df_all):,}")
    c2.metric("Уникальных отчётов", f"{df_all['report_id'].nunique():,}")
    c3.metric("Уникальных препаратов", f"{df_all['drug_name'].nunique():,}")
    c4.metric("Серьёзных сообщений", f"{(df_all['serious'] == 'Serious').sum():,}")
    c5.metric("Фатальных исходов", f"{(df_all['outcome'] == 'Fatal').sum():,}")

    st.divider()

    # Временной тренд (по месяцам)
    st.markdown("##### Тренд сообщений по месяцам")
    df_trend = df_all[df_all["receive_month"].notna()]
    trend_counts = df_trend.groupby("receive_month").size().reset_index(name="Count")
    trend_counts = trend_counts.sort_values("receive_month")
    fig_trend = px.line(trend_counts, x="receive_month", y="Count", markers=True,
                        line_shape="spline", color_discrete_sequence=["#378ADD"])
    fig_trend.update_layout(margin=dict(t=20, b=20), height=300,
                            xaxis_title="Месяц", yaxis_title="Кол-во сообщений")
    st.plotly_chart(fig_trend, use_container_width=True)

    col_l, col_r = st.columns(2)

    # Treemap: топ-5 препаратов → реакции
    with col_l:
        st.markdown("##### Структура: топ-5 препаратов по реакциям")
        top_drugs = df_all["drug_name"].value_counts().head(5).index
        treemap_df = df_all[df_all["drug_name"].isin(top_drugs)]
        fig_tree = px.treemap(treemap_df, path=["drug_name", "reaction"],
                              color="drug_name", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_tree.update_layout(margin=dict(t=20, b=20), height=400)
        st.plotly_chart(fig_tree, use_container_width=True)

    # Boxplot возраста по полу
    with col_r:
        st.markdown("##### Распределение возраста по полу")
        age_clean = df_all[(df_all["age_years"] > 0) & (df_all["age_years"] < 110)]
        fig_box = px.box(age_clean, x="sex", y="age_years", color="sex",
                         color_discrete_map={"M": "#378ADD", "F": "#D4537E"},
                         points="outliers")
        fig_box.update_layout(margin=dict(t=20, b=20), height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    col2_l, col2_r = st.columns(2)

    # Serious / Non-serious pie
    with col2_l:
        st.markdown("##### Серьёзность сообщений")
        ser_counts = df_all.drop_duplicates("report_id")["serious"].value_counts().reset_index()
        ser_counts.columns = ["Seriousness", "Count"]
        fig_pie = px.pie(ser_counts, names="Seriousness", values="Count", hole=0.45,
                         color="Seriousness", color_discrete_map={"Serious": "#E24B4A", "Non-Serious": "#1D9E75"})
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(margin=dict(t=20, b=20), showlegend=False, height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Исходы (outcome)
    with col2_r:
        st.markdown("##### Исходы реакций")
        out_counts = df_all["outcome"].value_counts().reset_index()
        out_counts.columns = ["Outcome", "Count"]
        fig_out = px.bar(out_counts, x="Outcome", y="Count", color="Outcome", text="Count")
        fig_out.update_layout(showlegend=False, margin=dict(t=20, b=20), height=300)
        fig_out.update_traces(textposition="outside")
        st.plotly_chart(fig_out, use_container_width=True)

    col3_l, col3_r = st.columns(2)

    # Топ-15 препаратов
    with col3_l:
        st.markdown("##### Топ-15 препаратов по числу отчётов")
        top_drugs15 = df_all.groupby("drug_name")["report_id"].nunique().sort_values(ascending=False).head(15).reset_index()
        top_drugs15.columns = ["Drug", "Reports"]
        fig_drugs = px.bar(top_drugs15.sort_values("Reports"), x="Reports", y="Drug",
                           orientation="h", color="Reports", color_continuous_scale="Blues")
        fig_drugs.update_layout(coloraxis_showscale=False, margin=dict(t=20, b=20), height=400)
        st.plotly_chart(fig_drugs, use_container_width=True)

    # Топ-15 реакций
    with col3_r:
        st.markdown("##### Топ-15 побочных реакций")
        top_rxn = df_all["reaction"].value_counts().head(15).reset_index()
        top_rxn.columns = ["Reaction", "Count"]
        fig_rxn = px.bar(top_rxn.sort_values("Count"), x="Count", y="Reaction",
                         orientation="h", color="Count", color_continuous_scale="Reds")
        fig_rxn.update_layout(coloraxis_showscale=False, margin=dict(t=20, b=20), height=400)
        st.plotly_chart(fig_rxn, use_container_width=True)

    col4_l, col4_r = st.columns(2)

    # Пол
    with col4_l:
        st.markdown("##### Распределение по полу")
        sex_counts = df_all.drop_duplicates("report_id")["sex"].value_counts().reset_index()
        sex_counts.columns = ["Sex", "Count"]
        fig_sex = px.bar(sex_counts, x="Sex", y="Count", color="Sex",
                         color_discrete_map={"M": "#378ADD", "F": "#D4537E", "Unknown": "#888780"},
                         text="Count")
        fig_sex.update_layout(showlegend=False, margin=dict(t=20, b=20), height=280)
        st.plotly_chart(fig_sex, use_container_width=True)

    # Возрастные группы
    with col4_r:
        st.markdown("##### Возрастные группы")
        age_buckets = df_all.drop_duplicates("report_id")["age_bucket"].value_counts().reset_index()
        age_buckets.columns = ["Age Group", "Count"]
        order = ["0–17", "18–44", "45–64", "65+", "Unknown"]
        age_buckets["Age Group"] = pd.Categorical(age_buckets["Age Group"], categories=order, ordered=True)
        age_buckets = age_buckets.sort_values("Age Group")
        fig_ageb = px.bar(age_buckets, x="Age Group", y="Count",
                          color="Age Group", color_discrete_sequence=px.colors.sequential.Teal,
                          text="Count")
        fig_ageb.update_layout(showlegend=False, margin=dict(t=20, b=20), height=280)
        fig_ageb.update_traces(textposition="outside")
        st.plotly_chart(fig_ageb, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DRUG EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_drug:
    st.subheader("Исследование препарата")

    col_filt1, col_filt2, col_filt3, col_filt4 = st.columns(4)
    with col_filt1:
        drug_list = df_all["drug_name"].value_counts().head(50).index.tolist()
        selected_drug = st.selectbox("Выберите препарат:", drug_list)
    with col_filt2:
        age_range = st.slider("Диапазон возраста (лет):", 0, 100, (0, 100))
    with col_filt3:
        sex_filter = st.multiselect("Пол:", df_all["sex"].unique().tolist(),
                                     default=df_all["sex"].unique().tolist())
    with col_filt4:
        serious_filter = st.multiselect("Серьёзность:", df_all["serious"].unique().tolist(),
                                         default=df_all["serious"].unique().tolist())

    fdf = df_all[
        (df_all["drug_name"] == selected_drug) &
        (df_all["age_years"].fillna(50).between(*age_range)) &
        (df_all["sex"].isin(sex_filter)) &
        (df_all["serious"].isin(serious_filter))
    ]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Число ADR строк", f"{len(fdf):,}")
    k2.metric("Уникальных реакций", f"{fdf['reaction'].nunique()}")
    k3.metric("% Serious", f"{(fdf['serious'] == 'Serious').mean() * 100:.1f}%")
    k4.metric("% Fatal outcome", f"{(fdf['outcome'] == 'Fatal').mean() * 100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"##### Топ-15 реакций для {selected_drug}")
        top_r = fdf["reaction"].value_counts().head(15).reset_index()
        top_r.columns = ["Reaction", "Count"]
        fig_r = px.bar(top_r.sort_values("Count"), x="Count", y="Reaction",
                       orientation="h", color="Count", color_continuous_scale="Blues")
        fig_r.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10), height=420)
        st.plotly_chart(fig_r, use_container_width=True)

    with col2:
        st.markdown("##### Серьёзность")
        fig_s = px.pie(fdf, names="serious", hole=0.4,
                       color="serious", color_discrete_map={"Serious": "#E24B4A", "Non-Serious": "#1D9E75"})
        fig_s.update_layout(margin=dict(t=10, b=10), height=200)
        st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("##### Исходы")
        out_df = fdf["outcome"].value_counts().reset_index()
        out_df.columns = ["Outcome", "Count"]
        fig_o = px.bar(out_df.sort_values("Count", ascending=False),
                       x="Outcome", y="Count", text="Count",
                       color_discrete_sequence=["#378ADD"])
        fig_o.update_layout(margin=dict(t=10, b=10), height=200)
        fig_o.update_traces(textposition="outside")
        st.plotly_chart(fig_o, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("##### Распределение возраста (по полу)")
        age_valid = fdf[fdf["age_years"].notna()]
        if not age_valid.empty:
            fig_age = px.histogram(age_valid, x="age_years", nbins=30,
                                   color="sex",
                                   color_discrete_map={"M": "#378ADD", "F": "#D4537E", "Unknown": "#888780"},
                                   barmode="overlay", opacity=0.75)
            fig_age.update_layout(margin=dict(t=10, b=10), height=280,
                                  xaxis_title="Возраст (лет)", yaxis_title="Кол-во")
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("Нет данных о возрасте для этого препарата/фильтров.")

    with col4:
        st.markdown("##### Топ-10 реакций по серьёзности")
        rxn_serious = (
            fdf.groupby(["reaction", "serious"])
            .size()
            .reset_index(name="count")
        )
        top10_r = fdf["reaction"].value_counts().head(10).index
        rxn_serious = rxn_serious[rxn_serious["reaction"].isin(top10_r)]
        fig_rs = px.bar(rxn_serious, x="count", y="reaction",
                        color="serious", orientation="h",
                        color_discrete_map={"Serious": "#E24B4A", "Non-Serious": "#1D9E75"},
                        barmode="stack")
        fig_rs.update_layout(margin=dict(t=10, b=10), height=280)
        st.plotly_chart(fig_rs, use_container_width=True)

    st.markdown("##### Врождённые заболевания (топ-12)")
    congen_df = fdf["congenital_diseases"].value_counts().head(12).reset_index()
    congen_df.columns = ["Congenital Disease", "Count"]
    if not congen_df.empty:
        fig_cong = px.bar(congen_df, x="Count", y="Congenital Disease",
                          orientation="h", color="Count",
                          color_continuous_scale="Teal")
        fig_cong.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10), height=300)
        st.plotly_chart(fig_cong, use_container_width=True)

    st.divider()
    st.markdown("##### Детальная таблица данных")
    show_cols = ["report_id", "drug_name", "reaction", "outcome", "serious",
                 "age_years", "sex", "country", "reporter_qualification",
                 "receive_year", "indication", "congenital_diseases"]
    st.dataframe(fdf[show_cols].reset_index(drop=True), use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_signal:
    st.subheader("Обнаружение сигналов — disproportionality analysis (ROR)")
    st.caption(
        "ROR (Reporting Odds Ratio) ≥ 2 и n ≥ 3 — порог для слабого сигнала. "
        "Данные синтетические, анализ демонстрационный."
    )

    if df_signals.empty:
        st.warning("signal_stats.csv не найден. Запустите parser.py.")
    else:
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sig_drug = st.selectbox(
                "Препарат:",
                ["— Все —"] + df_signals["drug_name"].value_counts().head(50).index.tolist(),
                key="sig_drug"
            )
        with col_s2:
            min_n = st.slider("Минимальное число сообщений (n):", 1, 50, 3)
        with col_s3:
            min_ror = st.slider("Минимальный ROR:", 1.0, 20.0, 2.0, step=0.5)

        sig_df = df_signals[
            (df_signals["n"] >= min_n) &
            (df_signals["ror"] >= min_ror)
        ]
        if sig_drug != "— Все —":
            sig_df = sig_df[sig_df["drug_name"] == sig_drug]

        sig_df = sig_df.sort_values("ror", ascending=False)

        st.metric("Сигналов, удовлетворяющих критериям", f"{len(sig_df):,}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### Топ-20 сигналов (препарат → реакция) по ROR")
            top_sig = sig_df.head(20).copy()
            top_sig["label"] = top_sig["drug_name"] + " → " + top_sig["reaction"]
            fig_sig = px.bar(top_sig.sort_values("ror"),
                             x="ror", y="label", orientation="h",
                             color="ror", color_continuous_scale="Reds",
                             hover_data=["n", "pct_of_drug"])
            fig_sig.update_layout(coloraxis_showscale=False,
                                  margin=dict(t=10, b=10), height=500,
                                  yaxis_title="", xaxis_title="ROR")
            st.plotly_chart(fig_sig, use_container_width=True)

        with col_b:
            st.markdown("##### ROR vs. число сообщений (пузырёк = % от всех сообщений препарата)")
            scatter_df = sig_df[sig_df["ror"] < 100].head(150)
            if not scatter_df.empty:
                fig_sc = px.scatter(
                    scatter_df, x="n", y="ror",
                    size="pct_of_drug", color="drug_name",
                    hover_data=["reaction", "pct_of_drug"],
                    labels={"n": "Число сообщений", "ror": "ROR"},
                    height=500,
                )
                fig_sc.add_hline(y=2, line_dash="dash", line_color="gray",
                                 annotation_text="ROR=2 threshold")
                fig_sc.update_layout(showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("##### Таблица сигналов")
        display_cols = ["drug_name", "reaction", "n", "total_drug", "ror", "pct_of_drug"]
        st.dataframe(
            sig_df[display_cols].rename(columns={
                "drug_name": "Drug",
                "reaction": "Reaction",
                "n": "n",
                "total_drug": "Всего сообщений по препарату",
                "ror": "ROR",
                "pct_of_drug": "% от сообщений препарата",
            }).reset_index(drop=True),
            use_container_width=True,
            height=350,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CONGENITAL DISEASES
# ══════════════════════════════════════════════════════════════════════════════
with tab_congenital:
    st.subheader("Анализ по врождённым заболеваниям")
    st.caption(
        "Выберите врождённое заболевание (или несколько), чтобы увидеть связанные ADR, "
        "препараты и демографические распределения."
    )

    # Список всех врождённых заболеваний (учитывая "Нет")
    all_congenital = sorted(df_all["congenital_diseases"].dropna().unique())
    # Переносим "Нет" в начало списка
    if "Нет" in all_congenital:
        all_congenital.remove("Нет")
        all_congenital = ["Нет"] + all_congenital

    selected_congenital = st.multiselect(
        "Выберите врождённое заболевание (одно или несколько):",
        options=all_congenital,
        default=["Нет"] if "Нет" in all_congenital else []
    )

    if not selected_congenital:
        st.info("Пожалуйста, выберите хотя бы одно заболевание.")
    else:
        # Фильтруем данные
        mask = df_all["congenital_diseases"].isin(selected_congenital)
        filtered_df = df_all[mask].copy()

        # Метрики
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Число уникальных пациентов", f"{filtered_df['report_id'].nunique():,}")
        col2.metric("Всего ADR строк", f"{len(filtered_df):,}")
        col3.metric("Серьёзные ADR (%)", f"{(filtered_df['serious'] == 'Serious').mean() * 100:.1f}%")
        col4.metric("Фатальные исходы (%)", f"{(filtered_df['outcome'] == 'Fatal').mean() * 100:.1f}%")

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("##### Топ-10 препаратов")
            top_drugs = filtered_df["drug_name"].value_counts().head(10).reset_index()
            top_drugs.columns = ["Drug", "Count"]
            fig_drug = px.bar(top_drugs, x="Count", y="Drug", orientation="h",
                              color="Count", color_continuous_scale="Blues")
            fig_drug.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10), height=400)
            st.plotly_chart(fig_drug, use_container_width=True)

        with col_right:
            st.markdown("##### Топ-10 побочных реакций")
            top_reactions = filtered_df["reaction"].value_counts().head(10).reset_index()
            top_reactions.columns = ["Reaction", "Count"]
            fig_rxn = px.bar(top_reactions, x="Count", y="Reaction", orientation="h",
                             color="Count", color_continuous_scale="Reds")
            fig_rxn.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10), height=400)
            st.plotly_chart(fig_rxn, use_container_width=True)

        st.divider()

        col_age, col_sex = st.columns(2)

        with col_age:
            st.markdown("##### Распределение по возрасту")
            age_valid = filtered_df[filtered_df["age_years"].notna()]
            if not age_valid.empty:
                fig_age = px.histogram(age_valid, x="age_years", nbins=30,
                                       color="sex", barmode="overlay", opacity=0.75,
                                       color_discrete_map={"M": "#378ADD", "F": "#D4537E"})
                fig_age.update_layout(margin=dict(t=10, b=10), height=350,
                                      xaxis_title="Возраст (лет)", yaxis_title="Кол-во")
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.info("Нет данных о возрасте для выбранной группы.")

        with col_sex:
            st.markdown("##### Распределение по полу")
            sex_counts = filtered_df["sex"].value_counts().reset_index()
            sex_counts.columns = ["Sex", "Count"]
            fig_sex = px.pie(sex_counts, names="Sex", values="Count", hole=0.4,
                             color="Sex", color_discrete_map={"M": "#378ADD", "F": "#D4537E", "Unknown": "#888780"})
            fig_sex.update_layout(margin=dict(t=10, b=10), height=350)
            st.plotly_chart(fig_sex, use_container_width=True)

        st.divider()

        # Детальная таблица
        st.markdown("##### Детальные данные (первые 100 строк)")
        show_cols = ["report_id", "drug_name", "reaction", "outcome", "serious",
                     "age_years", "sex", "congenital_diseases", "country"]
        st.dataframe(filtered_df[show_cols].head(100).reset_index(drop=True),
                     use_container_width=True, height=400)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ADVANCED ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_advanced:

    st.subheader("Advanced Analytics")
    st.caption("Если график не построен, вы увидите пояснение.")

    # 1. Тепловая карта
    st.markdown("##### 1. Co-occurrence of reactions")
    normalize = st.checkbox("Normalize (Jaccard index)", value=False, key="cooc_norm")
    fig_cooc = adv.plot_cooccurrence_heatmap(df_all, top_n=15, normalize=normalize)
    st.plotly_chart(fig_cooc, use_container_width=True)

    st.divider()

    # 2. Sankey diagram
    st.markdown("##### 2. Sankey diagram")
    fig_sankey = adv.plot_sankey(df_all)
    st.plotly_chart(fig_sankey, use_container_width=True)

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("##### 3. World map")
        fig_map = adv.plot_world_map(df_all)
        st.plotly_chart(fig_map, use_container_width=True)

    with col_right:
        st.markdown("##### 4. Drug clustering (dendrogram)")
        fig_dendro = adv.plot_dendrogram(df_all)
        st.plotly_chart(fig_dendro, use_container_width=True)

    st.divider()

    # 5. Outcome by age and seriousness
    st.markdown("##### 5. Outcome by age and seriousness")
    fig_outcome = adv.plot_outcome_by_age_serious(df_all)
    st.plotly_chart(fig_outcome, use_container_width=True)

    st.divider()

    # 6. Reaction trend over time
    st.markdown("##### 6. Reaction trend over time")
    # Выбор реакции (вынесен из функции)
    reactions = sorted(df_all['reaction'].unique())
    if reactions:
        selected_rxn = st.selectbox("Select a reaction:", reactions, key="trend_select_adv")
        fig_trend = adv.plot_reaction_trend(df_all, selected_rxn)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No reactions available")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — 3D ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_3d:
    st.subheader("🎮 3D Visualizations")
    st.caption("Интерактивные 3D-графики для исследования связей в данных (можно вращать, масштабировать).")

    # --------------------------------------------------------------------------
    # 1. 3D Scatter Plot: Возраст → Количество реакций → Серьёзность
    # --------------------------------------------------------------------------
    st.markdown("##### 1. Age vs. Number of reactions vs. Seriousness")
    # Агрегируем по отчётам
    report_stats = df_all.groupby('report_id').agg(
        age=('age_years', 'first'),
        serious=('serious', 'first'),
        n_reactions=('reaction', 'count'),
        outcome=('outcome', 'first')
    ).dropna(subset=['age']).reset_index()
    report_stats['serious_binary'] = (report_stats['serious'] == 'Serious').astype(int)

    if len(report_stats) >= 10:
        fig_3d_scatter = px.scatter_3d(
            report_stats,
            x='age',
            y='n_reactions',
            z='serious_binary',
            color='serious',
            size='n_reactions',
            hover_data=['outcome', 'report_id'],
            color_discrete_map={'Serious': '#E24B4A', 'Non-Serious': '#1D9E75'},
            title='3D: Age → #Reactions → Seriousness',
            labels={'age': 'Age (years)', 'n_reactions': '# reactions per report', 'serious_binary': 'Serious (1) / Non-Serious (0)'}
        )
        fig_3d_scatter.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')))
        st.plotly_chart(fig_3d_scatter, use_container_width=True)
    else:
        st.info("Недостаточно данных для 3D Scatter (нужно ≥10 отчётов с возрастом).")

    st.divider()

    # --------------------------------------------------------------------------
    # 2. 3D Surface Plot: Co-occurrence matrix of reactions (top 10)
    # --------------------------------------------------------------------------
    st.markdown("##### 2. 3D Surface: Co-occurrence of reactions (top 10)")
    top_n_surface = 10
    top_reactions_surface = df_all['reaction'].value_counts().head(top_n_surface).index.tolist()
    if len(top_reactions_surface) >= 3:
        # Строим матрицу ко-встречаемости (аналогично advanced_analytics.plot_cooccurrence_heatmap)
        cooc = pd.DataFrame(0.0, index=top_reactions_surface, columns=top_reactions_surface, dtype=float)
        for pid, g in df_all.groupby('report_id'):
            rxns = g['reaction'].unique()
            for i, r1 in enumerate(rxns):
                if r1 not in top_reactions_surface: continue
                for r2 in rxns[i+1:]:
                    if r2 not in top_reactions_surface: continue
                    cooc.loc[r1, r2] += 1
                    cooc.loc[r2, r1] += 1
        if cooc.sum().sum() > 0:
            fig_surface = go.Figure(data=[go.Surface(
                z=cooc.values,
                x=list(range(top_n_surface)),
                y=list(range(top_n_surface)),
                colorscale='Reds',
                colorbar=dict(title='Co-occurrence')
            )])
            # Добавляем метки осей
            fig_surface.update_layout(
                title='3D Surface of reaction co-occurrence (absolute counts)',
                scene=dict(
                    xaxis=dict(title='Reaction', tickmode='array', tickvals=list(range(top_n_surface)), ticktext=top_reactions_surface, tickangle=45),
                    yaxis=dict(title='Reaction', tickmode='array', tickvals=list(range(top_n_surface)), ticktext=top_reactions_surface),
                    zaxis=dict(title='Count')
                ),
                height=600
            )
            st.plotly_chart(fig_surface, use_container_width=True)
        else:
            st.info("Нет совместных пар реакций для построения поверхности.")
    else:
        st.info(f"Недостаточно уникальных реакций для 3D Surface (нужно ≥3, найдено {len(top_reactions_surface)}).")

    st.divider()

    # --------------------------------------------------------------------------
    # 3. 3D Bubble Chart: ROR signals (ROR, n, % of drug reports)
    # --------------------------------------------------------------------------
    st.markdown("##### 3. 3D Signal Detection: ROR vs n vs % of drug reports")
    if not df_signals.empty:
        # Используем те же фильтры, что и во вкладке Signal Detection (можно добавить свои слайдеры)
        with st.expander("⚙️ Фильтры для сигналов (опционально)"):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                min_n_3d = st.slider("Минимальное n", 1, 50, 3, key="3d_min_n")
            with col_f2:
                min_ror_3d = st.slider("Минимальный ROR", 1.0, 20.0, 2.0, step=0.5, key="3d_min_ror")
        sig_3d = df_signals[(df_signals['n'] >= min_n_3d) & (df_signals['ror'] >= min_ror_3d)]
        if not sig_3d.empty:
            fig_3d_bubble = px.scatter_3d(
                sig_3d,
                x='n',
                y='ror',
                z='pct_of_drug',
                color='drug_name',
                size='n',
                hover_data=['reaction'],
                labels={'n': 'Number of reports', 'ror': 'ROR', 'pct_of_drug': '% of drug reports'},
                title='3D Signal Detection: ROR vs n vs % of drug reports'
            )
            fig_3d_bubble.update_layout(height=600)
            st.plotly_chart(fig_3d_bubble, use_container_width=True)
        else:
            st.info("Нет сигналов, удовлетворяющих выбранным критериям.")
    else:
        st.warning("Файл signal_stats.csv не найден. Запустите parser.py.")

    st.divider()

    # --------------------------------------------------------------------------
    # 4. 3D Bar Chart: Top 10 drugs × age groups × serious (пример)
    # --------------------------------------------------------------------------
    st.markdown("##### 4. 3D Bar: Top drugs by age group and seriousness")
    # Готовим данные: для топ-5 препаратов, возрастные группы, серьёзность
    top_drugs_3d = df_all['drug_name'].value_counts().head(5).index.tolist()
    age_order = ["0–17", "18–44", "45–64", "65+"]
    df_3d_bar = df_all[df_all['drug_name'].isin(top_drugs_3d) & df_all['age_bucket'].isin(age_order)]
    if not df_3d_bar.empty:
        grouped = df_3d_bar.groupby(['drug_name', 'age_bucket', 'serious']).size().reset_index(name='count')
        # Создаём 3D Bar через Mesh3d (проще всего через Scatter3d с маркерами в виде столбцов, но лучше использовать bar3d, которого нет в plotly express)
        # Альтернатива: группировка и объёмные маркеры
        fig_3d_bar = px.scatter_3d(
            grouped,
            x='drug_name',
            y='age_bucket',
            z='count',
            color='serious',
            size='count',
            symbol='serious',
            title='Top 5 drugs: #reports by age group & seriousness',
            labels={'drug_name': 'Drug', 'age_bucket': 'Age group', 'count': 'Number of reports'}
        )
        st.plotly_chart(fig_3d_bar, use_container_width=True)
    else:
        st.info("Недостаточно данных для 3D Bar (топ-5 препаратов и возрастные группы).")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — PROGNOSIS & DRUG COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab_prognosis:
    st.subheader(" Прогнозирование сезонности и сравнение эффективности лекарств")
    st.caption("Анализ временны́х трендов заболеваний и качества лечения препаратами.")


    # --------------------------------------------------------------------------
    # Загрузка дополнительных данных (patient_info.json и drug_info.json)
    # --------------------------------------------------------------------------
    @st.cache_data
    def load_patient_info():
        path = "patient_info.json"
        if not os.path.exists(path):
            st.error("Файл patient_info.json не найден. Запустите генератор данных.")
            return pd.DataFrame()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)


    @st.cache_data
    def load_drug_info():
        path = "drug_info.json"
        if not os.path.exists(path):
            st.error("Файл drug_info.json не найден. Запустите генератор данных.")
            return pd.DataFrame()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)


    df_patients = load_patient_info()
    df_drugs = load_drug_info()

    if df_patients.empty or df_drugs.empty:
        st.warning(
            "Не удалось загрузить данные для прогнозирования. Пожалуйста, сгенерируйте JSON-файлы с полями 'diagnosis' и 'outcome'.")
    else:
        # Объединяем пациентов с препаратами (по drug_id)
        df_merged = df_patients.merge(df_drugs, on="drug_id", how="left")

        # Преобразуем даты
        df_merged["start_date"] = pd.to_datetime(df_merged["start_date"])
        df_merged["recovery_date"] = pd.to_datetime(df_merged["recovery_date"])
        df_merged["month"] = df_merged["start_date"].dt.to_period("M").astype(str)
        df_merged["season"] = df_merged["start_date"].dt.month.map(
            lambda m: "Зима" if m in [12, 1, 2] else (
                "Весна" if m in [3, 4, 5] else ("Лето" if m in [6, 7, 8] else "Осень"))
        )

        # -------------------- 1. Прогнозирование сезонности --------------------
        st.markdown("### ️ Сезонность и прогноз заболеваний")

        # Выбор заболевания (диагноза)
        diagnoses = sorted(df_merged["diagnosis"].unique())
        default_diag = "Гипертоническая болезнь" if "Гипертоническая болезнь" in diagnoses else diagnoses[0]
        selected_diagnosis = st.selectbox("Выберите заболевание для анализа сезонности:", diagnoses,
                                          index=diagnoses.index(default_diag) if default_diag in diagnoses else 0)

        # Группировка по месяцам
        monthly_counts = df_merged[df_merged["diagnosis"] == selected_diagnosis].groupby("month").size().reset_index(
            name="count")
        monthly_counts["month_dt"] = pd.to_datetime(monthly_counts["month"] + "-01")
        monthly_counts = monthly_counts.sort_values("month_dt")

        if len(monthly_counts) < 3:
            st.info("Недостаточно данных для построения временного ряда.")
        else:
            # Анимированный график (накопление по месяцам)
            fig_season = px.bar(
                monthly_counts,
                x="month",
                y="count",
                title=f"Количество случаев '{selected_diagnosis}' по месяцам",
                labels={"month": "Месяц", "count": "Число пациентов"},
                color="count",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_season, use_container_width=True)

            # Простой прогноз: скользящее среднее + линейная экстраполяция на 3 месяца
            st.markdown("####  Прогноз на следующие 3 месяца")
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing

                # Подготовка временного ряда (индекс - дата, значения - количество)
                ts = monthly_counts.set_index("month_dt")["count"].asfreq("MS").fillna(0)
                if len(ts) >= 6:
                    model = ExponentialSmoothing(ts, trend="add", seasonal=None).fit()
                    forecast = model.forecast(3)
                    forecast_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=3, freq="MS")
                    forecast_df = pd.DataFrame({"month": forecast_dates.strftime("%Y-%m"), "count": forecast.values})

                    # Визуализация истории + прогноза
                    fig_forecast = px.line(
                        pd.concat([monthly_counts, forecast_df], ignore_index=True),
                        x="month",
                        y="count",
                        markers=True,
                        title=f"Прогноз для '{selected_diagnosis}' (Holt-Winters)",
                        labels={"month": "Месяц", "count": "Прогнозируемое число случаев"}
                    )
                    # Добавляем разделитель
                    fig_forecast.add_vline(x=monthly_counts["month"].iloc[-1], line_dash="dash", line_color="red")
                    st.plotly_chart(fig_forecast, use_container_width=True)
                else:
                    st.info("Недостаточно месяцев для статистического прогноза (нужно ≥6).")
            except Exception as e:
                st.warning(f"Прогноз не удался: {e}. Показываем простую экстраполяцию.")
                # fallback: линейная регрессия через numpy
                if len(monthly_counts) >= 3:
                    x = np.arange(len(monthly_counts))
                    y = monthly_counts["count"].values
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_pred = np.arange(len(monthly_counts), len(monthly_counts) + 3)
                    y_pred = p(x_pred)
                    pred_months = [f"Месяц +{i + 1}" for i in range(3)]
                    forecast_df = pd.DataFrame({"month": pred_months, "count": y_pred})
                    fig_forecast2 = px.line(
                        pd.concat([monthly_counts.assign(type="История"), forecast_df.assign(type="Прогноз")]),
                        x="month", y="count", color="type", markers=True,
                        title=f"Линейный прогноз для '{selected_diagnosis}'"
                    )
                    st.plotly_chart(fig_forecast2, use_container_width=True)

        st.divider()

        # -------------------- 2. Сравнение лекарств по качеству лечения --------------------
        st.markdown("###  Сравнение эффективности препаратов")
        st.caption(
            "Оценка качества лечения на основе исхода (outcome) из drug_info. Для каждого препарата и диагноза вычисляется доля положительных исходов.")

        # Фильтр по диагнозу
        compare_diagnosis = st.selectbox("Выберите диагноз для сравнения препаратов:", diagnoses, key="compare_diag")
        # Минимальное количество наблюдений для включения препарата
        min_patients = st.slider("Минимальное число пациентов на препарат:", 5, 100, 20, step=5)

        # Отбираем данные для выбранного диагноза
        diag_data = df_merged[df_merged["diagnosis"] == compare_diagnosis].copy()
        # Определяем положительный исход (улучшение, стабилизация, ремиссия, купирование симптомов)
        positive_outcomes = ["Улучшение", "Стабилизация", "Ремиссия", "Купирование симптомов"]
        diag_data["positive"] = diag_data["outcome"].apply(lambda x: 1 if x in positive_outcomes else 0)

        # Агрегация по препаратам
        drug_stats = diag_data.groupby("drug_name").agg(
            total_patients=("patient_id", "count"),
            positive_count=("positive", "sum")
        ).reset_index()
        drug_stats = drug_stats[drug_stats["total_patients"] >= min_patients]
        drug_stats["efficacy"] = (drug_stats["positive_count"] / drug_stats["total_patients"]) * 100

        if drug_stats.empty:
            st.info(f"Нет препаратов с ≥{min_patients} пациентами для диагноза '{compare_diagnosis}'.")
        else:
            # Сортировка по эффективности
            drug_stats = drug_stats.sort_values("efficacy", ascending=False)
            top_n = st.slider("Показать топ N препаратов:", 5, min(50, len(drug_stats)), 15)
            top_drugs = drug_stats.head(top_n)

            fig_eff = px.bar(
                top_drugs,
                x="efficacy",
                y="drug_name",
                orientation="h",
                color="efficacy",
                color_continuous_scale="Viridis",
                text="efficacy",
                title=f"Эффективность препаратов при '{compare_diagnosis}' (положительный исход: {', '.join(positive_outcomes)})",
                labels={"efficacy": "Доля положительных исходов (%)", "drug_name": "Препарат"}
            )
            fig_eff.update_traces(texttemplate='%{text:.1f}%', textposition="outside")
            fig_eff.update_layout(height=600, coloraxis_showscale=False)
            st.plotly_chart(fig_eff, use_container_width=True)

            # Таблица с деталями
            st.markdown("####  Детальная таблица эффективности")
            st.dataframe(
                drug_stats.head(top_n).rename(columns={
                    "drug_name": "Препарат",
                    "total_patients": "Всего пациентов",
                    "positive_count": "Положительных исходов",
                    "efficacy": "Эффективность (%)"
                }).reset_index(drop=True),
                use_container_width=True
            )

            # Дополнительно: пузырьковая диаграмма (эффективность vs количество пациентов)
            fig_bubble = px.scatter(
                drug_stats,
                x="total_patients",
                y="efficacy",
                size="total_patients",
                color="drug_name",
                hover_name="drug_name",
                title="Сравнение: число пациентов vs эффективность",
                labels={"total_patients": "Количество пациентов", "efficacy": "Эффективность (%)"}
            )
            fig_bubble.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig_bubble, use_container_width=True)