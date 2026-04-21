import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
import plotly.figure_factory as ff


def _empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14)
    )
    fig.update_layout(height=400, xaxis_visible=False, yaxis_visible=False)
    return fig


def plot_cooccurrence_heatmap(df, top_n=15, normalize=False):
    if df.empty:
        return _empty_figure("No data")

    top_reactions = df['reaction'].value_counts().head(top_n).index.tolist()
    if len(top_reactions) < 2:
        return _empty_figure(f"Need ≥2 reactions, found {len(top_reactions)}")

    # Матрица с плавающей точкой
    cooc = pd.DataFrame(0.0, index=top_reactions, columns=top_reactions, dtype=float)
    patient_counts = {rxn: set() for rxn in top_reactions}

    for pid, g in df.groupby('report_id'):
        rxns = g['reaction'].unique()
        for i, r1 in enumerate(rxns):
            if r1 not in top_reactions:
                continue
            patient_counts[r1].add(pid)
            for r2 in rxns[i + 1:]:
                if r2 not in top_reactions:
                    continue
                cooc.loc[r1, r2] += 1
                cooc.loc[r2, r1] += 1

    if cooc.sum().sum() == 0:
        return _empty_figure("No co-occurring pairs (each report has only one reaction?)")

    if normalize:
        for r1 in top_reactions:
            for r2 in top_reactions:
                if r1 == r2:
                    cooc.loc[r1, r2] = 1.0
                    continue
                n1 = len(patient_counts[r1])
                n2 = len(patient_counts[r2])
                denom = n1 + n2 - cooc.loc[r1, r2]
                if denom > 0:
                    cooc.loc[r1, r2] = cooc.loc[r1, r2] / denom
                else:
                    cooc.loc[r1, r2] = 0.0
        title = f'Co-occurrence (Jaccard index) of reactions (top {top_n})'
        color_scale = 'Blues'
    else:
        title = f'Co-occurrence (absolute count) of reactions (top {top_n})'
        color_scale = 'Reds'

    fig = px.imshow(cooc, text_auto=True, color_continuous_scale=color_scale,
                    title=title, zmin=0)
    fig.update_layout(height=600)
    return fig


def plot_sankey(df, top_n=10):
    if df.empty:
        return _empty_figure("No data")

    top_cong = df['congenital_diseases'].value_counts().head(top_n).index
    top_drugs = df['drug_name'].value_counts().head(top_n).index
    top_rxns = df['reaction'].value_counts().head(top_n).index

    subset = df[df['congenital_diseases'].isin(top_cong) &
                df['drug_name'].isin(top_drugs) &
                df['reaction'].isin(top_rxns)]

    if subset.empty:
        return _empty_figure("Not enough data for Sankey (try lower top_n)")

    agg = subset.groupby(['congenital_diseases', 'drug_name', 'reaction'])['report_id'].nunique().reset_index()
    nodes = list(set(agg['congenital_diseases']) | set(agg['drug_name']) | set(agg['reaction']))
    node_dict = {n: i for i, n in enumerate(nodes)}

    links = []
    for _, row in agg.iterrows():
        links.append({
            'source': node_dict[row['congenital_diseases']],
            'target': node_dict[row['drug_name']],
            'value': row['report_id']
        })
        links.append({
            'source': node_dict[row['drug_name']],
            'target': node_dict[row['reaction']],
            'value': row['report_id']
        })

    if not links:
        return _empty_figure("No links generated")

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=nodes, pad=15, thickness=20),
        link=dict(
            source=[l['source'] for l in links],
            target=[l['target'] for l in links],
            value=[l['value'] for l in links]
        )
    )])
    fig.update_layout(title='Sankey: Congenital Disease → Drug → Reaction', height=600)
    return fig


def plot_world_map(df):
    if df.empty:
        return _empty_figure("No data")

    country_counts = df['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']

    if len(country_counts) < 2:
        return _empty_figure(f"Only {len(country_counts)} country, need at least 2 for map")

    fig = px.choropleth(
        country_counts,
        locations='country',
        locationmode='country names',
        color='count',
        hover_name='country',
        color_continuous_scale='Blues',
        title='ADR reports by country'
    )
    fig.update_layout(height=500)
    return fig


def plot_dendrogram(df, top_drugs=20, top_reactions=20, normalize=True):
    drug_list = df['drug_name'].value_counts().head(top_drugs).index
    rxn_list = df['reaction'].value_counts().head(top_reactions).index
    filtered = df[df['drug_name'].isin(drug_list) & df['reaction'].isin(rxn_list)]
    pivot = filtered.groupby(['drug_name', 'reaction']).size().unstack(fill_value=0)

    if pivot.shape[0] < 2:
        return _empty_figure(f"Need ≥2 drugs, found {pivot.shape[0]}. Try increasing top_drugs or top_reactions.")

    if normalize:
        pivot = pivot.loc[pivot.sum(axis=1) > 0]
        if pivot.shape[0] < 2:
            return _empty_figure("After removing drugs with zero reactions, less than 2 drugs remain. Try increasing top_reactions or disable normalize.")
        pivot = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)

    pivot = pivot.loc[~(pivot.sum(axis=1) == 0)]
    if pivot.shape[0] < 2:
        return _empty_figure("After removing zero-row drugs, less than 2 drugs remain.")

    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage

    def distfun(x):
        return pdist(x, metric='cosine')
    def linkagefun(x):
        return linkage(x, method='average')

    try:
        # Убрали параметр colorscale, чтобы избежать ошибки валидации
        fig = ff.create_dendrogram(
            pivot.values,
            labels=pivot.index.tolist(),
            orientation='left',
            distfun=distfun,
            linkagefun=linkagefun
        )
        fig.update_layout(
            title=f'Drug clustering by reaction profile (top {top_drugs} drugs, top {top_reactions} reactions)',
            height=600,
            width=700,
            xaxis_title='Cosine distance'
        )
        return fig
    except Exception as e:
        return _empty_figure(f"Clustering error: {e}")

def plot_outcome_by_age_serious(df):
    if df.empty:
        return _empty_figure("No data")

    available_outcomes = df['outcome'].dropna().unique()
    if len(available_outcomes) == 0:
        return _empty_figure("No outcome data available")

    # Желаемые исходы, если есть
    desired = ['Fatal', 'Not Recovered', 'Recovered/Resolved']
    outcomes_to_plot = [o for o in desired if o in available_outcomes]
    if not outcomes_to_plot:
        # Берём все, кроме Unknown, если возможно
        outcomes_to_plot = [o for o in available_outcomes if o != 'Unknown']
        if not outcomes_to_plot:
            outcomes_to_plot = available_outcomes.tolist()

    filtered = df[df['outcome'].isin(outcomes_to_plot)].copy()
    if filtered.empty:
        return _empty_figure(f"No data for outcomes {outcomes_to_plot}")

    # Группировка
    grouped = filtered.groupby(['age_bucket', 'serious', 'outcome']).size().reset_index(name='count')
    total_per_group = grouped.groupby(['age_bucket', 'serious'])['count'].transform('sum')
    grouped['percent'] = grouped['count'] / total_per_group * 100

    fig = px.bar(
        grouped,
        x='age_bucket',
        y='percent',
        color='outcome',
        facet_col='serious',
        barmode='stack',
        title='Outcome by age group and seriousness (percent)',
        labels={'percent': 'Percentage (%)', 'age_bucket': 'Age group'}
    )
    fig.update_layout(height=500)
    return fig


def plot_reaction_trend(df, reaction_name):
    if df.empty:
        return _empty_figure("No data")

    trend = df[df['reaction'] == reaction_name].groupby('receive_month').size().reset_index(name='count')
    if trend.empty:
        return _empty_figure(f"No data for reaction '{reaction_name}'")

    fig = px.line(
        trend,
        x='receive_month',
        y='count',
        markers=True,
        title=f'Monthly trend for "{reaction_name}"',
        labels={'count': 'Number of reports', 'receive_month': 'Month'}
    )
    fig.update_layout(height=450)
    return fig