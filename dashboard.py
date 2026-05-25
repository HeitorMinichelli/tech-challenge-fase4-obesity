"""
Tech Challenge - Fase 4
Painel analítico de Obesidade (Streamlit)

Como rodar localmente:
    pip install -r requirements.txt
    streamlit run dashboard.py

Requer o arquivo `Obesity.csv` no mesmo diretorio.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

CSV_PATH = Path(__file__).parent / "Obesity.csv"

st.set_page_config(
    page_title="Painel Analítico - Obesidade",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Painel Analítico - Obesidade")
st.caption("Insights sobre o dataset de obesidade para a equipe médica.")

if not CSV_PATH.exists():
    st.error(f"Arquivo não encontrado: {CSV_PATH}")
    st.stop()


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["FCVC", "NCP", "CH2O", "FAF", "TUE"]:
        df[c] = df[c].round().astype(int)
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    return df


df = load_data(CSV_PATH)

CLASS_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

# -------------------- Filtros --------------------
st.sidebar.header("Filtros")
gender_sel = st.sidebar.multiselect(
    "Gênero", df["Gender"].unique(), default=list(df["Gender"].unique())
)
age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_sel = st.sidebar.slider("Idade", age_min, age_max, (age_min, age_max))
fh_sel = st.sidebar.multiselect(
    "Histórico familiar",
    df["family_history"].unique(),
    default=list(df["family_history"].unique()),
)

mask = (
    df["Gender"].isin(gender_sel)
    & df["Age"].between(age_sel[0], age_sel[1])
    & df["family_history"].isin(fh_sel)
)
dff = df[mask].copy()

if dff.empty:
    st.warning("Nenhum registro corresponde aos filtros selecionados. Ajuste os filtros na barra lateral.")
    st.stop()


def stacked_pct(data: pd.DataFrame, group_col: str, order: list | None = None) -> pd.DataFrame:
    ct = pd.crosstab(data[group_col], data["Obesity"], normalize="index") * 100
    if order is not None:
        ct = ct.reindex(index=order)
    ct = ct.reindex(columns=CLASS_ORDER, fill_value=0).dropna(how="all")
    return ct.reset_index().melt(id_vars=group_col, var_name="Obesity", value_name="%")


# -------------------- KPIs --------------------
total = len(dff)
pct_obese = (
    dff["Obesity"].str.startswith("Obesity").mean() * 100 if total else 0
)
pct_overweight = (
    dff["Obesity"].str.startswith("Overweight").mean() * 100 if total else 0
)
avg_bmi = dff["BMI"].mean() if total else 0
pct_family = (dff["family_history"].eq("yes").mean() * 100) if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pacientes", f"{total:,}")
c2.metric("% Obesos", f"{pct_obese:.1f}%")
c3.metric("% Sobrepeso", f"{pct_overweight:.1f}%")
c4.metric("IMC médio", f"{avg_bmi:.2f}")

st.divider()

# -------------------- Distribuição da classe --------------------
left, right = st.columns(2)
with left:
    st.subheader("Distribuição do nível de obesidade")
    counts = (
        dff["Obesity"]
        .value_counts()
        .reindex(CLASS_ORDER)
        .dropna()
        .reset_index()
    )
    counts.columns = ["Obesity", "Count"]
    fig = px.bar(
        counts,
        x="Count",
        y="Obesity",
        orientation="h",
        color="Obesity",
        category_orders={"Obesity": CLASS_ORDER},
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("IMC por nível de obesidade")
    fig = px.box(
        dff,
        x="Obesity",
        y="BMI",
        color="Obesity",
        category_orders={"Obesity": CLASS_ORDER},
    )
    fig.update_layout(showlegend=False, height=400, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Fatores comportamentais --------------------
st.subheader("Fatores comportamentais vs Obesidade")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Histórico familiar", "Atividade física", "Alimentação", "Transporte"]
)

with tab1:
    fig = px.bar(
        stacked_pct(dff, "family_history"),
        x="family_history",
        y="%",
        color="Obesity",
        category_orders={"Obesity": CLASS_ORDER},
        barmode="stack",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Pacientes com histórico familiar têm proporção significativamente maior de obesidade."
    )

with tab2:
    fig = px.box(
        dff,
        x="FAF",
        y="BMI",
        color="Obesity",
        category_orders={"Obesity": CLASS_ORDER},
    )
    fig.update_layout(height=420, xaxis_title="FAF (frequência atividade física)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Maior frequência de atividade física correlaciona com IMC mais baixo.")

with tab3:
    colA, colB = st.columns(2)
    with colA:
        fig = px.bar(
            stacked_pct(dff, "FAVC"),
            x="FAVC",
            y="%",
            color="Obesity",
            category_orders={"Obesity": CLASS_ORDER},
            barmode="stack",
            title="FAVC (alimentos calóricos)",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig = px.bar(
            stacked_pct(dff, "CAEC", order=["no", "Sometimes", "Frequently", "Always"]),
            x="CAEC",
            y="%",
            color="Obesity",
            category_orders={"Obesity": CLASS_ORDER},
            barmode="stack",
            title="CAEC (lanches entre refeições)",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    fig = px.bar(
        stacked_pct(dff, "MTRANS"),
        x="MTRANS",
        y="%",
        color="Obesity",
        category_orders={"Obesity": CLASS_ORDER},
        barmode="stack",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Transportes ativos (Walking/Bike) tendem a ter menor incidência de obesidade severa."
    )

# -------------------- Idade x IMC --------------------
st.subheader("Idade x IMC")
fig = px.scatter(
    dff,
    x="Age",
    y="BMI",
    color="Obesity",
    category_orders={"Obesity": CLASS_ORDER},
    opacity=0.6,
)
fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)

# -------------------- Tabela --------------------
with st.expander("Ver dados filtrados"):
    st.dataframe(dff, use_container_width=True, hide_index=True)
