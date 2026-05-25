"""
Tech Challenge - Fase 4
App preditivo de Obesidade (Streamlit)

Como rodar localmente:
    pip install -r requirements.txt
    streamlit run app.py

Requer o arquivo `obesity_model.joblib` (gerado pelo notebook obesity_ml.ipynb)
no mesmo diretorio.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "obesity_model.joblib"

st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Sistema Preditivo de Obesidade")
st.caption(
    "Tech Challenge - Fase 4 | Modelo de Machine Learning para apoio à decisão clínica."
)

if not MODEL_PATH.exists():
    st.error(
        f"Modelo não encontrado em {MODEL_PATH}. "
        "Execute o notebook `obesity_ml.ipynb` para gerar `obesity_model.joblib`."
    )
    st.stop()


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


model = load_model(MODEL_PATH)

CLASS_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]
CLASS_LABELS_PT = {
    "Insufficient_Weight": "Abaixo do peso",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso I",
    "Overweight_Level_II": "Sobrepeso II",
    "Obesity_Type_I": "Obesidade Tipo I",
    "Obesity_Type_II": "Obesidade Tipo II",
    "Obesity_Type_III": "Obesidade Tipo III",
}

st.subheader("Dados do paciente")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gênero", ["Female", "Male"])
    age = st.number_input("Idade", 14, 90, 25)
    height = st.number_input("Altura (m)", 1.30, 2.20, 1.70, step=0.01)
    weight = st.number_input("Peso (kg)", 30.0, 250.0, 70.0, step=0.5)
    family_history = st.selectbox(
        "Histórico familiar de excesso de peso", ["yes", "no"]
    )
    favc = st.selectbox("Consumo frequente de alimentos calóricos (FAVC)", ["yes", "no"])
    fcvc = st.slider("Frequência de consumo de vegetais (FCVC)", 1, 3, 2)
    ncp = st.slider("Refeições principais por dia (NCP)", 1, 4, 3)
with col2:
    caec = st.selectbox(
        "Lanches entre refeições (CAEC)",
        ["no", "Sometimes", "Frequently", "Always"],
        index=1,
    )
    smoke = st.selectbox("Fuma? (SMOKE)", ["yes", "no"], index=1)
    ch2o = st.slider("Consumo diário de água (CH2O)", 1, 3, 2)
    scc = st.selectbox("Monitora calorias (SCC)", ["yes", "no"], index=1)
    faf = st.slider("Frequência de atividade física (FAF)", 0, 3, 1)
    tue = st.slider("Tempo em dispositivos eletrônicos (TUE)", 0, 2, 1)
    calc = st.selectbox(
        "Consumo de álcool (CALC)",
        ["no", "Sometimes", "Frequently", "Always"],
        index=1,
    )
    mtrans = st.selectbox(
        "Meio de transporte (MTRANS)",
        ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"],
    )

bmi = weight / (height ** 2)
st.metric("IMC calculado", f"{bmi:.2f} kg/m²")

input_df = pd.DataFrame(
    [
        {
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history": family_history,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans,
            "BMI": bmi,
        }
    ]
)

st.divider()
if st.button("🔍 Predizer nível de obesidade", type="primary", use_container_width=True):
    pred = model.predict(input_df)[0]
    st.success(f"**Predição:** {CLASS_LABELS_PT.get(pred, pred)} (`{pred}`)")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        classes = model.classes_
        proba_df = (
            pd.DataFrame({"Classe": classes, "Probabilidade": proba})
            .assign(
                Classe_PT=lambda d: d["Classe"].map(CLASS_LABELS_PT).fillna(d["Classe"])
            )
            .sort_values("Probabilidade", ascending=False)
        )
        st.subheader("Probabilidades por classe")
        st.bar_chart(proba_df.set_index("Classe_PT")["Probabilidade"])
        st.dataframe(
            proba_df[["Classe_PT", "Probabilidade"]].rename(
                columns={"Classe_PT": "Classe"}
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "⚠️ Esta predição é uma ferramenta de apoio e não substitui avaliação médica."
    )
