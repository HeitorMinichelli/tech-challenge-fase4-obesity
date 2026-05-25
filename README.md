# Tech Challenge - Fase 4: Predição de Obesidade

Pipeline de Machine Learning + App preditivo + Dashboard analítico.

## Estrutura

- `obesity_ml.ipynb` — Notebook com EDA, feature engineering e treino do modelo. Gera `obesity_model.joblib`.
- `app.py` — Aplicação Streamlit preditiva (entrada de dados do paciente → predição).
- `dashboard.py` — Painel analítico Streamlit com insights sobre o dataset.
- `Obesity.csv` — Dataset.
- `requirements.txt` — Dependências Python.

## Setup (VSCode)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Execução

1. **Treinar modelo** — abrir `obesity_ml.ipynb` no VSCode e executar todas as células (gera `obesity_model.joblib`).
2. **App preditivo**:
   ```bash
   streamlit run app.py
   ```
3. **Dashboard analítico**:
   ```bash
   streamlit run dashboard.py
   ```

## Deploy (Streamlit Community Cloud)

1. Suba o repositório no GitHub (incluindo `Obesity.csv` e `obesity_model.joblib`).
2. Em https://share.streamlit.io conecte o repo e aponte para `app.py` (e/ou `dashboard.py`).
3. Compartilhe o link público gerado.
"# tech-challenge-fase4-obesity" 
"# tech-challenge-fase4-obesity" 
