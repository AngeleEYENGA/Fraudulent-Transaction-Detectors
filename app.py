# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.fraud_rules import enrich_transactions, ml_anomaly_score, combined_score
from streamlit_folium import st_folium
import folium
from datetime import datetime
import base64

# --- layout / style
st.set_page_config(page_title="Fraudulent Transaction Detector - MAEA Tech", layout="wide")
PRIMARY = "#ffffff"  # change si tu veux
ACCENT = "#0A2540"

# header with logo
col1, col2 = st.columns([1, 6])
with col1:
    try:
        st.image("assets/logo.png", width=120)
    except:
        st.markdown("<div style='font-weight:bold'>MAEA Tech</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<h1 style='color:{ACCENT};margin-bottom:0'>Fraudulent Transaction Detector</h1>", unsafe_allow_html=True)
    st.markdown("**Prototype** · Détection de transactions suspectes · MAEA Tech", unsafe_allow_html=True)

st.markdown("---")

# upload / sample
st.sidebar.header("Options")
use_sample = st.sidebar.checkbox("Utiliser le dataset factice (ex: sample_transactions.csv)", value=True)
uploaded = st.sidebar.file_uploader("Ou uploade un fichier CSV de transactions", type=["csv"])

if use_sample:
    df = pd.read_csv("data/sample_transactions.csv")
elif uploaded:
    df = pd.read_csv(uploaded)
else:
    st.warning("Choisis un dataset (sample ou upload).")
    st.stop()

# Enrich & compute
df = enrich_transactions(df)
df = ml_anomaly_score(df)
df = combined_score(df)

# Top KPIs
st.subheader("Aperçu global")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Transactions", f"{len(df)}")
k2.metric("Alertes élevées", f"{(df['risk_label']=='Élevé').sum()}")
k3.metric("Montant moyen", f"{df['amount'].mean():.2f}")
k4.metric("Transactions anormales (ML)", f"{(df['ml_pred']==-1).sum()}")

st.markdown("### Graphiques")
# time series: total amount per day
df['date'] = df['timestamp'].dt.date
agg = df.groupby('date')['amount'].sum().reset_index()
fig_ts = px.bar(agg, x='date', y='amount', title="Montant total par jour")
st.plotly_chart(fig_ts, use_container_width=True)

# risk distribution
fig_pie = px.pie(df, names='risk_label', title="Répartition du risque")
st.plotly_chart(fig_pie, use_container_width=True)

# table of suspicious transactions
st.markdown("### Transactions suspectes (top)")
top = df.sort_values('final_score', ascending=False).head(20)
st.dataframe(top[['transaction_id','user_id','amount','timestamp','country','final_score','risk_label','rule_reasons']])

# carte (si latitude/longitude non fournies on simule par pays center)
st.markdown("### Carte des transactions (approx.)")
# dictionnaire simple de centres par pays (ajoute/modifie si besoin)
country_coords = {
    "FR": [46.6, 2.4],
    "NG": [9.0, 8.7],
    "RU": [61.5, 105.3],
    "IR": [32.0, 53.0],
    "PK": [30.3753, 69.3451],
    "PK": [30.3753, 69.3451]
}
# create folium map centered in Europe
m = folium.Map(location=[20,0], zoom_start=2)
for _, row in df.iterrows():
    c = row['country']
    latlon = country_coords.get(c, [20, 0])
    color = 'green' if row['final_score'] < 30 else ('orange' if row['final_score'] < 60 else 'red')
    folium.CircleMarker(location=latlon, radius=6, color=color,
                        popup=f"{row['transaction_id']} {row['amount']} {row['risk_label']}").add_to(m)

st_data = st_folium(m, width=700)

# download filtered CSV
st.markdown("---")
st.markdown("### Export")
filtered = st.checkbox("Afficher/exporter uniquement alertes (Élevé)")
if filtered:
    export_df = df[df['risk_label']=='Élevé']
else:
    export_df = df

csv = export_df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
st.markdown("Télécharger le CSV :")
st.markdown(f"[⬇️ Télécharger le rapport CSV](data:text/csv;base64,{b64})", unsafe_allow_html=True)

st.markdown("---")
st.info("Astuce : tu peux uploader un CSV réel et ajuster les règles dans utils/fraud_rules.py pour correspondre à un profil métier.")

