import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="Prévision Électrique (Côte d'Ivoire)", layout="centered")

st.title("🔌 Prévision de Factures d'Électricité")
st.markdown("""
Veuillez saisir vos **dernières factures d'électricité bimensuelles** pour obtenir une prévision des 3 prochaines factures.

Chaque facture doit inclure :
- La date de début de la période de facturation (ex : 2023-09-01)
- La consommation totale en kWh pour cette période de 2 mois
""")

# Tarifs CIE estimés par paliers (simplifiés)
def calcul_cout_cie(kwh):
    if kwh <= 110:
        return kwh * 42
    elif kwh <= 300:
        return 110 * 42 + (kwh - 110) * 72
    else:
        return 110 * 42 + 190 * 72 + (kwh - 300) * 115

# Interface mobile-friendly + session state
if 'rows' not in st.session_state:
    st.session_state.rows = 3

# Bouton d'ajout de facture placé en haut pour éviter de masquer le formulaire
if st.button("➕ Ajouter une facture"):
    st.session_state.rows += 1
    st.experimental_rerun()

st.subheader("🧾 Saisissez Vos Factures")

# Formulaire principal
with st.form("billing_form"):
    dates = []
    usages = []
    for i in range(st.session_state.rows):
        st.markdown(f"**Facture {i+1}**")
        col1, col2 = st.columns(2)
        date_val = col1.date_input("Date de début de période", key=f"ds_{i}", value=date(2025, 1, 1))
        kwh_val = col2.number_input("Consommation (kWh)", min_value=0.0, step=1.0, key=f"y_{i}")
        dates.append(date_val)
        usages.append(kwh_val)

    submitted = st.form_submit_button("Générer la Prévision")

if submitted:
    try:
        df = pd.DataFrame({
            'ds': pd.to_datetime(dates),
            'y': usages
        })
        df['Coût estimé (FCFA)'] = df['y'].apply(calcul_cout_cie)

        st.success("✅ Données enregistrées. Génération de la prévision en cours...")
        st.markdown("### 📋 Données saisies")
        st.dataframe(df.rename(columns={
            'ds': 'Période',
            'y': 'Consommation (kWh)'
        }))

        model = Prophet()
        model.fit(df[['ds', 'y']])

        future = model.make_future_dataframe(periods=3, freq='2MS')
        forecast = model.predict(future)

        st.subheader("📈 Graphique de Prévision")
        fig1 = model.plot(forecast)
        plt.xlabel("Période")
        plt.ylabel("Consommation (kWh)")
        st.pyplot(fig1)

        st.subheader("📊 Valeurs Prévisionnelles")
        preview = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3).copy()
        preview['Coût estimé (FCFA)'] = preview['yhat'].apply(calcul_cout_cie).round(0)
        st.dataframe(preview.rename(columns={
            'ds': 'Période Prévue',
            'yhat': 'Consommation Prévue (kWh)',
            'yhat_lower': 'Borne Inférieure',
            'yhat_upper': 'Borne Supérieure',
            'Coût estimé (FCFA)': 'Coût Estimé (FCFA)'
        }))

    except Exception as e:
        st.error(f"❌ Erreur : {e}")
