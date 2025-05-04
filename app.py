
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Electricity Consumption Forecast (Côte d'Ivoire)", layout="centered")

st.title("🔌 Electricity Bill Forecast (Bi-monthly)")

st.markdown("""
Upload your bi-monthly electricity usage in **kWh**. The app will forecast your future usage and help you plan ahead.

**Expected Format**:
- `ds`: Date of the start of billing period (YYYY-MM-DD)
- `y`: Consumption in kWh for the 2-month period

Example:
```
ds,y
2022-01-01,310
2022-03-01,360
...
```
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    try:
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])

        st.success("✅ Data loaded successfully!")
        st.dataframe(df)

        # Forecast with Prophet
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=3, freq='2MS')  # 3 bi-monthly periods (6 months)
        forecast = model.predict(future)

        st.subheader("📈 Forecast Plot")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("📊 Forecasted Values")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3))

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("📂 Please upload a CSV file to begin.")
