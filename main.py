import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Konfigurasi halaman (harus menjadi perintah Streamlit pertama)
st.set_page_config(page_title="Prediksi Profit Superstore", layout="wide")

# ==============================================================================
# ==> Daftar urutan fitur yang 100% akurat dari notebook Anda <==
MODEL_FEATURE_ORDER = [
    'Quantity', 'Discount', 'Order_Year', 'Order_Month', 'Order_Day', 'Shipping_Duration', 'Price_per_Unit',
    'Ship Mode_Same Day', 'Ship Mode_Second Class', 'Ship Mode_Standard Class', 
    'Segment_Corporate', 'Segment_Home Office', 'Category_Office Supplies', 'Category_Technology', 
    'Sub-Category_Appliances', 'Sub-Category_Art', 'Sub-Category_Binders', 'Sub-Category_Bookcases', 
    'Sub-Category_Chairs', 'Sub-Category_Copiers', 'Sub-Category_Envelopes', 'Sub-Category_Fasteners', 
    'Sub-Category_Furnishings', 'Sub-Category_Labels', 'Sub-Category_Machines', 'Sub-Category_Paper', 
    'Sub-Category_Phones', 'Sub-Category_Storage', 'Sub-Category_Supplies', 'Sub-Category_Tables', 
    'Region_East', 'Region_South', 'Region_West'
]
# ==============================================================================

@st.cache_resource
def load_model():
    """Memuat model yang sudah dilatih"""
    try:
        model = joblib.load('model_profit_paling_optimal.pkl')
        return model
    except FileNotFoundError:
        st.error("File model 'model_profit_paling_optimal.pkl' tidak ditemukan.")
        return None

model = load_model()

# --- Antarmuka Aplikasi ---
st.title('🤖 Aplikasi Analisis & Prediksi Profit Superstore')

tab1, tab2 = st.tabs(["Prediksi per Transaksi", "Prediksi Batch dari File"])

# --- KONTEN TAB 1: PREDIKSI PER TRANSAKSI ---
with tab1:
    st.header("Prediksi untuk Satu Transaksi")
    st.sidebar.header('Parameter Input Tunggal:')
    
    ship_mode_options = ['First Class', 'Same Day', 'Second Class', 'Standard Class']
    segment_options = ['Consumer', 'Corporate', 'Home Office']
    category_options = ['Furniture', 'Office Supplies', 'Technology']
    sub_category_options = ['Bookcases', 'Chairs', 'Labels', 'Tables', 'Storage', 'Furnishings', 'Art', 'Phones', 'Binders', 'Appliances', 'Paper', 'Accessories', 'Envelopes', 'Fasteners', 'Supplies', 'Machines', 'Copiers']
    region_options = ['Central', 'East', 'South', 'West']

    def user_input_features():
        sales = st.sidebar.number_input('Total Penjualan (Sales)', min_value=0.01, value=250.0, format="%.2f", key="single_sales")
        quantity = st.sidebar.number_input('Jumlah Barang (Quantity)', min_value=1, value=5, step=1, key="single_quantity")
        discount = st.sidebar.slider('Diskon', 0.0, 1.0, 0.2, 0.01, key="single_discount")
        shipping_duration = st.sidebar.number_input('Estimasi Durasi Pengiriman (Hari)', min_value=0, value=4, step=1, key="single_shipping")
        ship_mode = st.sidebar.selectbox('Mode Pengiriman', ship_mode_options, key="single_ship_mode")
        segment = st.sidebar.selectbox('Segmen Pelanggan', segment_options, key="single_segment")
        category = st.sidebar.selectbox('Kategori Produk', category_options, key="single_category")
        sub_category = st.sidebar.selectbox('Sub-Kategori Produk', sub_category_options, key="single_sub_category")
        region = st.sidebar.selectbox('Wilayah', region_options, key="single_region")
        
        data = {'Sales': sales, 'Quantity': quantity, 'Discount': discount, 'Shipping_Duration': shipping_duration, 
                'Ship Mode': ship_mode, 'Segment': segment, 'Category': category, 'Sub-Category': sub_category, 'Region': region}
        return data

    input_data = user_input_features()

    if st.button('Prediksi Profit Tunggal'):
        if model is not None:
            input_df = pd.DataFrame([input_data])
            today = datetime.now()
            input_df['Order_Year'], input_df['Order_Month'], input_df['Order_Day'] = today.year, today.month, today.day
            input_df['Price_per_Unit'] = input_df['Sales'] / input_df['Quantity']
            
            processed_df = pd.get_dummies(input_df)
            final_df = processed_df.reindex(columns=MODEL_FEATURE_ORDER, fill_value=0)
            
            prediction = model.predict(final_df)
            
            st.subheader('Hasil Prediksi:')
            st.metric(label="Prediksi Profit", value=f"$ {prediction[0]:,.2f}")

            # --- GRAFIK 1: FEATURE IMPORTANCE (DITAMBAHKAN KEMBALI) ---
            st.subheader("Faktor yang Paling Mempengaruhi Prediksi")
            importances = model.feature_importances_
            importance_df = pd.DataFrame({'feature': MODEL_FEATURE_ORDER, 'importance': importances}).sort_values('importance', ascending=False).head(10)
            st.bar_chart(importance_df.set_index('feature'))
        else:
            st.error("Model tidak berhasil dimuat.")

    # --- GRAFIK 2: SIMULASI "WHAT-IF" (DITAMBAHKAN KEMBALI) ---
    st.markdown("---")
    st.subheader("🧪 Simulasi: Dampak Perubahan Diskon Terhadap Profit")
    sim_discount = st.slider('Pilih rentang diskon untuk disimulasikan', 0.0, 1.0, (0.0, 0.8))

    if model is not None:
        discount_range = np.linspace(sim_discount[0], sim_discount[1], 20)
        simulation_results = []

        for discount_val in discount_range:
            sim_input_data = input_data.copy()
            sim_input_data['Discount'] = discount_val
            
            sim_df = pd.DataFrame([sim_input_data])
            today = datetime.now()
            sim_df['Order_Year'], sim_df['Order_Month'], sim_df['Order_Day'] = today.year, today.month, today.day
            sim_df['Price_per_Unit'] = sim_df['Sales'] / sim_df['Quantity']
            sim_processed_df = pd.get_dummies(sim_df)
            sim_final_df = sim_processed_df.reindex(columns=MODEL_FEATURE_ORDER, fill_value=0)
            
            predicted_profit = model.predict(sim_final_df)[0]
            simulation_results.append({'Diskon': discount_val, 'Prediksi Profit': predicted_profit})

        simulation_df = pd.DataFrame(simulation_results)
        st.line_chart(simulation_df.set_index('Diskon'))

# --- KONTEN TAB 2: PREDIKSI BATCH DARI FILE ---
with tab2:
    st.header("Prediksi untuk Banyak Transaksi dari File CSV")
    uploaded_file = st.file_uploader("Unggah file CSV Anda di sini", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file, encoding='windows-1252')
            st.write("**Pratinjau Data yang Diunggah:**")
            st.dataframe(batch_df.head())

            if st.button('Proses dan Prediksi Seluruh File'):
                with st.spinner('Memproses dan melakukan prediksi...'):
                    batch_df['Order Date'] = pd.to_datetime(batch_df['Order Date'], format='%d/%m/%Y', errors='coerce')
                    batch_df['Ship Date'] = pd.to_datetime(batch_df['Ship Date'], format='%d/%m/%Y', errors='coerce')
                    batch_df.dropna(subset=['Order Date', 'Ship Date'], inplace=True)

                    for col in ['Sales', 'Quantity', 'Discount']:
                        batch_df[col] = pd.to_numeric(batch_df[col], errors='coerce')
                    batch_df.dropna(subset=['Sales', 'Quantity', 'Discount'], inplace=True)
                    
                    batch_df['Order_Year'] = batch_df['Order Date'].dt.year
                    batch_df['Order_Month'] = batch_df['Order Date'].dt.month
                    batch_df['Order_Day'] = batch_df['Order Date'].dt.day
                    batch_df['Shipping_Duration'] = (batch_df['Ship Date'] - batch_df['Order Date']).dt.days
                    
                    batch_df.loc[batch_df['Quantity'] == 0, 'Quantity'] = 1
                    batch_df['Price_per_Unit'] = (batch_df['Sales'] / batch_df['Quantity']).replace([np.inf, -np.inf], 0).fillna(0)
                    
                    processed_batch_df = pd.get_dummies(batch_df)
                    final_batch_df = processed_batch_df.reindex(columns=MODEL_FEATURE_ORDER, fill_value=0)
                    
                    predictions = model.predict(final_batch_df)
                    
                    batch_df['Predicted_Profit'] = predictions
                    
                    total_predicted_profit = batch_df['Predicted_Profit'].sum()
                    
                    st.subheader('Hasil Prediksi Batch:')
                    st.metric(label=f"Total Prediksi Profit dari {len(batch_df)} Baris yang Valid", value=f"$ {total_predicted_profit:,.2f}")
                    
                    st.write("**Detail Prediksi per Transaksi:**")
                    st.dataframe(batch_df)
        
        except Exception as e:
            st.error(f"Terjadi error saat memproses file: {e}")
            st.warning("Pastikan file CSV Anda memiliki kolom yang sama dengan dataset 'Sample - Superstore'.")
