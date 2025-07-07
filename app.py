import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

# --- Sidebar Navigation ---
st.sidebar.title("Main Page")
page = st.sidebar.radio("Navigation", ["🏠 Main Page", "📊 Classification", "📈 Clustering"])

# --- Main Page ---
if page == "🏠 Main Page":
    st.title("Ujian Akhir Semester")
    st.subheader("Streamlit Apps")
    st.markdown("Collection of my apps deployed in Streamlit")
    st.markdown("**Nama:** Mizan Ikbar")
    st.markdown("**NIM:** 22146003")

# --- Classification Page ---
elif page == "📊 Classification":
    st.title("Klasifikasi Diabetes Menggunakan KNN")
    st.write("Proyek ini menggunakan dataset Pima Indians Diabetes untuk mengklasifikasikan apakah seseorang menderita diabetes berdasarkan fitur-fitur medis.")

    # Load dataset
    df = pd.read_csv("diabetes.csv")
    st.write("### Data Sample")
    st.dataframe(df.head())

    # Preprocessing
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Classification Metrics
    st.write("### Metrik Klasifikasi")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Input Baru
    st.write("### Prediksi Data Baru")
    inputs = []
    for col in X.columns:
        value = st.number_input(f"Masukkan nilai untuk {col}", value=float(df[col].mean()))
        inputs.append(value)

    if st.button("Prediksi"):
        input_scaled = scaler.transform([inputs])
        prediction = knn.predict(input_scaled)[0]
        st.success(f"Hasil Prediksi: {'Diabetes' if prediction == 1 else 'Tidak Diabetes'}")

# --- Clustering Page ---
elif page == "📈 Clustering":
    st.title("Clustering Pelanggan Berdasarkan Income dan Spend Score")
    st.write("Proyek ini menggunakan K-Means untuk mengelompokkan pelanggan berdasarkan pendapatan dan skor belanja.")

    # Load data pelanggan
    data = pd.read_csv("lokasi_gerai_kopi_clean.csv")
    st.write("### Data Sample")
    st.dataframe(data.head())

    # Ambil kolom income dan spend_score
    X = data[["income", "spend_score"]]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X)
    data["Cluster"] = clusters

    # Visualisasi Clustering
    st.write("### Visualisasi Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(x="income", y="spend_score", hue="Cluster", palette="tab10", data=data, ax=ax)
    st.pyplot(fig)

    # Input Data Baru
    st.write("### Prediksi Cluster Pelanggan Baru")
    income_new = st.number_input("Masukkan pendapatan (income)", value=float(X["income"].mean()))
    score_new = st.number_input("Masukkan skor belanja (spend_score)", value=float(X["spend_score"].mean()))
    if st.button("Prediksi Cluster"):
        new_cluster = kmeans.predict([[income_new, score_new]])[0]
        st.success(f"Pelanggan baru masuk ke dalam Cluster {new_cluster}")
