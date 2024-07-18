import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

st.title('DATA MINING PROJECT')
st.subheader('BIA 1 - Group 4')
st.markdown("""### Collaborators:
- **Mohammad DARMSY LADHA**: mohammad.darmsy-ladha@efrei.net
- **Ramy ABDELMOULA**: ramy.abdelmoula@efrei.net
- **Antoine GOULARD**: antoine.goulard@efrei.net
""")
 
st.markdown("---") 
# Function to load and store data in session state
def load_data(uploaded_file, separator, header_option):
    header = 0 if header_option == "Yes" else None
    data = pd.read_csv(uploaded_file, sep=separator, header=header)
    st.session_state['data'] = data
 
# Check if data is already loaded in session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
 
# File uploading

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    separator = st.text_input("Enter the separator used in the file", value=",")
    header_option = st.selectbox("Does your CSV file have a header?", options=["Yes", "No"], index=0)
    if st.button("Load Data"):
        load_data(uploaded_file, separator, header_option)
        st.write("Data loaded successfully!")
 
# Ensure data is loaded before displaying tabs
if st.session_state['data'] is not None:
    data = st.session_state['data']
 
    tabs = st.tabs(["Data Description", "Data Pre-processing and Cleaning", "Data Normalization", "Visualization", "Clustering", "Prediction"])
 
    with tabs[0]:
        #### Data Description
        st.subheader("Data Description")
        st.write("Preview of the data:")
        st.dataframe(data)
       
        #### Statistical Summary
        st.subheader("Statistical Summary")
        st.write("Basic statistics of the data:")
        st.write(data.describe(include='all'))
        st.write("Number of rows and columns:", data.shape)
        st.write("Column names:", data.columns.tolist())
        missing_values = data.isnull().sum()
        st.write("Number of missing values per column:")
        st.write(missing_values)
 
    with tabs[1]:
        #### Data Pre-processing and Cleaning
        st.subheader("Data Pre-processing and Cleaning")
        method = st.selectbox("Select method to handle missing values:",
                              ["Delete rows", "Delete columns", "Replace with mean", "Replace with median", "Replace with mode", "KNN Imputation", "Simple Imputation"])
 
        def encode_categorical_columns(df):
            label_encoders = {}
            for col in df.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
            return df, label_encoders

        def decode_categorical_columns(df, encoders):
            for col, le in encoders.items():
                df[col] = le.inverse_transform(df[col].astype(int))
            return df
 
        if st.button("Apply Missing Value Method"):
            data = st.session_state['data'].copy()
            if method in ["Replace with mean", "Replace with median", "Replace with mode"]:
                for column in data.columns:
                    if method == "Replace with mode":
                        mode_val = data[column].mode(dropna=True)
                        if mode_val.empty:
                            mode_val = data[column].dropna().unique()[0] if data[column].dropna().unique().size > 0 else 0
                        else:
                            mode_val = mode_val[0]
                        data[column].fillna(mode_val, inplace=True)
                    elif method == "Replace with mean":
                        if data[column].dtype in [np.float64, np.int64]:
                            mean_val = data[column].mean()
                            data[column].fillna(mean_val, inplace=True)
                    elif method == "Replace with median":
                        if data[column].dtype in [np.float64, np.int64]:
                            median_val = data[column].median()
                            data[column].fillna(median_val, inplace=True)
            elif method == "Delete rows":
                data.dropna(inplace=True)
            elif method == "Delete columns":
                data.dropna(axis=1, inplace=True)
            elif method == "KNN Imputation":
                numeric_data = data.select_dtypes(include=[np.number])
                imputer = KNNImputer(n_neighbors=5)
                numeric_data = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)
                data[numeric_data.columns] = numeric_data
            elif method == "Simple Imputation":
                imputer = SimpleImputer(strategy='constant', fill_value=0)
                data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
            st.session_state['data'] = data
            st.write("Missing value handling completed.")

            st.dataframe(data)
 
    with tabs[2]:
        ## Data Normalization
        st.subheader("Data Normalization")
        norm_method = st.selectbox("Select normalization method:",
                                   ["Min-Max Scaling", "Z-Score Standardization"])
     
        if st.button("Apply Normalization"):
            data = st.session_state['data'].copy()
            numeric_columns = data.select_dtypes(include=['int', 'float']).columns
            if norm_method == "Min-Max Scaling":
                scaler = MinMaxScaler()
                data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            elif norm_method == "Z-Score Standardization":
                scaler = StandardScaler()
                data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            st.session_state['data'] = data
            st.write("Data normalization completed.")
            st.dataframe(data)
 
    with tabs[3]:
        ## Visualization of the cleaned data
        st.subheader("Visualization of the cleaned data")
 
        viz_option = st.selectbox("Select visualization type:", ["Histogram", "Box Plot"])
        selected_column = st.selectbox("Select column for visualization:", data.columns)

        if st.button("Generate Visualization"):
            plt.figure(figsize=(10, 6))
            if viz_option == "Histogram":
                sns.histplot(data[selected_column], kde=True)
                plt.title(f'Histogram of {selected_column}')
                plt.xlabel(selected_column)
                plt.ylabel('Frequency')
            elif viz_option == "Box Plot":
                sns.boxplot(x=data[selected_column])
                plt.title(f'Box Plot of {selected_column}')
                plt.xlabel(selected_column)
            st.pyplot(plt)
 
    with tabs[4]:
        ## Clustering
        st.subheader("Clustering")
        cluster_algo = st.selectbox("Select clustering algorithm:", ["K-Means", "DBSCAN"])
 
        def plot_clusters(data, labels, algorithm_name):
            pca = PCA(2)
            df = pd.DataFrame(pca.fit_transform(data), columns=['PCA1', 'PCA2'])
            df['Cluster'] = labels
 
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis')
            plt.title(f'{algorithm_name} Clustering')
            st.pyplot(plt)
 
            # Calculating and displaying cluster densities for DBSCAN
            if algorithm_name == "DBSCAN":
                cluster_info = []
                for cluster in np.unique(labels):
                    cluster_points = df[df['Cluster'] == cluster]
                    x_min, x_max = cluster_points['PCA1'].min(), cluster_points['PCA1'].max()
                    y_min, y_max = cluster_points['PCA2'].min(), cluster_points['PCA2'].max()
                    area = (x_max - x_min) * (y_max - y_min)
                    num_points = len(cluster_points)
                    density = num_points / area if area > 0 else 0
                    cluster_info.append({'Cluster': cluster, 'NumPoints': num_points, 'Area': area, 'Density': density})
 
                cluster_info_df = pd.DataFrame(cluster_info)
                st.write("Cluster densities:")
                st.dataframe(cluster_info_df)
 
        if cluster_algo == "K-Means":
            n_clusters = st.number_input("Number of clusters (k):", min_value=2, max_value=10, value=3)
            if st.button("Apply K-Means Clustering"):
                data = st.session_state['data'].copy()
                numeric_data = data.select_dtypes(include=['int', 'float'])
                kmeans = KMeans(n_clusters=n_clusters)
                labels = kmeans.fit_predict(numeric_data)
                data['Cluster'] = labels
                st.session_state['data'] = data
 
                st.write("K-Means clustering completed.")
                st.dataframe(data)
               
                plot_clusters(numeric_data, labels, "K-Means")
 
                st.write("Cluster Centers:")
                centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_data.columns)
                st.dataframe(centers_df)
                st.write("Number of points in each cluster:")
                st.write(pd.Series(labels).value_counts())
 
        elif cluster_algo == "DBSCAN":
            eps = st.number_input("Epsilon (eps):", min_value=0.1, max_value=10.0, value=0.5)
            min_samples = st.number_input("Minimum samples:", min_value=1, max_value=100, value=5)
            if st.button("Apply DBSCAN Clustering"):
                data = st.session_state['data'].copy()
                numeric_data = data.select_dtypes(include=['int', 'float'])
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(numeric_data)
                data['Cluster'] = labels
                st.session_state['data'] = data
 
                st.write("DBSCAN clustering completed.")
                st.dataframe(data)
 
                plot_clusters(numeric_data, labels, "DBSCAN")
