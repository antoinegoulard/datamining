import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

st.title('DATA MINING PROJECT')
st.subheader('BIA 1 - Group 4')
st.markdown("""### Collaborators :
- **Mohammad DARMSY LADHA**: mohammad.darmsy-ladha@efrei.net
- **Ramy ABDELMOULA**: ramy.abdelmoula@efrei.net
- **Antoine GOULARD**: antoine.goulard@efrei.net
""")

st.markdown("---")

# Fonction pour charger et enregistrer les données dans l'état de session
def load_data(uploaded_file, separator, header_option):
    header = 0 if header_option == "Yes" else None
    data = pd.read_csv(uploaded_file, sep=separator, header=header)
    st.session_state['data'] = data

# Vérifier si les données sont déjà chargées dans l'état de session
if 'data' not in st.session_state:
    st.session_state['data'] = None

# Chargement du fichier
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    separator = st.text_input("Enter the separator used in the file", value=",")
    header_option = st.selectbox("Does your CSV file have a header?", options=["Yes", "No"], index=0)
    if st.button("Load Data"):
        load_data(uploaded_file, separator, header_option)
        st.write("Data loaded successfully!")

# S'assurer que les données sont bien chargées avant d'afficher les onglets
if st.session_state['data'] is not None:
    data = st.session_state['data']

    tabs = st.tabs(["Data Description", "Data Pre-processing and Cleaning", "Data Normalization", "Visualization"])

    with tabs[0]:
        #### 2. Data Description
        st.subheader("Data Description")
        st.write("Preview of the first few lines of the data:")
        st.dataframe(data.head())
        st.write("Preview of the last few lines of the data:")
        st.dataframe(data.tail())
        
        #### 3. Statistical Summary
        st.subheader("Statistical Summary")
        st.write("Basic statistics of the data:")
        st.write(data.describe(include='all'))
        st.write("Number of rows and columns:", data.shape)
        st.write("Column names:", data.columns.tolist())
        missing_values = data.isnull().sum()
        st.write("Number of missing values per column:")
        st.write(missing_values)

    with tabs[1]:
        #### Part II: Data Pre-processing and Cleaning
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
                data, encoders = encode_categorical_columns(data.copy())
                for column in data.columns:
                    if method == "Replace with mode":
                        imputer = SimpleImputer(strategy='most_frequent')
                    elif method == "Replace with mean":
                        imputer = SimpleImputer(strategy='mean')
                    elif method == "Replace with median":
                        imputer = SimpleImputer(strategy='median')
                    data[column] = imputer.fit_transform(data[[column]]).ravel()
                data = decode_categorical_columns(data, encoders)
            elif method == "Delete rows":
                data = data.dropna()
            elif method == "Delete columns":
                data = data.dropna(axis=1)
            elif method == "KNN Imputation":
                data, encoders = encode_categorical_columns(data.copy())
                imputer = KNNImputer(n_neighbors=5)
                data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
                data = decode_categorical_columns(data, encoders)
            elif method == "Simple Imputation":
                imputer = SimpleImputer(strategy='constant', fill_value=0)
                data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
            st.session_state['data'] = data
            st.write("Missing value handling completed.")
            st.dataframe(data.head())

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
            st.dataframe(data.head())

    with tabs[3]:
        #### Part III: Visualization of the cleaned data
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
