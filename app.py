import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

st.title('DATA MINING PROJECT')
st.subheader('BIA 1 - Group 4')
st.markdown("""### Collaborators :
- **Mohammad DARMSY LADHA**: mohammad.darmsy-ladha@efrei.net
- **Ramy ABDELMOULA**: ramy.abdelmoula@efrei.net
- **Antoine GOULARD**: antoine.goulard@efrei.nets
""")

st.markdown("---")

# File upload and input for separator and header
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    separator = st.text_input("Enter the separator used in the file", value=",")
    header_option = st.selectbox("Does your CSV file have a header?", options=["Yes", "No"], index=0)
    header = 0 if header_option == "Yes" else None

    # Load data
    data = pd.read_csv(uploaded_file, sep=separator, header=header)
    st.write("Data loaded successfully!")

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

    #### Part II: Data Pre-processing and Cleaning
    st.markdown("---")
    st.subheader("Data Pre-processing and Cleaning")

    ## Managing Missing Values
    st.subheader("Managing Missing Values")
    method = st.selectbox("Select method to handle missing values:",
                          ["Delete rows", "Delete columns", "Replace with mean (numeric only)", "Replace with median (numeric only)", "Replace with mode", "KNN Imputation", "Simple Imputation"])

    if st.button("Apply Missing Value Method"):
        if method in ["Replace with mean (numeric only)", "Replace with median (numeric only)", "Replace with mode"]:
            for column in data.columns:
                if data[column].dtype == 'object':
                    # For non-numeric data, replace with the most frequent value
                    imputer = SimpleImputer(strategy='most_frequent')
                else:
                    # For numeric data, apply the selected method
                    if method == "Replace with mean (numeric only)":
                        imputer = SimpleImputer(strategy='mean')
                    elif method == "Replace with median (numeric only)":
                        imputer = SimpleImputer(strategy='median')
                    elif method == "Replace with mode":
                        imputer = SimpleImputer(strategy='most_frequent')
                
                data[column] = imputer.fit_transform(data[[column]]).ravel()
        elif method == "Delete rows":
            data = data.dropna()
        elif method == "Delete columns":
            data = data.dropna(axis=1)
        elif method == "KNN Imputation":
            imputer = KNNImputer(n_neighbors=5)
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        elif method == "Simple Imputation":
            imputer = SimpleImputer(strategy='constant', fill_value=0)
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        st.write("Missing value handling completed.")
        st.dataframe(data.head())

    ## Data Normalization
    st.subheader("Data Normalization")
    norm_method = st.selectbox("Select normalization method:",
                               ["Min-Max Scaling", "Z-Score Standardization"])
    
    if st.button("Apply Normalization"):
        if norm_method == "Min-Max Scaling":
            scaler = MinMaxScaler()
            data[data.columns] = scaler.fit_transform(data)
        elif norm_method == "Z-Score Standardization":
            scaler = StandardScaler()
            data[data.columns] = scaler.fit_transform(data)
        st.write("Data normalization completed.")
        st.dataframe(data.head())

#NewPullRequest