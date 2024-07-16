import streamlit as st
import pandas as pd

st.title('DATA MINING PROJECT')
st.subheader('BIA 1 - Group 4')
st.markdown("""### Collaborators :
- **Mohammad DARMSY LADHA**: mohammad.darmsy-ladha@efrei.net
- **Ramy ABDELMOULA**: ramy.abdelmoula@efrei.net
- **Antoine GOULARD**: antoine.goulard@efrei.net
""")

st.markdown("""---""")
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
    st.write(data.describe())
    st.write("Number of rows and columns:", data.shape)
    st.write("Column names:", data.columns.tolist())
    missing_values = data.isnull().sum()
    st.write("Number of missing values per column:")
    st.write(missing_values)

