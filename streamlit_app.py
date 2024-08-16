import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Rain Prediction Model",
    page_icon="🌧️",
    layout="wide"
)

# Custom CSS for enhanced appearance
st.markdown(
    """
    <style>
    body {
        background-color: #e0f7fa;
        color: #000;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #6a1b9a;
        font-family: 'Arial Black', sans-serif;
        text-align: center;
    }
    .stButton>button {
        background-color: #00796b;
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .stFileUploader>div>button {
        background-color: #004d40;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
    }
    .data-box, .prediction-box {
        background-color: #e0f2f1;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stDataFrame {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 10px;
    }
    .stMarkdown p {
        color: #006064;
        font-size: 16px;
    }
    .prediction-text {
        font-size: 20px;
        margin-bottom: 10px;
    }
    .predicted-class {
        font-size: 30px;
        color: #ff7043;
        
    }
    footer {
        color: #777;
        font-size: 14px;
        text-align: center;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.title("🌧️ Rain Prediction Model")

# Load models and encoders
model = joblib.load('NN_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scale = joblib.load('scaler.pkl')

# File uploader
file = st.file_uploader("Upload your CSV or Excel file:", type=["csv", "xlsx"])
df = None

if file is not None:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
        
    if df is not None:
        st.subheader("Data Preview")
        st.markdown("<div class='data-box'>", unsafe_allow_html=True)
        st.write(df.head())
        st.markdown("</div>", unsafe_allow_html=True)
        
        df["Date"] = df["Date"].astype("category")
        df["Date"] = label_encoders["Date"].transform(df["Date"])

        IQR_interval = [[-6.35, 30.85], [2.3, 43.9], [-1.2, 2.0], [-3.81, 13.49], [-1.59, 16.79], [5.5, 73.5],
                        [-11.0, 37.0], [-3.5, 40.5], [18.0, 122.0], [-6.5, 109.5], [1000.65, 1034.65], [998.0, 1032.4],
                        [-4.17, 13.70], [-2.05, 11.41], [-1.65, 35.55], [1.75, 41.35]]
        columnss = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am',
                    'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
                    'Cloud3pm', 'Temp9am', 'Temp3pm']
        
        for column, (lower_bound, upper_bound) in zip(columnss, IQR_interval):
            if column in df.columns:
                df.loc[df[column] < lower_bound, column] = lower_bound
                df.loc[df[column] > upper_bound, column] = upper_bound

        columns_to_drop = ['Temp9am', 'Pressure3pm', 'MaxTemp', 'Rainfall', 'Temp3pm', 'Unnamed: 0', 'RainTomorrow']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        obj_col = df.select_dtypes(include="object").columns.tolist()
        num_col = ['MinTemp', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
                   'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Cloud9am', 'Cloud3pm', 'Date']

        if set(num_col).issubset(df.columns):
            df[num_col] = scale.transform(df[num_col])

        if obj_col:
            for col in obj_col:
                if col in label_encoders:
                    try:
                        most_frequent_value = df[col].mode()[0]
                        df[col].fillna(most_frequent_value, inplace=True)
                        
                        df[col] = label_encoders[col].transform(df[col])
                    except KeyError:
                        st.error(f"No encoder found for column: {col}")
                        st.stop()
                    except ValueError as e:
                        st.error(f"Error in encoding column {col}: {e}")
                        st.stop()

        try:
            prediction = model.predict(df)
            pred = ["Rain" if p > 0.5 else "No Rain" for p in prediction]
            st.subheader("Prediction Results")
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            if len(pred) > 1:
                st.markdown("<p class='prediction-text'>Predicted Classes:</p>", unsafe_allow_html=True)
                for p in pred:
                    st.markdown(f"<p class='predicted-class'>{p}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='predicted-class'>{pred[0]}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except ValueError as e:
            st.error(f"Error in prediction: {e}")
            st.stop()

# Footer
st.markdown(
    """
    <footer>
          Made with ❤️ | Powered by Streamlit
    </footer>
    """,
    unsafe_allow_html=True
)
