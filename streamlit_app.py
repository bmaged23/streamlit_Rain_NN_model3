import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="K-means Prediction",
    page_icon=":0"
)

# Load the KMeans model, scaler, and encoders
model = joblib.load('NN_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scale = joblib.load('scaler.pkl')

# File uploader
file = st.file_uploader("Upload the file: ", type=["csv", "xlsx"])
df = None

if file is not None:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
        
    if df is not None:
        st.write(df.head())
        
        # IQR Interval handling
        IQR_interval = [[-6.35, 30.85], [2.3, 43.9], [-1.2, 2.0], [-3.81, 13.49], [-1.59, 16.79], [5.5, 73.5], [-11.0, 37.0], [-3.5, 40.5], [18.0, 122.0], [-6.5, 109.5], [1000.65, 1034.65], [998.0, 1032.4], [-4.17, 13.70], [-2.05, 11.41], [-1.65, 35.55], [1.75, 41.35]]
        columnss = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
        
        for column, (lower_bound, upper_bound) in zip(columnss, IQR_interval):
            if column in df.columns:
                outliers_lower = df[column] < lower_bound
                outliers_upper = df[column] > upper_bound
                df.loc[outliers_lower, column] = lower_bound
                df.loc[outliers_upper, column] = upper_bound
        
        y = df["RainTomorrow"]
        columns_to_drop = ['Temp9am', 'Pressure3pm', 'MaxTemp', 'Rainfall', 'Temp3pm', 'Unnamed: 0', 'RainTomorrow']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # Handle Date column
        if "Date" in df.columns:
            try:
                # Attempt to parse date with automatic format detection
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                df["Date"] = df["Date"].astype("int64") // 10**9  # Convert to Unix timestamp
                df["Date"] = df["Date"].astype("float32")
                # Handle NaT values if any
                df["Date"].fillna(df["Date"].mean(), inplace=True)  # Or another method of handling NaT values
            except ValueError as e:
                st.error(f"Error parsing Date column: {e}")
                st.stop()

        obj_col = df.select_dtypes(include="object").columns.tolist()
        num_col = ['MinTemp', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Cloud9am', 'Cloud3pm', 'Date']

        if set(num_col).issubset(df.columns):
            df[num_col] = scale.transform(df[num_col])

        if obj_col:
            for col in obj_col:
                if col in label_encoders:
                    try:
                        # Fill NaNs with the most frequent value from training or a suitable value
                        most_frequent_value = df[col].mode()[0]
                        df[col].fillna(most_frequent_value, inplace=True)
                        
                        df[col] = label_encoders[col].transform(df[col])
                    except KeyError:
                        st.error(f"No encoder found for column: {col}")
                        st.stop()
                    except ValueError as e:
                        st.error(f"Error in encoding column {col}: {e}")
                        st.stop()
                else:
                    st.error(f"No encoder found for column: {col}")
                    st.stop()

        try:
            prediction = model.predict(df)
            pred_acc = ["Yes" if p > 0.5 else "No" for p in prediction]
            pred = ["Rain" if p > 0.5 else "No Rain" for p in prediction]
            
            if len(pred) > 1:
                accuracy = np.sum(np.array(pred_acc) == y) / len(y)
                st.write("Accuracy: ", accuracy)
                st.session_state.text_list = pred
                st.write("Predicted Classes:")
                st.write(st.session_state.text_list)
            else:
                st.write(f"Predicted Class: {pred[0]}")
        except ValueError as e:
            st.error(f"Error in prediction: {e}")
            st.stop()
