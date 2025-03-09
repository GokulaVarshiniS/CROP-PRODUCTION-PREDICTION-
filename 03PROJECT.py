import base64
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set page configuration
st.set_page_config(page_title="🌾 Crop Production Prediction Dashboard", layout="wide")

# Path to your FAOSTAT Excel file
FILE_PATH = "D:/GUVI/MINI PROJECT/GUVI-Prj03/FAOSTAT_data.xlsx"

# Load image and convert to base64
with open('D:/GUVI/MINI PROJECT/GUVI-Prj03/02', "rb") as image_file:
    img_str = base64.b64encode(image_file.read()).decode()

# Set the background image and styling
st.markdown(f"""
    <style>
    .stApp {{
        background: url(data:image/jpeg;base64,{img_str}) no-repeat center center fixed;
        background-size: cover;
    }}
    .main-title {{ 
        color: black; 
        font-size: 36px; 
        font-weight: bold; 
        text-align: center; 
        margin-bottom: 30px; 
        text-shadow: 2px 2px 5px white; 
    }}
    .plain-text {{ 
        color: black; 
        font-size: 18px; 
        font-weight: bold; 
        margin: 10px 0; 
    }}
    .performance-box {{
        background-color: black; 
        color: white; 
        padding: 10px; 
        border-radius: 8px; 
        margin: 20px 0; 
    }}
    .performance-table th, .performance-table td {{
        color: white; 
        background-color: black; 
        text-align: center; 
        padding: 10px; 
        border: 1px solid white; 
    }}
    .detailed-data-table th, .detailed-data-table td {{
        color: black; 
        background-color: #f5f5f5; 
        text-align: center; 
        padding: 8px; 
        border: 1px solid #ddd; 
    }}
    </style>
    """, unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    return pd.read_excel(FILE_PATH)

# Preprocess data
def preprocess_data(df):
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Area': 'Country', 'Item': 'Crop'})
    df = df[df['Element'].isin(['Area harvested', 'Yield', 'Production'])]
    df = df.pivot_table(index=['Country', 'Crop', 'Year'], columns='Element', values='Value', aggfunc='first').reset_index()
    return df.fillna(0)

# Plot exploratory data analysis
def plot_eda(df):
    st.markdown("<div class='plain-text'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.dataframe(df)

    if 'Production' in df.columns:
        st.markdown("<div class='plain-text'>📈 Production Trend Over Time</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x='Year', y='Production', marker='o', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.markdown("<div class='plain-text'>📦 Outlier Detection (Production)</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Year', y='Production', ax=ax)
        st.pyplot(fig)

        st.markdown("<div class='plain-text'>🔎 Area vs Production</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.regplot(data=df, x='Area harvested', y='Production', scatter_kws={'s': 60}, line_kws={"color": "red"}, ax=ax)
        st.pyplot(fig)

# Train the linear regression model
def train_linear_model(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    return {
        'model': lr,
        'MSE': mean_squared_error(y_test, y_pred_lr),
        'R²': r2_score(y_test, y_pred_lr),
        'MAE': mean_absolute_error(y_test, y_pred_lr)
    }

# Prediction interface
def prediction_interface(model):
    st.markdown("<div class='plain-text'>🔮 Predict Future Crop Production</div>", unsafe_allow_html=True)
    area = st.number_input("Area Harvested (ha)", min_value=0.0)
    yield_val = st.number_input("Yield (kg/ha)", min_value=0.0)
    year = st.number_input("Year", min_value=2000, max_value=2050)

    if st.button("Predict Production"):
        prediction = model.predict([[area, yield_val, year]])
        st.success(f"🌾 Predicted Production: **{prediction[0]:,.2f} tons**")

# Main function
def main():
    st.markdown("<div class='main-title'>🌾 Crop Production Prediction Dashboard</div>", unsafe_allow_html=True)

    # Load and preprocess data
    df = load_data()
    df_clean = preprocess_data(df)

    # Dropdown headers with plain text styling
    st.markdown("<div class='plain-text'>🌍 Select Country</div>", unsafe_allow_html=True)
    country = st.selectbox("", df_clean['Country'].unique())

    st.markdown("<div class='plain-text'>🌾 Select Crop</div>", unsafe_allow_html=True)
    crop = st.selectbox("", df_clean[df_clean['Country'] == country]['Crop'].unique())

    # Filtered data message with plain text styling
    filtered_df = df_clean[(df_clean['Country'] == country) & (df_clean['Crop'] == crop)]
    st.markdown(f"<div class='plain-text'>Filtered Data - {len(filtered_df)} records for {crop} in {country}.</div>", unsafe_allow_html=True)

    # Dataframe display
    st.dataframe(filtered_df)

    if len(filtered_df) == 0:
        st.error("❌ No data found.")
        st.stop()

    # Detailed Crop Production Data
    st.markdown("<div class='plain-text'>Detailed Crop Production Data</div>", unsafe_allow_html=True)
    detailed_df = filtered_df[['Crop', 'Production']].groupby('Crop').sum().reset_index()
    st.markdown(
        detailed_df.to_html(index=False, classes='detailed-data-table'),
        unsafe_allow_html=True
    )

    plot_eda(filtered_df)

    filtered_df = filtered_df[(filtered_df['Area harvested'] > 0) & (filtered_df['Yield'] > 0) & (filtered_df['Production'] > 0)]
    if len(filtered_df) < 5:
        st.warning("⚠️ Not enough data for training.")
        st.stop()

    X = filtered_df[['Area harvested', 'Yield', 'Year']]
    y = filtered_df['Production']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_results = train_linear_model(X_train, X_test, y_train, y_test)

    # Display Model Performance with black background styling
    st.markdown("<div class='performance-box'>📈 Model Performance</div>", unsafe_allow_html=True)
    results_df = pd.DataFrame([{
        'Model': 'Linear Regression',
        'MSE': model_results['MSE'],
        'R²': model_results['R²'],
        'MAE': model_results['MAE']
    }])

    # Apply custom styling to the performance table
    st.markdown(
        results_df.to_html(index=False, classes='performance-table'),
        unsafe_allow_html=True
    )

    prediction_interface(model_results['model'])

if __name__ == "__main__":
    main()
