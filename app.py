# app.py — House Price Predictor Web App
# This version trains the model directly
# so it works on Streamlit Cloud without pkl files

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── STEP 1: Train model (cached so it only runs once) ──────
@st.cache_resource
def train_model():
    # import everything needed inside the function
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    # load data
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    # feature engineering
    df['RoomsPerPerson']   = df['AveRooms'] / df['AveOccup']
    df['BedroomRatio']     = df['AveBedrms'] / df['AveRooms']
    df['PopulationDensity']= df['Population'] / df['AveOccup']

    # split features and target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # train model with best hyperparameters we found
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns.tolist()

# train the model — shows spinner while loading
with st.spinner("Loading model... please wait!"):
    model, scaler, feature_columns = train_model()

# ─── STEP 2: App title and description ──────────────────────
st.title("House Price Predictor")

st.markdown("""
This app predicts **California house prices** using a
Random Forest model trained on 20,640 houses.
- Model accuracy: **R² = 0.80**
- Average error: **$50,000**
""")

st.divider()

# ─── STEP 3: Input sliders ──────────────────────────────────
st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    med_inc = st.slider(
        "Median Income (in $10,000s)",
        min_value=0.5, max_value=15.0,
        value=5.0, step=0.1
    )
    house_age = st.slider(
        "House Age (years)",
        min_value=1, max_value=52,
        value=20, step=1
    )
    ave_rooms = st.slider(
        "Average Rooms",
        min_value=1.0, max_value=15.0,
        value=5.0, step=0.1
    )
    ave_bedrms = st.slider(
        "Average Bedrooms",
        min_value=0.5, max_value=5.0,
        value=1.0, step=0.1
    )

with col2:
    population = st.slider(
        "Neighborhood Population",
        min_value=3, max_value=35000,
        value=1500, step=10
    )
    ave_occup = st.slider(
        "Average Occupants per House",
        min_value=1.0, max_value=10.0,
        value=3.0, step=0.1
    )
    latitude = st.slider(
        "Latitude",
        min_value=32.5, max_value=42.0,
        value=37.0, step=0.1
    )
    longitude = st.slider(
        "Longitude",
        min_value=-124.0, max_value=-114.0,
        value=-119.0, step=0.1
    )

# ─── STEP 4: Engineer features ──────────────────────────────
rooms_per_person    = ave_rooms / ave_occup
bedroom_ratio       = ave_bedrms / ave_rooms
population_density  = population / ave_occup

# ─── STEP 5: Prepare and scale input ────────────────────────
input_data = np.array([[
    med_inc, house_age, ave_rooms, ave_bedrms,
    population, ave_occup, latitude, longitude,
    rooms_per_person, bedroom_ratio, population_density
]])

input_scaled = scaler.transform(input_data)

# ─── STEP 6: Predict and display ────────────────────────────
st.divider()
st.subheader("Predicted Price")

prediction = model.predict(input_scaled)
predicted_price = prediction[0] * 100000

st.metric(
    label="Estimated House Value",
    value=f"${predicted_price:,.0f}"
)

# ─── STEP 7: Feature importance chart ───────────────────────
st.divider()
st.subheader("What drives house prices?")

importances = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 4))
importances.plot(kind='bar', color='steelblue', edgecolor='white', ax=ax)
ax.set_title('Feature Importance')
ax.set_xlabel('Features')
ax.set_ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)

# ─── STEP 8: Input summary table ────────────────────────────
st.divider()
st.subheader("Your Input Summary")

summary = {
    "Feature": [
        "Median Income", "House Age", "Ave Rooms",
        "Ave Bedrooms", "Population", "Ave Occupants",
        "Latitude", "Longitude"
    ],
    "Your Value": [
        f"${med_inc * 10000:,.0f}",
        f"{house_age} years",
        f"{ave_rooms:.1f} rooms",
        f"{ave_bedrms:.1f} bedrooms",
        f"{population:,} people",
        f"{ave_occup:.1f} people/house",
        f"{latitude}°N",
        f"{longitude}°E"
    ]
}
st.dataframe(pd.DataFrame(summary), hide_index=True)