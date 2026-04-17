# app.py — House Price Predictor Web App
# This is a Streamlit app that loads our trained model
# and lets users predict house prices interactively

import streamlit as st      # streamlit creates the web interface
import pickle               # pickle loads our saved model files
import numpy as np          # numpy for number operations

# ─── STEP 1: Load the saved model and scaler ───────────────
# @st.cache_resource means "load this once and keep in memory"
# without this, model reloads every time user clicks anything — very slow!
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:  
        # 'rb' means read binary — opposite of 'wb' we used to save
        model = pickle.load(f)               
        # pickle.load() reconstructs the model from the file
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)              
        # loads the exact same scaler used during training
    return model, scaler

model, scaler = load_model()  
# actually calls the function and gets model + scaler

# ─── STEP 2: App title and description ─────────────────────
st.title("House Price Predictor")
# st.title() creates a large heading at the top of the page

st.markdown("""
This app predicts **California house prices** using a 
Random Forest model trained on 20,640 houses.
- Model accuracy: **R² = 0.80**
- Average error: **$50,000**
""")
# st.markdown() renders formatted text with bold, bullets etc

st.divider()  
# draws a horizontal line to separate sections

# ─── STEP 3: Input sliders for house features ───────────────
st.subheader("Enter House Details")
# st.subheader() creates a smaller heading

# create two columns side by side for a cleaner layout
col1, col2 = st.columns(2)
# st.columns(2) splits the page into 2 equal columns

with col1:  
    # everything inside 'with col1' goes in the LEFT column
    
    med_inc = st.slider(
        "Median Income (in $10,000s)",  # label shown to user
        min_value=0.5,                  # minimum slider value
        max_value=15.0,                 # maximum slider value
        value=5.0,                      # default starting value
        step=0.1                        # how much each tick moves
    )
    
    house_age = st.slider(
        "House Age (years)",
        min_value=1,
        max_value=52,
        value=20,
        step=1
    )
    
    ave_rooms = st.slider(
        "Average Rooms",
        min_value=1.0,
        max_value=15.0,
        value=5.0,
        step=0.1
    )
    
    ave_bedrms = st.slider(
        "Average Bedrooms",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

with col2:  
    # everything inside 'with col2' goes in the RIGHT column
    
    population = st.slider(
        "Neighborhood Population",
        min_value=3,
        max_value=35000,
        value=1500,
        step=10
    )
    
    ave_occup = st.slider(
        "Average Occupants per House",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.1
    )
    
    latitude = st.slider(
        "Latitude",
        min_value=32.5,
        max_value=42.0,
        value=37.0,
        step=0.1
    )
    
    longitude = st.slider(
        "Longitude",
        min_value=-124.0,
        max_value=-114.0,
        value=-119.0,
        step=0.1
    )

# ─── STEP 4: Engineer the same features we created earlier ──
# IMPORTANT: we must create the same 3 features the model was trained on
rooms_per_person    = ave_rooms / ave_occup
# same formula as df_eng['RoomsPerPerson']

bedroom_ratio       = ave_bedrms / ave_rooms
# same formula as df_eng['BedroomRatio']

population_density  = population / ave_occup
# same formula as df_eng['PopulationDensity']

# ─── STEP 5: Prepare input for the model ────────────────────
# combine all features into one array in the EXACT same order
# as the training data columns!
input_data = np.array([[
    med_inc, house_age, ave_rooms, ave_bedrms,
    population, ave_occup, latitude, longitude,
    rooms_per_person, bedroom_ratio, population_density
]])
# np.array([[...]]) creates a 2D array — model expects 2D input
# shape will be (1, 11) — 1 house, 11 features

# scale the input using the SAME scaler from training
input_scaled = scaler.transform(input_data)
# must scale new data the same way training data was scaled

# ─── STEP 6: Make prediction ────────────────────────────────
st.divider()
st.subheader("Predicted Price")

prediction = model.predict(input_scaled)
# model.predict() returns an array — we take [0] for first value

predicted_price = prediction[0] * 100000
# multiply by 100,000 because prices are in $100k units
# so 2.5 → $250,000

# display the prediction in a big metric card
st.metric(
    label="Estimated House Value",
    value=f"${predicted_price:,.0f}"
    # :,.0f formats number with commas and no decimals
    # 285000 → $285,000
)

# ─── STEP 7: Show feature importance chart ──────────────────
st.divider()
st.subheader("What drives house prices?")

import pandas as pd
import matplotlib.pyplot as plt

# recreate feature importance chart
feature_names = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude',
    'RoomsPerPerson', 'BedroomRatio', 'PopulationDensity'
]

importances = pd.Series(
    model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 4))
# plt.subplots() creates a figure and axes object
# we pass 'ax' to plot on — required for Streamlit

importances.plot(kind='bar', color='steelblue', edgecolor='white', ax=ax)
ax.set_title('Feature Importance')
ax.set_xlabel('Features')
ax.set_ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

st.pyplot(fig)
# st.pyplot() displays matplotlib charts in Streamlit
# different from plt.show() which we used in Jupyter!

# ─── STEP 8: Show input summary ─────────────────────────────
st.divider()
st.subheader("Your Input Summary")

# display what the user entered in a clean table
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
# st.dataframe() displays a pandas DataFrame as an interactive table