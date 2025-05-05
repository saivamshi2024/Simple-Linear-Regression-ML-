import streamlit as st
import pickle
import numpy as np

# 1. Load the saved house‑price model
model = pickle.load(open(r"C:\Users\Admin\A VS CODE\House Prediction\linear_Regression_model.pkl",'rb'))

# 2. App title & description
st.title("House Price Prediction App")
st.write("Predict the sale price of a house based on its size (in square feet).")

# 3. User inputs
square_feet = st.number_input(
    "Enter house size (square feet):",
    min_value=100.0,
    max_value=10_000.0,
    value=1_000.0,
    step=50.0
)

# (Optional) If your model uses more features, just add more widgets:
# bedrooms = st.number_input("Bedrooms:", min_value=1, max_value=10, value=3, step=1)
# age = st.number_input("House age (years):", min_value=0, max_value=100, value=10, step=1)
# etc.

# 4. Prediction trigger
if st.button("Predict Price"):
    # Pack inputs into a 2D array
    X_input = np.array([[square_feet]])  # add more cols if you added more features
    price_pred = model.predict(X_input)
    
    # 5. Display
    st.success(f"Predicted house price: ${price_pred[0]:,.2f}")

# 6. Model info
st.write("This model was trained on historical sale prices vs. square footage.")
