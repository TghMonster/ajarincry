import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def main():
    """Main function to structure your Streamlit app"""

    # Add a title and description
    st.title("Churn Prediction App")
    st.write("Use this app to predict customer churn based on their profile.")

    # User input section with clear labels
    credit_score = st.number_input("Credit Score:")
    geography = st.selectbox("Geography:", ["France", "Germany", "Spain"])  # Adjust options based on data
    gender = st.selectbox("Gender:", ["Male", "Female"])
    age = st.slider("Age:", min_value=0, max_value=100, step=1)
    tenure = st.slider("Tenure (years with company):", min_value=0, max_value=50, step=1)
    balance = st.number_input("Balance (account balance):")
    num_of_products = st.number_input("Number of Products:")
    has_cr_card = st.selectbox("Has Credit Card? (Yes/No)", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member? (Yes/No)", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary:")

    # Preprocessing for categorical features (assuming One-Hot Encoding)
    # categorical_features = ["Geography", "HasCrCard", "IsActiveMember"]
    # encoder = OneHotEncoder(sparse=False)  # Set sparse=False for easier handling

    # Prepare user input as a DataFrame (assuming model expects a DataFrame)
    user_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [1 if has_cr_card == "Yes" else 0],
        "IsActiveMember": [1 if is_active_member == "Yes" else 0],
        "EstimatedSalary": [estimated_salary]
    })
    print(user_data)

    def load_scalers_encoder(scaler_path="scaler.pkl", encoder_path="encoder.pkl"):
        with open(scaler_path, "rb") as scaler_file:
            scalers = pickle.load(scaler_file)
        with open(encoder_path, "rb") as encoder_file:
            encoder = pickle.load(encoder_file)
        return scalers, encoder

    # Make prediction button with a loading indicator
    if st.button("Predict Churn Risk"):
        all_filled = credit_score and geography and gender and age and tenure and balance and num_of_products and has_cr_card and is_active_member and estimated_salary
        if not all_filled:
            st.error("Please fill in all fields before submitting.")
            return
        # Load the scalers and encoder
        scalers, encoder = load_scalers_encoder()

        categorical = ['Geography', 'Gender']
        conti = ['Age','Balance','CreditScore', 'EstimatedSalary']
        
        user_data_subset = user_data[categorical]
        user_data_encoded = pd.DataFrame(encoder.transform(user_data_subset).toarray(), columns=encoder.get_feature_names_out(categorical))
        user_data = user_data.reset_index(drop=True)
        user_data = pd.concat([user_data, user_data_encoded], axis=1)
        user_data.drop(categorical, axis=1, inplace=True)

        # scaler
        user_data[conti] = scalers.transform(user_data[conti])
        with st.spinner("Making prediction..."):
            # Load the pickled model from its saved location
            with open("finalized_model.pkl", "rb") as model_file:
                model = pickle.load(model_file)

            # Make prediction
            prediction = model.predict(user_data)[0]  # Assuming prediction is a probability

            # Display prediction with clear interpretation
            if prediction == 1:
                st.write("Predicted: **CHURN**")
            else:
                st.write("Predicted: **NOT CHURN**")

if __name__ == "__main__":
    main()
