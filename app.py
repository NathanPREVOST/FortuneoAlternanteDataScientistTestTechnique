import streamlit as st
import numpy as np
from joblib import load
import pandas as pd

# Chargement du modèle
model = load('modele_assurance_auto.joblib')

def prepare_input(age, gender, driving_license, region_code, previously_insured, vehicle_age, vehicle_damage, annual_premium, policy_sales_channel, vintage):
    # Création d'un dictionnaire pour les données d'entrée
    data = {
        'Age': [age],
        'Gender': [gender],
        'Driving_License': [driving_license],
        'Region_Code': [region_code],
        'Previously_Insured': [previously_insured],
        'Vehicle_Age': [vehicle_age],
        'Vehicle_Damage': [vehicle_damage],
        'Annual_Premium': [annual_premium],
        'Policy_Sales_Channel': [policy_sales_channel],
        'Vintage': [vintage]
    }
    
    # Conversion en DataFrame
    user_input = pd.DataFrame(data)
    
    # Assurez-vous que l'ordre des colonnes correspond à l'ordre attendu par votre modèle
    # user_input = user_input[['Age', 'Gender', ... et ainsi de suite pour toutes les colonnes nécessaires]]
    
    return user_input

def predict_interest(model, user_input):
    prediction = model.predict(user_input)
    return prediction[0]

def predict_interest_proba(model, user_input):
    interest_probability = model.predict_proba(user_input)[0][1]
    return interest_probability

# Interface utilisateur de Streamlit
def main():
    st.title('Prédiction d\'intérêt pour l\'assurance automobile')

    with st.sidebar:
        st.image('car_insurance.jpg', caption='Assurance Automobile')

    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider('Âge', 18, 100, 25)
        gender = st.selectbox('Genre', options=['Male', 'Female'], index=0)
        driving_license = st.selectbox('Possède un permis de conduire ?', options=[1, 0], format_func=lambda x: 'Oui' if x == 1 else 'Non', index=0)
        previously_insured = st.selectbox('Déjà assuré ?', options=[0, 1], format_func=lambda x: 'Non' if x == 0 else 'Oui', index=0)
        vehicle_damage = st.selectbox('Dommages antérieurs au véhicule ?', options=['Yes', 'No'], format_func=lambda x: 'Oui' if x == 'Yes' else 'Non', index=0)
    
    with col2:
        vehicle_age = st.selectbox('Âge du véhicule', options=['< 1 Year', '1-2 Year', '> 2 Years'], index=1)
        annual_premium = st.slider('Prime annuelle', 0, 100000, 10000)
        policy_sales_channel = st.number_input('Canal de vente de la politique', min_value=0, max_value=200, value=100)
        region_code = st.number_input('Code région', min_value=0, max_value=50, value=15)
        vintage = st.slider('Ancienneté (jours)', 0, 300, 150)

    if st.button('Prédire'):
        user_input = prepare_input(age, gender, driving_license, region_code, previously_insured, vehicle_age, vehicle_damage, annual_premium, policy_sales_channel, vintage)
        result = predict_interest(model, user_input)
        if result == 1:
            st.success("Le client est intéressé par l'assurance.")
        else:
            st.error("Le client n'est pas intéressé par l'assurance.")

    if st.button('Prédire en %'):
        user_input = prepare_input(age, gender, driving_license, region_code, previously_insured, vehicle_age, vehicle_damage, annual_premium, policy_sales_channel, vintage)
        interest_probability = predict_interest_proba(model, user_input)
        st.write(f"Probabilité d'intérêt : {interest_probability:.2%}")
        st.progress(interest_probability)

if __name__ == '__main__':
    main()