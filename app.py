import streamlit as st
import pandas as pd
import pickle

scalers = {
    'StandardScaler': pickle.load(open('StandardScaler.pkl', 'rb')),
    'MinMaxScaler': pickle.load(open('MinMaxScaler.pkl', 'rb')),
    'RobustScaler': pickle.load(open('RobustScaler.pkl', 'rb'))
}


models = {
    'LinearRegression': pickle.load(open('LinearRegression_StandardScaler.pkl', 'rb')),
    'Ridge': pickle.load(open('Ridge_StandardScaler.pkl', 'rb')),
    'Lasso': pickle.load(open('Lasso_StandardScaler.pkl', 'rb')),
    'DecisionTreeRegressor': pickle.load(open('DecisionTreeRegressor_StandardScaler.pkl', 'rb')),
    'RandomForestRegressor': pickle.load(open('RandomForestRegressor_StandardScaler.pkl', 'rb')),
    'GradientBoostingRegressor': pickle.load(open('GradientBoostingRegressor_StandardScaler.pkl', 'rb')),
    'CatBoostRegressor': pickle.load(open('CatBoostRegressor_StandardScaler.pkl', 'rb')),
    'XGBRegressor': pickle.load(open('XGBRegressor_StandardScaler.pkl', 'rb')),

    'SVR': pickle.load(open('SVR_StandardScaler.pkl', 'rb'))
}

st.set_page_config(page_title='Bentonite Prediction of expansive soil', layout='wide')

st.title('Bentonite Prediction of expansive soil')
st.write("Choose a scaler and a model to see the results. You can also input your own data for prediction.")

st.sidebar.header('Model and Scaler Selection')
scaler_choice = st.sidebar.selectbox('Scaler', list(scalers.keys()))
model_choice = st.sidebar.selectbox('Model', list(models.keys()))
st.sidebar.header('User Input Features')
def user_input_features():
    Mt.c = st.sidebar.number_input('Montmorillonite [%] ', value=64)
    LL= st.sidebar.number_input('Liquid limit ', value=260)
    PL= st.sidebar.number_input('Plastic limit [%]', value=220)
    PI= st.sidebar.selectbox('Plasticity index  [%] ', 40)
    Wi  = st.sidebar.number_input('Initial water content [%]', value=15)
    γd  = st.sidebar.number_input('Dry unit weight  [g.m-3] ', value=1.6)
    
    data = {
        'Mt.c': Mt.c,
        'LL': LL,
        'PL': PL,
        'PI': PI,
        'Wi': Wi,
        'γd': γd 
    }
    features = pd.DataFrame(data, index=[0])
    return features
st.header('Prediction Input and Results')
input_df = user_input_features()
if st.button('Predict'):
    scaler = scalers[scaler_choice]
    input_scaled = scaler.transform(input_df)
    model = models[model_choice]
    prediction = model.predict(input_scaled)
    st.subheader('Prediction Results')
    st.write(f'Model used: **{model_choice}**')
    st.write(f'Scaler used: **{scaler_choice}**')
    st.write(f'Swelling Pressure ): ** {prediction[0]:.2f} [Mpa]')
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<div class='sidebar .sidebar-content'>", unsafe_allow_html=True)
