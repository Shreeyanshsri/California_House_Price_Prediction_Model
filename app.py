import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
col=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
# Title
st.title('California Housing Price Prediction')
st.image('https://images.pexels.com/photos/323780/pexels-photo-323780.jpeg')
st.header('Model of Housing Prices To Predict Median House Values In California')
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('''User Must Enter Given values to predict Price:
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')
st.sidebar.title('Select House Features 🏠')
st.sidebar.image('https://png.pngtree.com/thumb_back/fh260/background/20230804/pngtree-an-upside-graph-showing-prices-and-houses-in-the-market-image_13000262.jpg')
temp_df = pd.read_csv('california.csv')

random.seed(52)

all_values = []

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])

with open('house price prediction model.pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]
st.write(pd.DataFrame(dict(zip(col,all_values)),index=[1]))
progress_bar=st.progress(0)
placeholder=st.empty()
place=st.empty()
place.image('https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcjY1bXNwaDB6aHZ5anBramx4eDludDFxaHJicWZ3aWFmY3h1eGpsayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xTk9ZvMnbIiIew7IpW/giphy.gif')

import time
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Price')

if price>0:

    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)

    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    # st.subheader(body)

    st.success(body)
else:
    body = 'Invalid House features Values'
    st.warning(body)
