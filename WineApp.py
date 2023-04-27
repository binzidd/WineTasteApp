# Import packages
import numpy as np
import pandas as pd
import streamlit as st


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay

# from streamlit_pandas_profiling import st_profile_report
# from ydata_profiling import profile_report

###### Sidebar | Quality threshold ######
st.sidebar.header('Wine Tasting with AI', help='A Machine 101 exercise')
st.sidebar.subheader('Problem Statement')
st.sidebar.write("You want to impress your wine enthusiast friends by buying them fine Portuguese red wine.")
st.sidebar.write("However You ate too much chilli as a child and you can't taste")
st.sidebar.write("Can you pick a winner using Machine Learning?")


###### About Data ######
st.header(':blue[Step 1 - Understand the data]')
st.subheader('About the red wines🍷')
st.write('You went on the internet and found data for ~1,600 bottles of wine previously tasted by wine experts')

## Set parameters
rand_seed=123
Quality_threshold = 6

###### Import Data ######
wine=pd.read_csv('winequality-red.csv', sep=';')
wine['quality'] = np.where(wine['quality']>=Quality_threshold , 1,0) # 1 stands for decent wine,0 - cooking wine

# Simplify dataset
Drop_cols=['volatile acidity','free sulfur dioxide','pH','sulphates','density']
wine=wine.drop(Drop_cols, axis=1)

wine=wine.rename(columns={"fixed acidity": "Sourness", "citric acid": "Fruitiness",
                          "residual sugar":"Sweetness","chlorides":"Saltiness","total sulfur dioxide":"Preservatives",
                         "alcohol":"Alcohol%","quality":"Quality"})


with st.expander('Show me the data'):
    st.dataframe(wine)

with st.expander('Data Profile Report'):
    st.write('Placeholder')
    # pr = wine.profile_report()
    # st_profile_report(pr)

st.header(':blue[Step 2 - Set a baseline]')
baseline_1,baseline_2,baseline_3=st.columns(3)
with baseline_1:
    bsline1=st.radio("Can you taste before you buy?",['Yes','No'])
    if bsline1=='Yes':
        st.write(':red[Are you sure? Try again]')
    else:
        st.write(':green[You are right! (Sadly)]')

with baseline_2:
    bsline2=st.radio("Can you ask your friends?",['Yes','No'])
    if bsline2=='Yes':
        st.write(":red[Don't you want to surprise them? Try again]")
    else:
        st.write(':green[Of course not, you want to surprise them!]')

with baseline_3:
    bsline3=st.radio("Do you have other sources?",['Yes','No'])
    if bsline3=='Yes':
        st.write(":red[Maybe internet reviews? But you don't trust the fake foodies out there! Try again]")
    else:
        st.write(':green[Not really (Sadly)]')

if bsline1=="No" and bsline2=="No" and bsline3=="No":
    st.write('**:green[Your baseline is therefore a blind guess]:sunglasses:**')
    st.write(':green[Success rate: 50%]')


st.header(':blue[Step 3 - Feature selection]')

###### Separate Features and Label ######
y=wine['Quality']
x_temp=wine.drop(['Quality'],axis=1)


###### Select Featurs ######
st.subheader('Human Intelligence before Artificial Intelligence')
feat_col1, feat_col2 = st.columns(2)

with feat_col1:
    selected_features=st.multiselect("What inputs/features help predict wine quality?",x_temp.columns,'Sourness')

###### Train model ######
DS_dep1,DS_dep2 = st.columns(2)
with DS_dep1:
    DS_Depth = st.slider('Decision Tree Max Depth',1,10,5)


x=wine[selected_features]
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=rand_seed)

winemodel = DecisionTreeClassifier(random_state=rand_seed, max_depth=DS_Depth)
winemodel.fit(train_x, train_y)

winemodel.predict(val_x)

###### Model prediction and scoring ######
val_predictions=winemodel.predict(val_x)
model_acc=round(accuracy_score(val_y, val_predictions)*100,2)
model_f1=round(f1_score(val_y, val_predictions)*100,2)

st.write('Your model accuracy is',accuracy_score(val_y, val_predictions))

