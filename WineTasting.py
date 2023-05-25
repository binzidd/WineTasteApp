############################# Import packages #############################
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, ConfusionMatrixDisplay

# from streamlit_pandas_profiling import st_profile_report
# from ydata_profiling import profile_report

############################# Set parameters #############################
rand_seed=123
Quality_threshold = 6

st.title("Wine Tasting using Machine Learning")

############################# Import Data #############################
wine=pd.read_csv('winequality-red.csv', sep=';')
wine['quality'] = np.where(wine['quality']>=Quality_threshold , 1,0) # 1 stands for decent wine,0 - cooking wine

# Simplify dataset
Drop_cols=['volatile acidity','free sulfur dioxide','pH','sulphates','density']
wine=wine.drop(Drop_cols, axis=1)

wine=wine.rename(columns={"fixed acidity": "Sourness", "citric acid": "Fruitiness",
                          "residual sugar":"Sweetness","chlorides":"Saltiness","total sulfur dioxide":"Preservatives",
                         "alcohol":"Alcohol%","quality":"Quality"})

############################# Sidebar | Quality threshold #############################
st.sidebar.header('Wine Tasting with AI', help='A Machine Learning 101 exercise')
st.sidebar.subheader('Problem Statement')
st.sidebar.write("""You want to impress your wine enthusiast friends by buying them fine Portuguese red wine.

However You ate too much chilli as a child and you can't taste

Can you pick a winner using Machine Learning?
""")

############################# About Data #############################
st.caption("For best experience, please do this exercise on your PC")

st.header(':blue[Step 1 - Understand the data]', help="Have a look through the data first before you proceed further")
st.subheader('About the red wines🍷')
st.write('You went on the internet and found data for ~1,600 bottles of wine previously tasted by wine experts')

st.dataframe(wine)
st.divider()
# with st.expander('Data Profile Report'):
#     pr = wine.profile_report()
#     st_profile_report(pr)


############################# Step 2 - Baseline #############################

st.header(':blue[Step 2 - Set a baseline]📏')
baseline_1,baseline_2,baseline_3=st.columns(3)
with baseline_1:
    bsline1=st.radio("Can you taste?",['Yes','No'],help="Read the problem statement")
    if bsline1=='Yes':
        st.write(':red[❌Are you sure? Try again]')
    else:
        st.write(':green[✔You are right! (Sadly)]')

    with baseline_2:
        bsline2=st.radio("Can you ask your friends?",['Yes','No'])
        if bsline2=='Yes':
            st.write(":red[❌Don't you want to surprise them? Try again]")
        else:
            st.write(':green[✔Of course not, you want to surprise them!]')

    with baseline_3:
        bsline3=st.radio("Do you have other sources?",['Yes','No'])
        if bsline3=='Yes':
            st.write(":red[❌Maybe internet reviews? But you don't trust the fake foodies out there! Try again]")
        else:
            st.write(':green[✔Not really (Sadly)]')

if bsline1=="No" and bsline2=="No" and bsline3=="No":
    st.write('**:green[✔Your baseline is therefore a blind guess🙈]**')
    st.write(':green[Success rate: 50%]')

    st.write("Will Machine Learning perform significantly better than a blind guess?")
    st.write(':green[✔Think so!]')


else:
    st.write(":red[❌Have a go at answering these 3 questions first]")


st.divider()

############################# Step 3 - Decision Tree #############################

st.header(':blue[Step 3 - First ML Model - Decision Tree]')

st.write("Find out more about [decision trees](https://mlu-explain.github.io/decision-tree/)")

###### Separate Features and Label ######
y=wine['Quality']
x_temp=wine.drop(['Quality'],axis=1)

###### Select Featurs ######
st.write('**Human Intelligence** before Artificial Intelligence')
st.write('Have a go at building 3 decision trees with different settings and see how different they are')

DS1, DS2, DS3 = st.columns(3)

with DS1:
    st.subheader("Tree 1🌲")
    selected_features1=st.multiselect("Features",x_temp.columns,'Sourness'
                                     , help="What **inputs/features** help predict wine quality?", key="feat1")
    DS_Depth1 = st.slider('Decision Tree Max Depth', 1, 10, 5, help='How big do you want the decision tree to be?'
                          , key="ds_dep1")


with DS2:
    st.subheader("Tree 2🌲")
    selected_features2=st.multiselect("Features",x_temp.columns,'Sourness', key="feat2")
    DS_Depth2 = st.slider('Decision Tree Max Depth', 1, 10, 5, key="ds_dep2")

with DS3:
    st.subheader("Tree 3🌲")
    selected_features3=st.multiselect("Features",x_temp.columns,'Sourness', key="feat3")
    DS_Depth3 = st.slider('Decision Tree Max Depth', 1, 10, 5, key="ds_dep3")



############################# Decision Tree Models #############################
x1=wine[selected_features1]
x2=wine[selected_features2]
x3=wine[selected_features3]

train_x1, val_x1, train_y1, val_y1 = train_test_split(x1, y, random_state=rand_seed)
train_x2, val_x2, train_y2, val_y2 = train_test_split(x2, y, random_state=rand_seed)
train_x3, val_x3, train_y3, val_y3 = train_test_split(x3, y, random_state=rand_seed)

wine_ds1 = DecisionTreeClassifier(random_state=rand_seed, max_depth=DS_Depth1)
wine_ds2 = DecisionTreeClassifier(random_state=rand_seed, max_depth=DS_Depth2)
wine_ds3 = DecisionTreeClassifier(random_state=rand_seed, max_depth=DS_Depth3)

wine_ds1.fit(train_x1, train_y1)
wine_ds2.fit(train_x2, train_y2)
wine_ds3.fit(train_x3, train_y3)

###### Model prediction and scoring ######
val_predict1=wine_ds1.predict(val_x1)
val_predict2=wine_ds2.predict(val_x2)
val_predict3=wine_ds3.predict(val_x3)

model_acc_ds1=round(accuracy_score(val_y1, val_predict1)*100,2)
model_acc_ds2=round(accuracy_score(val_y2, val_predict2)*100,2)
model_acc_ds3=round(accuracy_score(val_y3, val_predict3)*100,2)

model_f1_ds1=round(f1_score(val_y1, val_predict1)*100,2)
model_f1_ds2=round(f1_score(val_y2, val_predict2)*100,2)
model_f1_ds3=round(f1_score(val_y3, val_predict3)*100,2)

st.subheader("Split Data")
st.write("For simplicity, we have split the data **75:25**")

st.write("""**75%** of the data is used to train the model

**25%** is used to test the model 
""")
st.write("Find out more about [Train, Validation, Testing Splits](https://mlu-explain.github.io/train-test-validation/)")

st.subheader("Train the Model")

st.write(":blue[$◀$]Check your model scores in the **left sidebar**")

st.divider()


############################# See decision tree results #############################
st.header(':blue[See Decision Tree Outcomes]')
st.subheader("Based on Decision Tree 3🌲")

st.caption("Hint: You can go back up and edit the settings of Tree 3")
st.write("What has the model actually predicted?")

actual_output=pd.concat((val_x3,val_y3,
                        pd.DataFrame(val_predict3,index = val_x3.index.copy(),columns=['Predicted Quality']),
                        pd.DataFrame(wine_ds3.predict_proba(val_x3),index = val_x3.index.copy(),columns=['a','Model Score']).drop(['a'],axis=1)
                        ),axis=1)

actual_output['Quality'] = np.where(actual_output['Quality']==1 , '1|🍷','0|💩')
actual_output['Predicted Quality'] = np.where(actual_output['Predicted Quality']==1 , '1|🍷','0|💩')
actual_output['Model Correct?']= np.where(actual_output['Quality']==actual_output['Predicted Quality'],"✅","❌")


actual_output= actual_output[selected_features3+['Model Score', 'Predicted Quality','Quality','Model Correct?']]
actual_output=actual_output.rename(columns={"Quality": "Actual Quality"})
st.write(actual_output)
st.divider()

############################# Confusion Matrix #############################
st.header(':blue[Confusion Matrix]')

cm = confusion_matrix(val_y3, val_predict3)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)


disp.plot()
st.pyplot()

st.divider()
############################# Random Forest #############################
st.header(':blue[Step 4 - Random Forest]')

st.write("Find out more about [Random Forest](https://mlu-explain.github.io/random-forest/)")

x_list=x_temp.columns.to_list()
x_list=", ".join(map(str,x_list))
st.write("Using **all features**: ",x_list)

RF_Dep = st.slider('How many trees do you want in this forest?', 1, 250, 150, key="RF_D")

x4=wine.drop(['Quality'],axis=1)
train_x4, val_x4, train_y4, val_y4 = train_test_split(x4, y, random_state=rand_seed)

wine_rf = RandomForestClassifier(random_state=rand_seed, n_estimators = RF_Dep)
wine_rf.fit(train_x4, train_y4)

###### Model prediction and scoring ######
val_predict4=wine_rf.predict(val_x4)

model_acc_rf=round(accuracy_score(val_y4, val_predict4)*100,2)
model_f1_rf=round(f1_score(val_y4, val_predict4)*100,2)

############################# Model Scoring #############################
st.sidebar.subheader("Model Accuracy %")
st.sidebar.write("*Adjust settings to the right* :blue[▶]")
# st.sidebar.write("Baseline 🙈: ",50.0)
# st.sidebar.write('Decision Tree 1🌲', model_acc_ds1)
# st.sidebar.write('Decision Tree 2🌲', model_acc_ds2)
# st.sidebar.write('Decision Tree 3🌲', model_acc_ds3)
# st.sidebar.write('Random Forest 🌲🌲🌲🌲', model_acc_rf)

model_scores = pd.DataFrame(
    [
        ['Baseline 🙈', 50],
        ['Decision🌲1', model_acc_ds1],
        ['Decision🌲2', model_acc_ds2],
        ['Decision🌲3', model_acc_ds3],
        ['Random🌲🌲🌲', model_acc_rf]

    ],
    columns=['Model', 'Accuracy%'])

base = alt.Chart(model_scores).encode(
    alt.X('Accuracy%', title=""),
    alt.Y('Model', title=''),
    text='Accuracy%',
    opacity=alt.value(0.7),
).properties(
    width=300)
# .configure_axis(
#     grid=False
# )

st.sidebar.altair_chart(base.mark_bar()+base.mark_text(align='left', dx=2))

