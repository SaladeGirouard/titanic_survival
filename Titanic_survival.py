import pandas as pd 
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


st.set_page_config(
     page_title="Titanic",
     page_icon="🚢")

### PREPARATION DU DATASET
link = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/titanic.csv"
df_titanic = pd.read_csv(link)
df_titanic['Survived'] = df_titanic['Survived'].apply(lambda x: "Survived" if x == 1 else "Dead")
df_titanic['Sex'] = df_titanic['Sex'].factorize()[0]

### MACHINE LEARNING
X = df_titanic[['Pclass', 'Sex', 'Age']]
y = df_titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1912)

model = LogisticRegression().fit(X_train,y_train)


### TITRE
st.markdown("<h1 style='text-align: center;'>Bienvenue à bord du Titanic !</h1>", unsafe_allow_html=True)

### IMAGE
col1, col2, col3 = st.columns([1,3,1])

with col1:
    st.write("")

with col2:
    st.image("https://media1.giphy.com/media/1eAg3lPj1dJYY/giphy.gif?cid=ecf05e474kh7qxlh4630px9oav0dz21cczsm9jz56zmwggg3&rid=giphy.gif&ct=g")

with col3:
    st.write("")

### DESCRIPTION
st.write("C'est le plus grand paquebot du monde. On l'appelle l'insubmersible, Dieu lui-même ne pourrait pas couler ce bateau !")

st.write("Pouvez-vous me montrer votre ticket, s'il vous plaît ?")

### INPUT

nom = st.text_input("Votre nom")
genre = st.selectbox("Votre genre", options=("Femme","Homme","Autre"))
age = st.text_input("Votre age")
classe = st.selectbox("Dans quelle classe êtes-vous situé ?", options=(3,2,1))


if age != "":
       
    
    if genre == "Femme":
        genrebin = 1
    elif genre == "Homme":
        genrebin = 0
    else :
        genrebin = 0
    
    classe=int(classe)
    genrebin=int(genrebin)
    age=int(age)
    survivance = np.array([classe, genrebin, age]).reshape(1,3)
    probasurvie = model.predict_proba(survivance)


    st.markdown("<h1 style='text-align: center;'>PATATRA !</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,3,1])

    with col1:
        st.write("")

    with col2:
        st.image("https://media1.giphy.com/media/OJw4CDbtu0jde/giphy.gif?cid=ecf05e47u5fj96oqkwv5bkdf87gfkfwnmvqx112x6ymfrudn&rid=giphy.gif&ct=g")

    with col3:
            st.write("")

    st.write("Aïe, le Titanic est en train de couler...")            
            
    st.write("")
    st.write("")
    st.write("")
    
    ### EN CAS DE SURVIE :
    if probasurvie[0][1] > 0.5:
        st.write("Bravo,",nom,", vous avez survécu !")
        st.write("Vos chances de survie étaient de", round(probasurvie[0][1]*100),"%")


        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.write("")

        with col2:
            st.image("https://media2.giphy.com/media/4ryp9Ihw0BEyc/giphy.gif?cid=ecf05e478n5hpvvg4dargtx91jdw5su18bzywf7cqauhmrw5&rid=giphy.gif&ct=g")

        with col3:
            st.write("")    

    ### EN CAS DE DECES :
    else: 
        st.write("Pas de chance,",nom,", vous êtes mort !")
        st.write("Vos chances de survie étaient de", round(probasurvie[0][1]*100),"%") 

        col1, col2, col3 = st.columns([1,3,1])

        with col1:
            st.write("")

        with col2:
            st.image("https://media2.giphy.com/media/14vXAPPJPZRzsA/giphy.gif?cid=ecf05e478n5hpvvg4dargtx91jdw5su18bzywf7cqauhmrw5&rid=giphy.gif&ct=g")

        with col3:
            st.write("") 
















