from turtle import color
from attr import attr
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from bokeh.plotting import figure
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#from forecast import generate_forecast, calculate_smape

def main():
    #st.title('Forecast Gold Price')
    st.markdown("<h1 style='text-align: center; color: #8cdf27; text-decoration:underline;'>Bankruptcy Prevention</h1>", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()    

df = pd.read_excel('bankruptcy-prevention.xlsx')

st.sidebar.title("1. Data")
if st.sidebar.checkbox("Display data", False):
    st.subheader("Showing 'Bankruptcy' dataset")
    st.table(df)    


df1 = pd.read_excel("bankruptcy-prevention.xlsx")


st.sidebar.title("2. Options")
page = st.sidebar.radio('select', ["Classification Analysis"])

                    
if page=="Classification Analysis":
                                       
    genre = st.sidebar.radio("Select",('Evaluation Metrics', 'Predict Values')) 
    if genre == 'Evaluation Metrics':
        # Label encoder
        from sklearn.preprocessing import LabelEncoder
        encode = LabelEncoder()
        df['class'] = encode.fit_transform(df['class'])
        X = df.drop(['class'],axis=1)
        y = df['class']
        #train, test = train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
      
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        model_LR = LogisticRegression(random_state= 42)
        results_LR = cross_val_score(model_LR, X, y, cv=kfold, scoring="accuracy")
      
        model_LR.fit(X_train, y_train)
        preds = model_LR.predict(X_test)
        f1_LR = f1_score(y_test, preds)
        precision_LR = precision_score(y_test, preds)
        recall_LR = recall_score(y_test, preds) 

        st.write("Training Accuracy: ", model_LR.score(X_train, y_train))
        st.write('Testing Accuarcy: ', model_LR.score(X_test, y_test))
        st.subheader('C (using Logistic Regression)')
        st.write('F1 score is: ', f1_LR)
        st.write('Precision is: ', precision_LR)
        st.write('Recall is: ', recall_LR)


    if genre == 'Predict Values':
        st.subheader("Choose only these values:")
        st.markdown("* **0.0 - Low**")
        st.markdown("* **0.5 - Medium**")
        st.markdown("* **1.0 - High**")

        def user_input_features():
            industrial_risk = st.selectbox("Industrial Risk",[0.0,0.5,1.0])
            management_risk = st.selectbox("Management Risk",[0.0,0.5,1.0])
            financial_flexibility = st.selectbox("Financial flexibility",[0.0,0.5,1.0])
            credibility = st.selectbox("Credibility",[0.0,0.5,1.0])
            competitiveness = st.selectbox("Competitiveness",[0.0,0.5,1.0])
            operating_risk = st.selectbox("Operating Risk",[0.0,0.5,1.0])

            data = {'industrial_risk':industrial_risk, 'management_risk':management_risk, 'financial_flexibility':financial_flexibility,'credibility':credibility,'competitiveness':competitiveness, 'operating_risk':operating_risk}
            features = pd.DataFrame(data,index = [0])
            return features
            
        df2 = user_input_features()

        if st.button("Predict", key="Predict"):
            encode = LabelEncoder()
            df['class'] = encode.fit_transform(df['class'])
            X = df.drop(['class'],axis=1)
            y = df['class']
            #train, test = train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
            # Building a model on logisticregression
            
            from sklearn.linear_model import LogisticRegression
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            model_LR = LogisticRegression(random_state= 42)
            model_LR.fit(X_train,y_train)   
            y_pred = model_LR.predict(df2)
            st.header("Result :")
            st.subheader("The Company is...")
            #st.write(y_pred) 
            if y_pred==0:
                image = Image.open('bankruptcy.jpg')
                st.image(image, width=300)
            else:
                image = Image.open('nonbankrupt.PNG')
                st.image(image, width=400)

            



                                
    
    
