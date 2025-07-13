import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import cross_val_score
model=joblib.load('model.pkl')
df=sns.load_dataset("penguins")
def About_Model():
    st.snow()
    st.title("About Model")
    df=sns.load_dataset("penguins")
    df = df.drop(columns=["sex"])
    X = df.drop(columns=["species"])
    y = df["species"]
    X = X.dropna()
    y = y[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    island_col = ['island']
    log_col = ['body_mass_g']
    num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    preprocessor = ColumnTransformer(transformers=[
        ('island_ohe', OneHotEncoder(drop='first'), island_col),
        ('log_body_mass', FunctionTransformer(np.log1p), log_col),
        ('scaler', StandardScaler(), num_cols)
    ], remainder='passthrough')
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000,multi_class='multinomial',solver='lbfgs'))
    ])
    pipeline.fit(X_train, y_train)
    y_pred=pipeline.predict(X_test)
    
    st.write("**I built a logistic regression model to classify penguin species using features like island, bill length, bill depth, flipper length, and body mass. The model takes these inputs, applies preprocessing (like one-hot encoding, log transform, and scaling), and predicts whether the penguin is Adelie, Gentoo, or Chinstrap. This model helps in understanding how physical traits relate to species classification.**")
    
    st.write("**Cross Val Score :**")
    st.code(cross_val_score(pipeline,X,y,scoring='accuracy',cv=5))
    st.write("**Model Accuracy :**")
    st.code(accuracy_score(y_test,y_pred))
    st.write("**Classification Report :**")
    st.code(classification_report(y_test,y_pred))
    st.divider()
def Time_to_predict():
    st.title("Time To Predict")
    
    
    model=joblib.load('model.pkl')
    st.subheader("Enter Parameters")
    island = st.selectbox("Select Island", ['Biscoe', 'Dream', 'Torgersen'])
    bill_length = st.number_input("Bill Length (mm)", min_value=0.0)
    bill_depth = st.number_input("Bill Depth (mm)", min_value=0.0)
    flipper_length = st.number_input("Flipper Length (mm)", min_value=0.0)
    body_mass = st.number_input("Body Mass (g)", min_value=0.0)
    # Predict button
    if st.button("Predict Species"):
        st.balloons()
        import pandas as pd
        input_df = pd.DataFrame({
            'island': [island],
            'bill_length_mm': [bill_length],
            'bill_depth_mm': [bill_depth],
            'flipper_length_mm': [flipper_length],
            'body_mass_g': [body_mass]
        })

        # Prediction
        prediction = model.predict(input_df)
        st.success(f"Predicted Penguin Species: *{prediction[0]}*")

def dataset_visualization():
    st.snow()
    X=df.iloc[:,1:]
    st.title("Data Visualizations")
    st.divider()
    st.header("Penguin Dataset Visualization")
    st.markdown("""
    <style>
    .custom-codebox {
        background-color: #1e1e1e;  /* Dark background for code look */
        color: #d4d4d4;             /* Light text */
        padding: 16px;
        border-radius: 10px;
        font-family: monospace;
        white-space: pre;
        overflow-x: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='custom-codebox'>{df.iloc[0:6,1:].to_string(index=False)}</div>", unsafe_allow_html=True)
    st.write("")
    st.write("**Dataset Shape :**")
    st.code(df.iloc[:,1:].shape)
    st.write("")
    st.write("**Target Values :**")
    st.code(df["species"].value_counts())
    st.subheader("Dataset NULL counts and dtypes")
    null_col, dtype_col = st.columns(2)
    with null_col:
        st.code(df.iloc[:,1:].isna().sum())
    with dtype_col:
        st.code(df.iloc[:,1:].dtypes)
    st.write("**Distribution of Species W.R.T Gender :**")
    st.code(df.groupby('species')['sex'].value_counts())
    st.write("**Since According to above distribution it seems that there is no effect of gender on species type therefore I will drop Sex Column and in only two rows values are missing i will drop those two rows as well**")
    st.divider()
    fig=sns.pairplot(df,hue="species")
    st.pyplot(fig)
    st.divider()
    fig, ax = plt.subplots(nrows=1, ncols=4)
    fig.set_figwidth(15)
    for i in range(1,5):
        sns.boxplot(x=X.iloc[..., i], ax=ax[i-1])
    st.pyplot(fig)

    # plotting the histplot for each feature
    fig, ax = plt.subplots(nrows=1, ncols=4)
    fig.set_figwidth(15)
    for i in range(1,5):
        sns.histplot(x=X.iloc[..., i], kde=True, ax=ax[i-1])
    st.pyplot(fig)

    st.markdown("""### Observations From Visualization:
    * Use of StandardScaler on bill_length,bill_depth
    * Use of Roburst Scaler or StandardScaler on flipper_length
    * Use of Log Transform on body_mass and then StandardScaler""")

    



pg=st.navigation([
    st.Page(About_Model,title="About Model"),
    st.Page(dataset_visualization,title="Dataset Visualization"),
    st.Page(Time_to_predict,title="Time To Predict"),
    
])
pg.run()