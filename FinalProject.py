import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    classification_report
)
import streamlit as st

# Call the dataframe "s"  
s = pd.read_csv(r"C:\Users\mattd\Desktop\School\School Documents\Mod 2\Programming II\Final Project\social_media_usage.csv")

# check the dimensions of the dataframe
#s.shape

def clean_sm(x):             # Define a function called clean_sm that takes one input, x,
    x = np.where(x == 1,     # and uses `np.where` to check whether x is equal to 1.
             1,              # If it is, make the value of x = 1,
             0)              # otherwise make it 0.
    return x                 # Return x.

df = pd.DataFrame(           #Create a toy dataframe with three rows and two columns
     {'col 1': [1, 2, 3],
     'col 2': [6, 1, 5]})

clean_sm(df)                 # and test your function to make sure it works as expected

# Create a new dataframe called "ss"
ss = pd.DataFrame({'sm_li': clean_sm(s['web1h']), #target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn
       'income': np.where(s['income']<=9,         
                          s['income'], 
                          np.nan),
       'education': np.where(s['educ2']<=8, 
                             s['educ2'], 
                             np.nan),
       'parent': clean_sm(s['par']),
       'married': clean_sm(s['marital']),
       'female': clean_sm(s['gender']),
       'age': np.where(s['age'] <= 98, 
                       s['age'], 
                       np.nan)}) 

# Drop any missing values. 
ss.dropna(inplace=True)

#ss.describe()

#ss.dtypes

ss['target_label'] = ss['sm_li'].map({0: "Not a LinkedIn User", 1: "LinkedIn User"})

y = 'sm_li'
x = ['income', 'education', 'parent', 'married', 'female', 'age']

train_data, test_data = train_test_split(
    ss,
    test_size=0.20,
    random_state=210,
    )

#train_data.dtypes

#Regression Model
z = LogisticRegression(class_weight="balanced", random_state=210)
model = z.fit(train_data[x], train_data[y])

y_pred = model.predict(test_data[x])

cm = pd.DataFrame(confusion_matrix(test_data[y], y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"])

#cm

precision = round(float(cm.at["Actual positive", "Predicted positive"] / cm["Predicted positive"].values.sum()),4)
recall = round(float(cm.at["Actual positive", "Predicted positive"] / cm.loc["Actual positive"].values.sum()),4)
f1 = 2 * ((precision * recall) / (precision + recall))

comp = pd.DataFrame([precision, recall, f1],
                    columns=["Values"],
                    index=["Precision:", "Recall:", "F-1 Score:"])

#comp

print(classification_report(test_data[y], y_pred))

new_data = pd.DataFrame({
    "income": [8, 8],
    "education": [6, 6],
    "parent": [0, 0],
    "married": [1, 1],
    "female": [1, 1],
    "age": [42, 82]
})

#new_data

new_data['sm_li'] = model.predict(new_data)

#new_data

st.title("LinkedIn User Prediction Tool")

tab1, tab2, tab3 = st.tabs(["Predict", "Sample Data Visualizations", "Model Performance Metrics"])

st.divider()

with tab1:

    col1, col2 = st.columns(2)

    with col1:
        st.header("About this App")
        st.markdown(
            """
            **Instructions:**
            1. Select the demographic characteristics of the individual using the dropdown menus and slider.
            2. The app will display whether the individual is predicted to be a LinkedIn user along with the probability of LinkedIn use.
            """
        )

    ss['Income_Label'] = ss['income'].map({1: "Less than $10,000",
                                        2: "10 to under $20,000",
                                        3: "20 to under $30,000",
                                        4: "30 to under $40,000",
                                        5: "40 to under $50,000",
                                        6: "50 to under $75,000",    
                                        7: "60 to under $70,000",
                                        8: "75 to under $100,000",
                                        9: "100 to under $150,000",
                                        10: "150,000 or more"})
    optionIncome = st.selectbox(
        'Choose Income Level:',
        ss['Income_Label'].unique())

    ss['Education_Label'] = ss['education'].map({1: "Less than high school",    
                                                2: "High school incomplete",
                                                3: "High school graduate",
                                                4: "Some college, no degree",
                                                5: "Two-year associate degree",
                                                6: "Four-year college degree",
                                                7: "Some postgraduate or professional schooling, no degree",
                                                8: "Postgraduate or professional degree"})

    optionEducation = st.selectbox(
        'Choose Education Level:',  
        ss['Education_Label'].unique())

    ss['Parent_Label'] = ss['parent'].map({0: "Not a Parent",
                                        1: "Parent"})

    optionParent = st.selectbox(
        'Parental Status:',
        ss['Parent_Label'].unique())

    ss['Married_Label'] = ss['married'].map({0: "Not Married", 
                                            1: "Married"})

    optionMarried = st.selectbox(
        'Marital Status:',
        ss['Married_Label'].unique())

    ss['Gender_Label'] = ss['female'].map({0: "Male",  
                                        1: "Female"})

    optionGender = st.selectbox(
        'Gender:',
        ss['Gender_Label'].unique())    

    age = st.slider('Select Age:', 18, 98, 30)

    input_data = pd.DataFrame({
        "income": [ss.loc[ss['Income_Label'] == optionIncome, 'income'].mode()[0]],
        "education": [ss.loc[ss['Education_Label'] == optionEducation, 'education'].mode()[0]],
        "parent": [ss.loc[ss['Parent_Label'] == optionParent, 'parent'].mode()[0]],
        "married": [ss.loc[ss['Married_Label'] == optionMarried, 'married'].mode()[0]],
        "female": [ss.loc[ss['Gender_Label'] == optionGender, 'female'].mode()[0]],
        "age": [age]})

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]   # probability of being a LinkedIn user

    # ---- Display results ----
    with col2:

        st.header("Prediction")
        if pred == 1:
            st.success("**This individual *is predicted* to be a LinkedIn User.**")
        else:
            st.error("**This individual *is predicted* to NOT be a LinkedIn User.**")
        st.write(f"**Predicted Probability of LinkedIn Use:** {prob:.3f}")

with tab2:

    plt.style.use("dark_background")
    
    st.markdown(
        "Density plots of key predictors, split by LinkedIn usage status."
    )

    # Loop over each predictor in x and create a density plot
    for col in x:
        # Create a figure for Streamlit
        fig, ax = plt.subplots()

        sns.kdeplot(
            data=ss,
            x=col,
            hue="target_label",   # use human-readable labels
            fill=True,
            common_norm=False,
            ax=ax,
            legend=False
        )

        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Density")
        ax.set_title(f"{col.capitalize()} Distribution by LinkedIn Use")
        fig.legend(title="LinkedIn Use", labels=["Not a LinkedIn User", "LinkedIn User"], loc="upper left", bbox_to_anchor=(1, 0.5))
        

        st.pyplot(fig)
        plt.close(fig)

with tab3:

    st.markdown(
        "Model performance metrics on the test dataset."
    )

    st.subheader("Confusion Matrix")
    st.dataframe(cm)

    st.subheader("Precision, Recall, and F1 Score")
    st.dataframe(comp)

    y_prob = model.predict_proba(test_data[x])[:, 1]
    fpr, tpr, thresholds = roc_curve(test_data[y], y_prob)
    auc = roc_auc_score(test_data[y], y_prob)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")

    st.pyplot(fig2)
    plt.close(fig2) 