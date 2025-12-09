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
s = pd.read_csv("social_media_usage.csv")


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
       'income': np.where(s['income']<=9,         #income (ordered numeric from 1 to 9, above 9 considered missing)     
                          s['income'], 
                          np.nan),
       'education': np.where(s['educ2']<=8,       # education (ordered numeric from 1 to 8, above 8 considered missing)
                             s['educ2'], 
                             np.nan),
       'parent': clean_sm(s['par']),              #parent (binary)
       'married': clean_sm(s['marital']),         #married (binary)
       'female': clean_sm(s['gender']),           #female (binary)
       'age': np.where(s['age'] <= 98,            #age (numeric, above 98 considered missing)
                       s['age'], 
                       np.nan)}) 

# Drop any missing values. 
ss.dropna(inplace=True)

# Perform exploratory analysis to examine how the features are related to the target.

#ss.describe()

#ss.dtypes

ss['target_label'] = ss['sm_li'].map({0: "Not a LinkedIn User", 1: "LinkedIn User"})

# Create a target vector (y) and feature set (X)
y = 'sm_li'
x = ['income', 'education', 'parent', 'married', 'female', 'age']

# Split the data into training and test sets. Hold out 20% of the data for testing.
train_data, test_data = train_test_split(
    ss,
    test_size=0.20,
    random_state=210,
    )

# Explain what each new object contains and how it is used in machine learning

#   After splitting the dataframe 'ss', the dataframe 'test_data' contains a subset of 20% of the data from ss that will be utilized to compare 
# the predictions from the model and determine how well the model performs. The dataframe 'train_data' contains the remaining 80% of the data from ss  
# that will be used for training the model, meaning that the variables will be observed and a mathematical relationship developed. More precisely,
# because sm_li was assigned to y that will be the value we are predicting for (whether an individual is a LinkedIn user). The rest of the variables, 
# assigned to x, will be used as the variables to determine the probability that an individual is a LinkedIn user. 

#train_data.dtypes

# Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# Regression Model
z = LogisticRegression(class_weight="balanced", random_state=210)
model = z.fit(train_data[x], train_data[y])

# Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then 
y_pred = model.predict(test_data[x])

# generate a confusion matrix from the model
# Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

cm = pd.DataFrame(confusion_matrix(test_data[y], y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"])

#cm

# Interpret the confusion matrix and explain what each number means.

#   The confusion matrix displays the performance of the logistics regression model by calculating the number of positive and negative observations 
# (LinkedIn users or non-user) and the number of predicted positive and negative observations. The top left intersection (Predicted negative and 
# Actual negative) shows that the model correctly predicted 113 individuals as not being LinkedIn users, or a true negative. The top right intersection (Predicted positive
# and Actual negative) shows that the model incorrectly predicted 51 individuals as being LinkedIn users when they were not, or a false positive. The bottom left intersection
# (Predicted negative and Actual positive) shows that the model incorrectly predicted 22 individuals as not being LinkedIn users when they actually were, or a false negative.
# The bottom right intersection (Predicted positive and Actual positive) shows that the model correctly predicted 66 individuals as being LinkedIn users, or a true positive.

# Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand.

precision = round(float(cm.at["Actual positive", "Predicted positive"] / cm["Predicted positive"].values.sum()),4)
recall = round(float(cm.at["Actual positive", "Predicted positive"] / cm.loc["Actual positive"].values.sum()),4)
f1 = 2 * ((precision * recall) / (precision + recall))

comp = pd.DataFrame([precision, recall, f1],
                    columns=["Values"],
                    index=["Precision:", "Recall:", "F-1 Score:"])

#comp

# Discuss each metric and give an actual example of when it might be the preferred metric of evaluation.

#   The precision of the model states that 56.4% of the predicted LinkedIn users are actually LinkedIn users; the recall states that the model accurately predicts 75% of LinkedIn users; the F-1 Score provides a 
# metric that evaluates both and is especially useful when classes are imbalanced in the data (Geeks for Geeks, 2025). Precision is more important when the risk of when the risk of a false positive is greater than 
# the risk of a false negative, and recall is more important in the inverse scenario. 

# After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

print(classification_report(test_data[y], y_pred))

# Use the model to make predictions. How does the probability change if another person is 82 years old, but otherwise the same?
new_data = pd.DataFrame({
    "income": [8, 8],            # high income (e.g. income=8)
    "education": [6, 6],         # high level of education (e.g. 7)
    "parent": [0, 0],            # non-parent
    "married": [1, 1],           # married
    "female": [0, 0],            # female
    "age": [42, 82]              # 42 years old
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

    ss['Gender_Label'] = ss['female'].map({0: "Female",  
                                        1: "Male"})

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



# Citations

# “F1 Score in Machine Learning.” GeeksforGeeks, July 23, 2025. https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/. 

# “Precision and Recall in Machine Learning.” GeeksforGeeks, August 2, 2025. https://www.geeksforgeeks.org/machine-learning/precision-and-recall-in-machine-learning/. 