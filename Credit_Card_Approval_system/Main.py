import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

df_app=pd.read_csv('application_record.csv')
df_app.head()

df_app.info()

df_app.isnull().sum()

df_app[df_app.duplicated()]

df_credit=pd.read_csv('credit_record.csv')
df_credit.head()

df_credit.STATUS.unique()





"""# **TASK 1** - Exploratory Data Analysis

The status is divided into different categories:

*  0: 1-29 days past due - This status indicates that the payment is overdue by 1 to 29 days.

*  1: 30-59 days past due - This status indicates that the payment is overdue by 30 to 59 days.

*  2: 60-89 days overdue - This status indicates that the payment is overdue by 60 to 89 days.

* 3: 90-119 days overdue - This status indicates that the payment is overdue by 90 to 119 days.

*  4: 120-149 days overdue - This status indicates that the payment is overdue by 120 to 149 days.

*  5: Overdue or bad debts, write-offs for more than 150 days - This status indicates that the payment is overdue for more than 150 days or considered a bad debt that needs to be written off.

*  C: paid off that month - This status indicates that the loan has been paid off during the month.

*  X: No loan for the month - This status indicates that there is no loan associated with that particular month.
"""

df_credit.shape

df_credit.describe()

df_credit.STATUS.value_counts().plot.bar()

"""### Creating a data frame wherein the status column is separated according to all the categories"""

credit_grouped=pd.get_dummies(data=df_credit,columns=['STATUS'],
                              prefix='',prefix_sep='').groupby('ID')[sorted(df_credit['STATUS'].unique().tolist())].sum()
credit_grouped=credit_grouped.rename(columns=
                      {'0':'pastdue_1_29','1':'pastdue_30_59','2':'pastdue_60_89','3':'pastdue_90_119','4':'pastdue_120_149','5':'pastdue_over_150',
                       'C':'paid_off','X':'no_loan',})

overall_pastdue=['pastdue_1_29','pastdue_30_59',	'pastdue_60_89',	'pastdue_90_119'	,'pastdue_120_149',	'pastdue_over_150']
credit_grouped['number_of_months']=df_credit.groupby('ID')['MONTHS_BALANCE'].count()
credit_grouped['over_90']=credit_grouped[['pastdue_90_119'	,'pastdue_120_149'	,'pastdue_over_150']].sum(axis=1)
credit_grouped['less_90']=credit_grouped[['pastdue_1_29','pastdue_30_59',	'pastdue_60_89']].sum(axis=1)
credit_grouped['overall_pastdue']=credit_grouped[overall_pastdue].sum(axis=1)
credit_grouped['paid_pastdue_diff']=credit_grouped['paid_off']- credit_grouped['overall_pastdue']
credit_grouped.head()

"""###Good and Bad Customers
**Good customer: credit card approved**:
--> If the difference between number of times customer paid off and the number of lately paid is more than 4 or no_loan is equal to number of months , he/she's credit card is approved.

--> In this model , I have also considered the person as approved if the difference is between 0 to 4  

**(So, basically the difference should be greater than 0 and if he/she has not taken any loan then card is approved and he is a good customer)**

--> If the customer doesn't achive this conditions then he is a **bad customer: credit card not approved**.
"""

target=[]
for index,row in credit_grouped.iterrows() :
  if row['paid_pastdue_diff'] >=4 or (row ['no_loan']==row['number_of_months']) :
    target.append(1)
  elif row['paid_pastdue_diff'] >=0 and row['paid_pastdue_diff'] <4 :
    target.append(1)
  else:
    target.append(0)

credit_grouped['good_or_bad']=target
credit_grouped['good_or_bad'].value_counts()

"""### **This classification solves the problem of unbalanced data as well which was mentioned in the problem statement.**"""



credit_grouped.head()

"""# Q1: What is the distribution of credit card approval status in the dataset? How many applications are approved and how many are not approved?"""

keys = ['1', '0']
plt.pie(credit_grouped['good_or_bad'].value_counts(),autopct='%1.2f%%',labels=keys)

"""## OBSERVATIONS

Of the total, **52.19%** are **approved** while the rest **47.81%** are **unapproved**.


"""



"""### **Merging the data set**"""

features=['no_loan',	'number_of_months',	'over_90',	'less_90',	'overall_pastdue'	,'paid_pastdue_diff','good_or_bad']
most_important_features=credit_grouped.loc[:,features]
customers_df=pd.merge(df_app,most_important_features,on='ID')
customers_df.index=customers_df['ID']
customers_df=customers_df.drop('ID',axis=1)
customers_df

"""##Q2 : Is there any relationship between the applicant's gender and credit card approval status? Can you calculate the approval rate for each gender category (male and female) and determine if there is a significant difference using a hypothesis test?"""

relation=customers_df.groupby(['CODE_GENDER','good_or_bad']).size().reset_index().rename(columns={0:'Count'})
relation.head()

relation_df=customers_df[['CODE_GENDER','good_or_bad']]
relation_df.head()

from scipy import stats
gender_approval = pd.crosstab(relation_df['CODE_GENDER'], relation_df['good_or_bad'])
approval_rate_male = gender_approval.loc['M', 1] / gender_approval.loc['M'].sum()
approval_rate_female = gender_approval.loc['F', 1] / gender_approval.loc['F'].sum()

# CHI SQUARE TEST
chi2, p_value, dof, expected = stats.chi2_contingency(gender_approval)

print('Approval Rate for Male:', approval_rate_male)
print('Approval Rate for Female:', approval_rate_female)
print('Chi-square Statistic:', chi2)
print('Degrees of Freedom:', dof)
print('P-value:', p_value)

"""### OBSERVATIONS


* The approval rates for males and females are relatively close, with males having a slightly higher approval rate.

* **Chi-square Test Results:**
The **chi-square statistic** is approximately **0.99**, with **1 degree of freedom**.
The associated **p-value** is approximately **0.3196** which is greater than commonly used level of 0.05.

* This means that we fail to reject the null hypothesis and conclude that there is not enough evidence to suggest a significant association between gender and credit card approval status.

* **Interpretation:**
The results indicate that **gender may not be a significant factor** in determining credit card approval decisions.
Other factors might have a stronger influence on credit card approval rates.








"""

x=customers_df.loc[:,:'paid_pastdue_diff']
y=customers_df['good_or_bad']
x

"""### **SPLITTING DATA INTO THE RATIO OF 80:20 (train:test)**"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

"""### DATA CLEANING
1.FLAG_MOBILE because it has one value

2.FLAG_WORK_PHONE ,because it has small effect to determine bad or good

3.FLAG_PHONE ,because it has small effect to determine bad or good

4.FLAG_EMAIL ,because it has small effect to determine bad or good
"""

x_train=x_train.drop(['FLAG_MOBIL','FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL'],axis=1)

x_train

"""### **DATA PREPROCESSING**
**Converting DAYS_BIRTH to AGE and DAYS_EMPLOYED to WORK_YEARS**
"""

x_train['AGE']=(x_train['DAYS_BIRTH']/365)*-1
x_train['AGE']=x_train['AGE'].apply(lambda v : int(v))
x_train['WORK_YEARS']=x_train['DAYS_EMPLOYED']/365
x_train['WORK_YEARS']=x_train['WORK_YEARS'].apply(lambda v : int(v*-1) if v <0 else 0)
x_train=x_train.drop(columns=['DAYS_BIRTH','DAYS_EMPLOYED'])
x_train

"""### Removing COLUMN CNT_CHILDREN as it is being already considered in CNT_FAM_MEMBERS"""

x_train=x_train.drop(columns=['CNT_CHILDREN'])

"""##Q3: Are there any missing values in the dataset for variables like "Years_employed" and "Education_type"? Can you identify variables with missing data and suggest an appropriate strategy, such as imputation or removal, for handling them?"""

customers_df.isnull().sum()

from sklearn.impute import SimpleImputer
imputer =SimpleImputer(strategy='most_frequent')
x_imputed=pd.DataFrame(imputer.fit_transform(x_train),index=x_train.index,columns=x_train.columns)
# Imputer change the type of the features so we should reset it again
x_imputed=x_imputed.astype(x_train.dtypes)

x_imputed['no_loan']=x_imputed['no_loan'].astype('int64')
x_imputed['CNT_FAM_MEMBERS']=x_imputed['CNT_FAM_MEMBERS'].astype('int64')
x_imputed.info()

"""###OBSERVATIONS
There were **11214 missing values** in **OCCUPATION COLUMN.**

I have handled it using **Imputation method** by filing the places with the **most frequent values**.

##Q4: Can you visualize the distribution of "Total_income" for approved and not approved credit card applications using a suitable plot, such as a boxplot or histogram? Are there any noticeable differences in income between the two groups?
"""

total_income_approved = customers_df[customers_df['good_or_bad'] == 1]['AMT_INCOME_TOTAL']
total_income_not_approved = customers_df[customers_df['good_or_bad'] == 0]['AMT_INCOME_TOTAL']
fig, ax = plt.subplots(figsize=(8, 6))
boxplot = ax.boxplot([total_income_approved, total_income_not_approved],
                     labels=['Approved', 'Not Approved'],
                     patch_artist=True)
colors = ['lightblue', 'lightgrey']
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)
ax.set_title('Distribution of Total Income', fontsize=14)
ax.set_xlabel('Approval Status', fontsize=12)
ax.set_ylabel('Total Income', fontsize=12)
ax.set_facecolor('lightyellow')

plt.figure(figsize=(8, 6))
plt.hist(total_income_approved, bins=30, alpha=0.5, label='Approved', color='yellow')
plt.hist(total_income_not_approved, bins=30, alpha=0.5, label='Not Approved', color='green')
plt.title('Distribution of Total Income', fontsize=14)
plt.xlabel('Total Income', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend()
plt.show()

plt.show()

"""###OBSERVATIONS
There is **not a major difference in average income values** of the approved and non approved ones as seen from the box plot. However, as inferred from outliers the **maximum income** values for both is **same** while the **minimum income values for approved is a bit less as compared to non approved** .
"""



"""##Q5:What is the most common income type and education type among the applicants? Can you create a bar plot or pie chart to visualize the distribution of income types and education types?"""

customers_df['NAME_INCOME_TYPE'] = customers_df['NAME_INCOME_TYPE'].astype('category')
customers_df['NAME_EDUCATION_TYPE'] = customers_df['NAME_EDUCATION_TYPE'].astype('category')
most_common_income_type = customers_df['NAME_INCOME_TYPE'].value_counts().idxmax()
most_common_education_type = customers_df['NAME_EDUCATION_TYPE'].value_counts().idxmax()
print("Most Common Income Type: {}".format(most_common_income_type))

# Plot of income types
plt.figure(figsize=(10, 6))
sns.countplot(data=customers_df, x='NAME_INCOME_TYPE', palette='viridis')
plt.xlabel('Income Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Income Types', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

"""### OBSERVATION
The most **common income** type is **Working**.
"""

print("Most Common Education Type: {}".format(most_common_education_type))

plt.figure(figsize=(8, 8))
sns.set(style='whitegrid')
sns.color_palette("viridis")
education_type_counts = customers_df['NAME_EDUCATION_TYPE'].value_counts()

plt.pie(education_type_counts.values, labels=education_type_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Education Types', fontsize=14)
plt.axis('equal')
plt.show()

"""###OBSERVATION
The most **common education type** is **secondary(68%)**.

"""



"""# **TASK 2** - Classification/Regression

##Dealing with categorical value
"""

from pandas.core.algorithms import value_counts
categorical_df=x_imputed.select_dtypes('object')
categorical_df.nunique()

"""**We can notice that only OCCUPATION_TYPE feature has a high cadinality and the rest is low cadinality .
So I used target encoding for OCCUPATION_TYPE and one hot encoding for the rest.**
"""

!pip install category_encoders

from category_encoders import MEstimateEncoder
target_encoder=MEstimateEncoder(m=5,cols=['OCCUPATION_TYPE'])
# training the encoder with the 0.25 of the data to prevent overfitting
x_encode=x_imputed.sample(frac=0.25)
y_encode=y_train[x_encode.index]
target_encoder.fit(x_encode,y_encode)

x_encoded=target_encoder.transform(x_imputed)
x_encoded['OCCUPATION_TYPE'].unique()

"""###**Standardising the data**"""

numiric_data=x_encoded._get_numeric_data()
numiric_data

from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
numiric_data_scaled=scaler.fit_transform(numiric_data)
numiric_data_scaled=pd.DataFrame(numiric_data_scaled,index=numiric_data.index,columns=numiric_data.columns)
numiric_data_scaled

x_encoded[numiric_data_scaled.columns]=numiric_data_scaled[numiric_data_scaled.columns]
x_standarized=x_encoded.copy()

x_train=pd.get_dummies(x_standarized)
x_train

x_train.info()

"""### Applying same process on test dataset"""

x_test=x_test.drop(['FLAG_MOBIL','FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL'],axis=1)

x_test['AGE']=(x_test['DAYS_BIRTH']/365)*-1
x_test['AGE']=x_test['AGE'].apply(lambda v : int(v))
x_test['WORK_YEARS']=x_test['DAYS_EMPLOYED']/365
x_test['WORK_YEARS']=x_test['WORK_YEARS'].apply(lambda v : int(v*-1) if v <0 else 0)
x_test=x_test.drop(columns=['DAYS_BIRTH','DAYS_EMPLOYED'])

x_test=x_test.drop(columns=['CNT_CHILDREN'])

x_test_imputed=pd.DataFrame(imputer.transform(x_test),index=x_test.index,columns=x_test.columns)

x_test_imputed=x_test_imputed.astype(x_test.dtypes)
x_test_imputed['no_loan']=x_test_imputed['no_loan'].astype('int64')
x_test_imputed['CNT_FAM_MEMBERS']=x_test_imputed['CNT_FAM_MEMBERS'].astype('int64')


x_test_encoded=target_encoder.transform(x_test_imputed)

numiric_test_data=x_test_encoded._get_numeric_data()
numiric_test_data_scaled=scaler.transform(numiric_test_data)
numiric_test_data_scaled=pd.DataFrame(numiric_test_data_scaled,index=numiric_test_data.index,columns=numiric_test_data.columns)
x_test_encoded[numiric_test_data_scaled.columns]=numiric_test_data_scaled[numiric_test_data_scaled.columns]
x_test_standarized=x_test_encoded.copy()

x_test=pd.get_dummies(x_test_standarized)
x_test

x_test.info()

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import itertools

#x_train=x_train.drop(['NAME_EDUCATION_TYPE_Academic degree'],axis=1)

"""##**APPLYING LOGISTIC REGRESSION ON THE MODEL**"""



model = LogisticRegression(C=0.01,
                           random_state=0,max_iter=1000)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))

"""##I have set the max iter to 1000 as it was ealier showing convergence error.

# **The accuracy is 96.4%**
"""

model = LogisticRegression(C=0.01, random_state=0, max_iter=1000)
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)

train_accuracy = accuracy_score(y_train, y_train_pred)
print('Training Accuracy: {:.5f}'.format(train_accuracy))

"""## Checked on training data also to check that there is no overfitting. Also the value of C is taken as 0.01 to prevent the same."""

print(pd.DataFrame(confusion_matrix(y_test,y_predict)))

sns.set_style('white')

class_names = ['0', '1']
cm = confusion_matrix(y_test, y_predict)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix: Logistic Regression')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

thresh = cm_normalized.max() / 2.

for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
    plt.text(j, i, format(cm_normalized[i, j], '.2f'),
             horizontalalignment="center",
             color="white" if cm_normalized[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))

"""**Precision** represents the accuracy of the positive predictions. In this case, it means that around **97.45% of the predicted positive cases are actually true positive cases**.

**Recall**, also known as sensitivity or true positive rate, measures the model's ability to correctly identify positive cases. With a recall score of 0.9474, it suggests that the model can identify **94.74% of the actual positive cases**.

The **F1 score** is the harmonic mean of precision and recall, providing a balanced measure of the model's performance. The F1 score of **96.07% indicates a good balance between precision and recall**.
"""

