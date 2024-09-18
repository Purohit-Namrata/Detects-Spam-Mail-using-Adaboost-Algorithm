import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("C:/Users/BLAUPLUG/Documents/Python_programs/Detect SpamHam Mail/mail_data.csv")

data=df.where(pd.notnull(df),'')

data.loc[data['Category']=='spam','Category',]=0
data.loc[data['Category']=='ham','Category',]=1


X=data['Message'] # i/p
Y=data['Category'] # o/p

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2)


feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')

model=AdaBoostClassifier()
model.fit(X_train_features,Y_train)
prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)
print("Accuracy on training data: ",accuracy_on_training_data)

prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
print("Accuracy on testing data: ",accuracy_on_test_data)

input_mail=['Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C\'s apply 08452810075over18\'s ']
input_data_features=feature_extraction.transform(input_mail)
prediction=model.predict(input_data_features)

if(prediction[0]==1):
    print("Ham mail")
else:
    print("Spam mail")









