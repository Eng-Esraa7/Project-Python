#age

#anaemia

#creatinine_phosphokinase ( انزيم)فوسفوكيناز الكرياتينين

#diabetes داء السكري

#ejection_fraction نسبه الدم المضخه من القلب
#يتم التعبير عن EF الخاص بك كنسبة مئوية. يمكن أن يكون مؤشر EF أقل من المعدل الطبيعي علامة على فشل القلب.

#platelets الصفائح الدموية 
#serum_creatinine الكرياتينين في الدم

#sex

#smoking

#time

#DeathEvent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

data=pd.read_csv("data.csv")
data.info()
data.describe()

# histogarm show who is more death
#sns.histplot(x="smoking",data=data,kde=True,hue="DEATH_EVENT")
d2=data.head(50)
plt.plot(d2["DEATH_EVENT"])
plt.plot(d2["smoking"])

plt.title("ralationship between smoking and death")
plt.xlabel("smoking")
plt.ylabel("death-event")
plt.show()

#0 smoke and 0 death
#0 smoke and 1 death
#1 somke and 0 death
#1 smoke and 1 death
data.sort_values(["serum_creatinine"],ascending=False)
d1=data.head(10)
plt.pie(d1["serum_creatinine"])
plt.show()


# importing libraries

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn import tree
from sklearn.tree import export_graphviz 


#readig dataset
#data = pd.read_csv('F:\(data scince)\data.csv')
x = data[["age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction","platelets","serum_creatinine","sex","smoking"]].values
y = data["DEATH_EVENT"]

##############
#Multiple LinearRegression
model = LinearRegression() 
clf = model.fit(x,y)
print ('Coefficient: ', clf.coef_)
print ('Intercept: ', clf.intercept_)

predictions = model.predict(x)
for index in range(len(predictions)):
    print ('Actual :',y[index], 'Predicted: ',
round(predictions[index]), 'Weight: ', x[index,0])
    
#############################################

#Decision Tree
model = tree.DecisionTreeRegressor() 
model.fit(x,y)
predictions = model.predict(x)

print (model.feature_importances_)

for index in range(len(predictions)):
    print('Actual : ', y[index], 'Predicted: ', predictions[index])
  
export_graphviz(model , out_file = 'DecisionTree.dot')

#from graphviz import Source
#Source.from_file("DecisionTree.dot")

############################################

#knn
model = neighbors.KNeighborsRegressor(n_neighbors=3) 
model.fit(x,y)
predictions = model.predict(x)

for index in range(len(predictions)):
    print('Actual: ', y[index], 'Prtdicttd :	', round(predictions[index]), 'Weight', x[index,0])

#########################
#### lof   

from sklearn.neighbors import LocalOutlierFactor
from numpy import where
import matplotlib.pyplot as plt
import pandas as pd
# load data
#data=pd.read_csv("data.csv")

x=data['creatinine_phosphokinase']
x=x.values.reshape(-1,1)

# contamnation = 60(value<100 -->outliers)/300=0.2    
lof=LocalOutlierFactor(n_neighbors=2,contamination=0.2)
y_predict=lof.fit_predict(x)
print(y_predict)
outliers_index=where(y_predict==-1)
print(outliers_index)
values=x[outliers_index]
print("******************outliers**********")
print(values)
plt.scatter(x[:,0], x[:,0],color='blue')
plt.scatter(values[:,0],values[:,0], color='orange')
plt.show()

#############################
#Text Mining

import numpy as np
import pandas as pd
import string
import nltk
nltk.download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
df = open("projectda.txt","r", encoding="utf8")
hh = df.readlines()

stemmer = PorterStemmer()
lem = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stops = set(stopwords.words('english'))
for i in range(0, 19):
    hh[i] = hh[i].lower()
    res1 = hh[i].translate(str.maketrans('', '', string.punctuation))
    tok = word_tokenize(res1)
    res = [i for i in tok if not i in stops]
    fin_res1 = []
    fin_res2 = []
    for word in res:
        fin_res1.append(stemmer.stem(word))
    print(fin_res1)
    for word in res:
        fin_res2.append(lem.lemmatize(word))
    print(fin_res2)

    print(sia.polarity_scores(hh[i]))
