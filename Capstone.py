#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Requests allows us to make HTTP requests which we will use to get data from an API
import requests
# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Datetime is a library that allows us to represent dates
import datetime

# Setting this option will print all collumns of a dataframe
pd.set_option('display.max_columns', None)
# Setting this option will print all of the data in a feature
pd.set_option('display.max_colwidth', None)


# In[2]:


# Takes the dataset and uses the rocket column to call the API and append the data to the list
def getBoosterVersion(data):
    for x in data['rocket']:
       if x:
        response = requests.get("https://api.spacexdata.com/v4/rockets/"+str(x)).json()
        BoosterVersion.append(response['name'])


# In[3]:


# Takes the dataset and uses the launchpad column to call the API and append the data to the list
def getLaunchSite(data):
    for x in data['launchpad']:
       if x:
         response = requests.get("https://api.spacexdata.com/v4/launchpads/"+str(x)).json()
         Longitude.append(response['longitude'])
         Latitude.append(response['latitude'])
         LaunchSite.append(response['name'])


# In[4]:


# Takes the dataset and uses the payloads column to call the API and append the data to the lists
def getPayloadData(data):
    for load in data['payloads']:
       if load:
        response = requests.get("https://api.spacexdata.com/v4/payloads/"+load).json()
        PayloadMass.append(response['mass_kg'])
        Orbit.append(response['orbit'])


# In[5]:


# Takes the dataset and uses the cores column to call the API and append the data to the lists
def getCoreData(data):
    for core in data['cores']:
            if core['core'] != None:
                response = requests.get("https://api.spacexdata.com/v4/cores/"+core['core']).json()
                Block.append(response['block'])
                ReusedCount.append(response['reuse_count'])
                Serial.append(response['serial'])
            else:
                Block.append(None)
                ReusedCount.append(None)
                Serial.append(None)
            Outcome.append(str(core['landing_success'])+' '+str(core['landing_type']))
            Flights.append(core['flight'])
            GridFins.append(core['gridfins'])
            Reused.append(core['reused'])
            Legs.append(core['legs'])
            LandingPad.append(core['landpad'])


# In[7]:


spacex_url="https://api.spacexdata.com/v4/launches/past"
response = requests.get(spacex_url)
print(response.content)


# In[8]:


static_json_url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'


# In[11]:


response = requests.get(static_json_url)

# Decode the response content as JSON
data_json = response.json()

# Convert the JSON data to a Pandas DataFrame
data = pd.json_normalize(data_json)
data.head()


# In[12]:


# Lets take a subset of our dataframe keeping only the features we want and the flight number, and date_utc.
data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]

# We will remove rows with multiple cores because those are falcon rockets with 2 extra rocket boosters and rows that have multiple payloads in a single rocket.
data = data[data['cores'].map(len)==1]
data = data[data['payloads'].map(len)==1]

# Since payloads and cores are lists of size 1 we will also extract the single value in the list and replace the feature.
data['cores'] = data['cores'].map(lambda x : x[0])
data['payloads'] = data['payloads'].map(lambda x : x[0])

# We also want to convert the date_utc to a datetime datatype and then extracting the date leaving the time
data['date'] = pd.to_datetime(data['date_utc']).dt.date

# Using the date we will restrict the dates of the launches
data = data[data['date'] <= datetime.date(2020, 11, 13)]


# In[13]:


#Global variables 
BoosterVersion = []
PayloadMass = []
Orbit = []
LaunchSite = []
Outcome = []
Flights = []
GridFins = []
Reused = []
Legs = []
LandingPad = []
Block = []
ReusedCount = []
Serial = []
Longitude = []
Latitude = []


# In[15]:


getBoosterVersion(data)


# In[16]:


BoosterVersion[0:5]


# In[17]:


getLaunchSite(data)
getPayloadData(data)
getCoreData(data)


# In[18]:


launch_dict = {'FlightNumber': list(data['flight_number']),
'Date': list(data['date']),
'BoosterVersion':BoosterVersion,
'PayloadMass':PayloadMass,
'Orbit':Orbit,
'LaunchSite':LaunchSite,
'Outcome':Outcome,
'Flights':Flights,
'GridFins':GridFins,
'Reused':Reused,
'Legs':Legs,
'LandingPad':LandingPad,
'Block':Block,
'ReusedCount':ReusedCount,
'Serial':Serial,
'Longitude': Longitude,
'Latitude': Latitude}


# In[22]:


df = pd.DataFrame(launch_dict)
df.head()


# In[23]:


df.info()


# In[24]:


df.describe()


# In[25]:


data_falcon9 = df.loc[df['BoosterVersion'] == 'Falcon 9']


# In[64]:


data_falcon9.loc[:,'FlightNumber'] = list(range(1, data_falcon9.shape[0]+1))
data_falcon9


# In[63]:


data_falcon9.shape()


# In[27]:


data_falcon9.isnull().sum()


# In[32]:


payload_mean = data_falcon9['PayloadMass'].mean()

data_falcon9.loc[:, 'PayloadMass'] = data_falcon9['PayloadMass'].fillna(payload_mean)


# In[33]:


data_falcon9.isnull().sum()


# In[36]:


data_falcon9.to_csv('./dataset_part_1.csv', index=False)


# In[34]:


import sys

import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import pandas as pd


# In[35]:


def date_time(table_cells):
    """
    This function returns the data and time from the HTML  table cell
    Input: the  element of a table data cell extracts extra row
    """
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
    """
    This function returns the booster version from the HTML  table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=''.join([booster_version for i,booster_version in enumerate( table_cells.strings) if i%2==0][0:-1])
    return out

def landing_status(table_cells):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=[i for i in table_cells.strings][0]
    return out


def get_mass(table_cells):
    mass=unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass=mass[0:mass.find("kg")+2]
    else:
        new_mass=0
    return new_mass


def extract_column_from_header(row):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    if (row.br):
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()
        
    colunm_name = ' '.join(row.contents)
    
    # Filter the digit and empty names
    if not(colunm_name.strip().isdigit()):
        colunm_name = colunm_name.strip()
        return colunm_name    


# In[37]:


static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"


# In[39]:


response = requests.get(static_url)


# In[45]:


html_content = response.text

soup = BeautifulSoup(html_content, 'html.parser')


# In[65]:


soup.title


# In[47]:


all_tables = soup.find_all('table')

# Iterate through each table to find the table headers
html_tables = []
for table in all_tables:
    html_tables.append(table)


# In[54]:


first_launch_table = html_tables[2]


# In[55]:


column_names = []

for th in first_launch_table.find_all('th'):
    column_names.append(extract_column_from_header(th))


# In[56]:


print(column_names)


# In[59]:


launch_dict= dict.fromkeys(column_names)

# Remove an irrelvant column
del launch_dict['Date andtime (UTC)']

# Let's initial the launch_dict with each value to be an empty list
launch_dict['Flight No.'] = []
launch_dict['Launch site'] = []
launch_dict['Payload'] = []
launch_dict['Payload mass'] = []
launch_dict['Orbit'] = []
launch_dict['Customer'] = []
launch_dict['Launch outcome'] = []
# Added some new columns
launch_dict['Version Booster']=[]
launch_dict['Booster landing']=[]
launch_dict['Date']=[]
launch_dict['Time']=[]


# In[60]:


extracted_row = 0
#Extract each table 
for table_number,table in enumerate(soup.find_all('table',"wikitable plainrowheaders collapsible")):
   # get table row 
    for rows in table.find_all("tr"):
        #check to see if first table heading is as number corresponding to launch a number 
        if rows.th:
            if rows.th.string:
                flight_number=rows.th.string.strip()
                flag=flight_number.isdigit()
        else:
            flag=False
        #get table element 
        row=rows.find_all('td')
        #if it is number save cells in a dictonary 
        if flag:
            extracted_row += 1
            # Flight Number value
            # TODO: Append the flight_number into launch_dict with key `Flight No.`
            #print(flight_number)
            datatimelist=date_time(row[0])
            
            # Date value
            # TODO: Append the date into launch_dict with key `Date`
            date = datatimelist[0].strip(',')
            #print(date)
            
            # Time value
            # TODO: Append the time into launch_dict with key `Time`
            time = datatimelist[1]
            #print(time)
              
            # Booster version
            # TODO: Append the bv into launch_dict with key `Version Booster`
            bv=booster_version(row[1])
            if not(bv):
                bv=row[1].a.string
            print(bv)
            
            # Launch Site
            # TODO: Append the bv into launch_dict with key `Launch Site`
            launch_site = row[2].a.string
            #print(launch_site)
            
            # Payload
            # TODO: Append the payload into launch_dict with key `Payload`
            payload = row[3].a.string
            #print(payload)
            
            # Payload Mass
            # TODO: Append the payload_mass into launch_dict with key `Payload mass`
            payload_mass = get_mass(row[4])
            #print(payload)
            
            # Orbit
            # TODO: Append the orbit into launch_dict with key `Orbit`
            orbit = row[5].a.string
            #print(orbit)
            
            # Customer
            # TODO: Append the customer into launch_dict with key `Customer`
            customer = row[6].a.string
            #print(customer)
            
            # Launch outcome
            # TODO: Append the launch_outcome into launch_dict with key `Launch outcome`
            launch_outcome = list(row[7].strings)[0]
            #print(launch_outcome)
            
            # Booster landing
            # TODO: Append the launch_outcome into launch_dict with key `Booster landing`
            booster_landing = landing_status(row[8])
            #print(booster_landing)
            


# In[151]:


df= pd.DataFrame({ key:pd.Series(value) for key, value in launch_dict.items() })


# In[62]:


df.to_csv('./spacex_web_scraped.csv', index=False)


# In[68]:


data_falcon9.isnull().sum()/len(df)*100


# In[69]:


data_falcon9.dtypes


# In[71]:


launch_site_counts = data_falcon9['LaunchSite'].value_counts()
launch_site_counts 


# In[72]:


orbit_counts = data_falcon9['Orbit'].value_counts()
orbit_counts


# In[73]:


landing_outcomes = data_falcon9['Outcome'].value_counts()
landing_outcomes


# In[74]:


for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)
bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
bad_outcomes


# In[76]:


landing_class = [0 if outcome in bad_outcomes else 1 for outcome in data_falcon9['Outcome']]
landing_class


# In[79]:


data_falcon9['Class']=landing_class
data_falcon9[['Class']].head(8)


# In[80]:


data_falcon9.head(5)


# In[81]:


data_falcon9["Class"].mean()


# In[89]:


import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns


# In[87]:


df=data_falcon9
df.head()


# In[90]:


sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show()


# In[95]:


sns.catplot(x='FlightNumber', y='LaunchSite', hue="Class", data=df, aspect = 5)
plt.show()


# In[96]:


sns.catplot(x='PayloadMass', y='LaunchSite', hue="Class", data=df, aspect = 5)
plt.show()


# In[98]:


orbit_success_rate = data_falcon9.groupby('Orbit')['Class'].mean()
orbit_success_rate.plot(kind='bar', ylabel='Success Rate', xlabel='Orbit', title='Success Rate of Each Orbit')
plt.show()


# In[100]:


sns.scatterplot(x='FlightNumber', y='Orbit', hue='Class', data=data_falcon9)
plt.show()


# In[101]:


sns.scatterplot(x='PayloadMass', y='Orbit', hue='Class', data=data_falcon9)
plt.show()


# In[104]:


year=[]
def Extract_year():
    for i in data_falcon9["Date"]:
        year.append(i.year)
    return year
Extract_year()
data_falcon9['Date'] = year
data_falcon9.head()


# In[107]:


success_rate = data_falcon9.groupby('Date')['Class'].mean()

# Plot the line chart
plt.plot(success_rate.index, success_rate.values, marker='o')


# In[108]:


features = data_falcon9[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()


# In[110]:


features_one_hot = pd.get_dummies(features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'])


# In[114]:


features_one_hot.shape


# In[111]:


numeric_columns = features_one_hot.select_dtypes(include=['int', 'float']).columns
features_one_hot[numeric_columns] = features_one_hot[numeric_columns].astype('float64')


# In[122]:


# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier


# In[123]:


def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 


# In[124]:


Y = data_falcon9['Class'].to_numpy()


# In[125]:


X=features_one_hot
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)


# In[126]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[127]:


Y_test.shape


# In[128]:


parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}


# In[129]:


parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()


# In[130]:


logreg_cv = GridSearchCV(lr, parameters, cv=10)


# In[131]:


logreg_cv.fit(X_train,Y_train)


# In[132]:


print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# In[150]:


accuracy = logreg_cv.score(X_test, Y_test)
print(accuracy)


# In[134]:


yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# In[135]:


parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()


# In[136]:


svm_cv = GridSearchCV(svm, parameters, cv=10)

# Fit the object to find the best parameters
svm_cv.fit(X_train,Y_train) 


# In[137]:


print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)


# In[149]:


accuracy = svm_cv.score(X_test, Y_test)
print(accuracy)


# In[139]:


yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# In[140]:


parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()


# In[141]:


tree_cv = GridSearchCV(tree, parameters, cv=10)

# Fit the object to find the best parameters
tree_cv.fit(X_train,Y_train) 


# In[142]:


print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)


# In[147]:


accuracy = tree_cv.score(X_test, Y_test)
print(accuracy)
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# In[144]:


parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()
knn_cv = GridSearchCV(KNN, parameters, cv=10)

# Fit the object to find the best parameters
knn_cv.fit(X_train,Y_train) 
print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)


# In[146]:


accuracy = knn_cv.score(X_test, Y_test)
print(accuracy)
yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




