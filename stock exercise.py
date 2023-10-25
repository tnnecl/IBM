#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install yfinance')


# In[2]:


import yfinance as yf
import pandas as pd


# In[12]:


#code for the company stock
apple = yf.Ticker("AMD")


# In[4]:


get_ipython().system('wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/data/apple.json')

    import json
with open('apple.json') as json_file:
    apple_info = json.load(json_file)
    # Print the type of data variable    
    #print("Type:", type(apple_info))
apple_info


# In[15]:


apple_share_price_data = apple.history(period="max")#change to data frame


# In[18]:


apple_share_price_data.iloc[0]['Volume']


# In[7]:


apple_share_price_data.head()


# In[9]:


apple_share_price_data.reset_index(inplace=True)
apple_share_price_data.head()


# In[10]:


apple_share_price_data.plot(x="Date", y="Open")


# In[11]:


apple.dividends.plot()


# ## webscraping
# 

# In[22]:


get_ipython().system('pip install bs4')
get_ipython().system('pip install html5lib')
get_ipython().system('pip install lxml')


# In[23]:


import pandas as pd
import requests
from bs4 import BeautifulSoup


# In[24]:


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/netflix_data_webpage.html"


# In[27]:


data  = requests.get(url).text
print(data)
soup = BeautifulSoup(data, 'html5lib')
soup


# In[33]:


netflix_data = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])


# In[35]:


for row in soup.find("tbody").find_all('tr'):
    col = row.find_all("td")
    date = col[0].text
    Open = col[1].text
    high = col[2].text
    low = col[3].text
    close = col[4].text
    adj_close = col[5].text
    volume = col[6].text
    
    # Finally we append the data of each row to the table
    netflix_data.loc[len(netflix_data.index)] = {"Date":date, "Open":Open, "High":high, "Low":low, "Close":close, "Adj Close":adj_close, "Volume":volume}
    #netflix_data = netflix_data.append({"Date":date, "Open":Open, "High":high, "Low":low, "Close":close, "Adj Close":adj_close, "Volume":volume}, ignore_index=True)    


# In[36]:


netflix_data.head()


# ### Dashboard

# In[37]:


get_ipython().system('pip install nbformat')


# In[38]:


import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[39]:


def make_graph(stock_data, revenue_data, stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Historical Share Price", "Historical Revenue"), vertical_spacing = .3)
    stock_data_specific = stock_data[stock_data.Date <= '2021--06-14']
    revenue_data_specific = revenue_data[revenue_data.Date <= '2021-04-30']
    fig.add_trace(go.Scatter(x=pd.to_datetime(stock_data_specific.Date, infer_datetime_format=True), y=stock_data_specific.Close.astype("float"), name="Share Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.to_datetime(revenue_data_specific.Date, infer_datetime_format=True), y=revenue_data_specific.Revenue.astype("float"), name="Revenue"), row=2, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($US Millions)", row=2, col=1)
    fig.update_layout(showlegend=False,
    height=900,
    title=stock,
    xaxis_rangeslider_visible=True)
    fig.show()


# In[42]:


tsla = yf.Ticker("TSLA")
tesla_data=tsla.history(period='max')
tesla_data.reset_index(inplace=True)
tesla_data.head()


# In[116]:


html_data=requests.get('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/revenue.htm').text


# In[117]:


soup = BeautifulSoup(html_data, 'html5lib')
print(soup)


# In[120]:


# ct=0
# for i in soup.find_all("table"):
#     print(str(ct)*100)
#     print(i)#.find_all('tr')
#     ct+=1
ta=soup.find_all("table",{'class':"historical_data_table table"})[1]
print(ta)
for i in ta.find_all('tr'):
    
    print(i)
# ta=soup.find_all('th')
# print(ta)

# for i in ta.find_all('tr'):
    
#     print(i)


# In[123]:


tesla_revenue = pd.DataFrame(columns=["Date", 'Revenue'])
for row in soup.find_all("table",{'class':"historical_data_table table"})[1].find_all('tr'):
    col = row.find_all("td")
    if len(col)!=0:
        date=col[0].text
        revenue=col[1].text
    tesla_revenue.loc[len(tesla_revenue)] = {"Date":date,"Revenue":revenue}
tesla_revenue["Revenue"] = tesla_revenue['Revenue'].str.replace(',',"").replace('$','')
tesla_revenue["Revenue"] = tesla_revenue['Revenue'].str.replace('$','')

tesla_revenue.dropna(inplace=True)

tesla_revenue = tesla_revenue[tesla_revenue['Revenue'] != ""]
tesla_revenue.tail()


# In[69]:


gme = yf.Ticker("GME")
gme_data=gme.history(period='max')
gme_data.reset_index(inplace=True)
gme_data.head()


# In[124]:


url=' https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/stock.html'
html_data=requests.get(url).text
soup = BeautifulSoup(html_data, 'html5lib')
print(soup)


# In[126]:


gme_revenue = pd.DataFrame(columns=["Date", 'Revenue'])
for row in soup.find_all("table",{'class':"historical_data_table table"})[1].find_all('tr'):
    col = row.find_all("td")
    if len(col)!=0:
        date=col[0].text
        revenue=col[1].text
    gme_revenue.loc[len(gme_revenue)] = {"Date":date,"Revenue":revenue}
gme_revenue["Revenue"] = gme_revenue['Revenue'].str.replace(',',"").replace('$','')
gme_revenue["Revenue"] = gme_revenue['Revenue'].str.replace('$','')

gme_revenue.dropna(inplace=True)

gme_revenue = gme_revenue[gme_revenue['Revenue'] != ""]
gme_revenue.tail()


# In[128]:


make_graph(tesla_data, tesla_revenue, 'Tesla')


# In[129]:


make_graph(gme_data, gme_revenue, 'GameStop')

