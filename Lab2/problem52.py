# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 01:09:13 2017

@author: shammakabir
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#tweets = pd.read_csv('/Users/shammakabir/Downloads/tweets.csv')
tweets = pd.read_csv('C:/Users/Tony/Downloads/tweets.csv')
tweets.head()

def area(location):
    if (location == 'California, USA'):
        return "California"
    if (location == 'New York, USA'):
        return "New York"
    if (location == 'Texas, USA'):
        return "Texas"
    if (location == 'Florida, USA'):
        return "Florida"
    if (location == 'Washington, DC'):
        return "Washington"
    if (location == 'Tennessee, USA'):
        return "Tennessee"
    #else: 
        #return "Other"

def get_candidate(row):
    candidates = []
    text = row["text"].lower()
    if "clinton" in text or "hillary" in text:
        candidates.append("clinton")
    if "trump" in text or "donald" in text:
        candidates.append("trump")
    if "sanders" in text or "bernie" in text:
        candidates.append("sanders")
    return ",".join(candidates)

tweets["candidate"] = tweets.apply(get_candidate,axis=1)
tweets["place"] = tweets["user_location"].apply(area)
tl = {}
for candidate in ["clinton", "sanders", "trump"]:
    tl[candidate] = tweets["place"][tweets["candidate"] == candidate].value_counts()

print tl["clinton"]
    
fig, ax = plt.subplots()
width = .5
x = np.array(range(0, 12, 2))
print len(x)
ax.bar(x, tl["clinton"], width, color='b')
ax.bar(x + width, tl["sanders"], width, color='g')
ax.bar(x + (width * 2), tl["trump"], width, color='r')

ax.set_ylabel('# of tweets')
ax.set_title('Number of Tweets based on Location')
ax.set_xticks(x + (width * 1.5))
ax.set_xticklabels(('Washington', 'Florida', 'California', 'Texas', 'New York', 'Tennessee'))
ax.set_xlabel('Location')
plt.show()