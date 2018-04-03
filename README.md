# Simple-Classifiacation-Clustering
Classify using DNN, Knn and SGD <br>
Cluster using K-means, fuzzy C-means and Hierarchy(bottom up)<br>

## Datasets:
1.Car.csv [link](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)<br>
2.CPBL1317_UTF8.csv (Chinese Professional Baseball League) [link](https://cpbl-plus.appspot.com/batting)<br>
2013~2017 players that PA >=G*3.1

**Car evaluation:**

Price|Maint|Doors|Persons|Lug Boot|Safety|Class
------|------|------|------|--------|------|----
vhigh |vhigh |5more |more  |big     |high|vgood
high  |high  |4     |4     |med     |med|good
med   |med   |3     |2     |small   |low|acc
low   |low   |2     |      |        |   |unacc |


**CPBL Batters :**

Year|Player|AVG|OBP|SLG|Team(Class)|
------|------|------|------|--------|------|
2016 |王柏融 |0.414  |0.476 |0.689   |Lamigo|

I delete "Player" column and add "Once go minor league or NPB" and predict team of custom players 

## Requirement
Python coding enviromnent<br>
Modules tensorflow, sklearn, skfuzzy and pandas

## Preprocessing
Use max-min normalization
