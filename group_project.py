
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

#read data from file
pathToCustomerData = "~/Downloads/bank/bank.csv"
customer_data = pd.read_csv(pathToCustomerData, sep=";")
print(customer_data)

#select desired atributtes
X = pd.DataFrame(data=customer_data, columns=["age", "job", "education", "default", "loan", "contact"])

credit = X["default"].unique()
list_credit = X["default"].values.tolist()
list_age = X['age'].values.tolist()


#list for numerical vals of default
num_list_credit = []

for item in list_credit:
    if item == "yes":
        num_list_credit.append(1)
    else:
        num_list_credit.append(0)

#add the column to selected data
X['num_cred'] = num_list_credit


#elbow plot to find number of K
cred_age =[]
for i in range(len(list_age)):
    new_item = [num_list_credit[i], list_age[i]]
    cred_age.append(new_item)
x = pd.DataFrame(cred_age)
print(x)

#find value of k
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(cred_age)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
#plt.savefig('images/11_03.png', dpi=300)
plt.show()


#number of clusters based off elbow method is k = 2
km = KMeans(n_clusters=2,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)

y_km = km.fit_predict(cred_age)

plt.scatter(X.loc[y_km==0,'num_cred'], X.loc[y_km==0, 'age'], s=50, c="green", label="cluster 1", edgecolor="black", marker='s')
plt.scatter(X.loc[y_km==1,'num_cred'], X.loc[y_km==1,'age'], s=50, c="blue", label="cluster 2", edgecolor='black', marker='s')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
#plt.savefig('images/11_02.png', dpi=300)
plt.show()
