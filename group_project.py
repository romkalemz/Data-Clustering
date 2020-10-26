
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sn
import numpy as np
from sklearn.model_selection import KFold

#read data from file
pathToCustomerData = "~/Downloads/bank/bank.csv"
customer_data = pd.read_csv(pathToCustomerData, sep=";")

attr_selected = ['age', 'job', 'education', 'default', 'loan', 'contact']

#select desired atributtes
dt = pd.DataFrame(data=customer_data, columns=attr_selected)


#switch to numerical vals for categorical attrs
age = dt['age'].unique().tolist()
job = dt['job'].unique().tolist()
education = dt['education'].unique().tolist()
credit = dt["default"].unique().tolist()
loan = dt['loan'].unique().tolist()
contact = dt['contact'].unique().tolist()
val_list = [age, job, education, credit, loan, contact]

list_age = dt['age'].values.tolist()
list_job = dt['job'].values.tolist()
list_education = dt['education'].values.tolist()
list_credit = dt['default'].values.tolist()
list_loan = dt['loan'].values.tolist()
list_contact = dt['contact'].values.tolist()
lists = [list_age, list_job, list_education, list_credit, list_loan, list_contact]

num_job = []
num_education = []
num_credit = []
num_loan = []
num_contact = []
num_lists = [list_age, num_job, num_education, num_credit, num_loan, num_contact]

#create numerical values list
for i in range(1, len(lists)):
    print(lists[i])
    for item in lists[i]:
        num_lists[i].append(val_list[i].index(item))
    print(num_lists[i])


#add num lists to dataframe
dt['num_job'] = num_lists[1]
dt['num_education'] = num_lists[2]
dt['num_credit'] = num_lists[3]
dt['num_loan'] = num_lists[4]
dt['num_contact'] = num_lists[5]


num_attr = ['age', 'num_job', 'num_education', 'num_credit', 'num_loan', 'num_contact']
#create dataframe with only numerical vals
X = pd.DataFrame(data=dt, columns=num_attr)

print(X)

#elbow plot to find number of K
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
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

y_km = km.fit_predict(X)
print(y_km)

plt.scatter(dt.loc[y_km==0,'age'], dt.loc[y_km==0, 'job'], s=50, c="green", label="cluster 1", edgecolor="black", marker='s')
plt.scatter(dt.loc[y_km==1,'age'], dt.loc[y_km==1,'job'], s=50, c="blue", label="cluster 2", edgecolor='black', marker='s')

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


#correlation matrix
corr_matrix = X.corr()
sn.heatmap(corr_matrix, annot=True)

plt.show()

#cross validation
