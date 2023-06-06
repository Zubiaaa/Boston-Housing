# Boston Housing Analysis in Power BI

Enabling Python in Power BI

Set-up Python virtual environment with required libraries:

1.	Open a Command Prompt or Power Shell console and navigate to the folder that you want to use as a working directory for your Python code. We will run the following command in order to create the Python environment and install the required libraries:
pipenv install numpy pandas matplotlib seaborn scikit-learn
2.	We want to check the actual Python path for this virtual environment since we will need to refer to that from Power BI:
pipenv –venv
3.	Start Power BI and go to the Options where you should see the Python scripting section on the left. Click on that to open the Python script options. As default Power BI lists the Python environments is has been able to detect in the system. We will need to change these settings since we created a separate virtual environment for Power BI. From Detected Python home directories choose the Other option. Set the Python home directory to the Scripts folder in the path where your virtual environment exists.
Now we are ready to utilize Python code in Power BI so get ready!
Power BI
Importing data using a Python script
We go to the Home tab in the ribbon and click Get data and choose the More option to start our data import. Go to the Other pane where you should find the Python script option. Select that and click Connect. A Python script dialog opens where you can add your own code. Copy and paste the below code to the Script text area and click OK:
import pandas as pd

import numpy as np

from sklearn.datasets import load_boston

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans# utilize the sklearn.datasets package to load the Boston Housing dataset

boston = load_boston()# scale the data to same value range first since PCA

#is sensitive to the scaling of data

sc = StandardScaler()

X = sc.fit_transform(boston.data)# create PCA with n_components=2 to allow visualization in 2 dimensions

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)# divide data into 5 clusters (refer to .ipynb for motivation)

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10)

y_kmeans = kmeans.fit_predict(X_pca)# create pandas dataframe of the housing data for Power BI

columns = np.append(boston.feature_names, ['MEDV', 'PC1', 'PC2', 'CLUSTER'])

data = np.concatenate((boston.data,
                       boston.target.reshape(-1, 1),
                       X_pca,
                       y_kmeans.reshape(-1, 1)),
                      axis=1)
                      
df_housing = pd.DataFrame(data=data, columns=columns)

#we need to convert all columns as string because of different

#decimal separator in Python (.) and Finnish locale (,) that Power BI uses.

#comment out below line if Power BI uses dot as a decimal separator.

df_housing = df_housing.astype('str')# create pandas dataframe of the pca data for Power BI

columns = np.append(boston.feature_names, ['VARRATIO'])

data = np.concatenate((pca.components_,
                       pca.explained_variance_ratio_.reshape(-1, 1)),
                      axis=1)
                      
df_pca = pd.DataFrame(data=data, columns=columns, index=['PC1', 'PC2'])

df_pca = df_pca.astype('str')

In the next window we are able to choose which datasets to import. Select both the df_housing as well as the df_pca datasets and click Load. Next we will go and make the final adjustments to our import data. Click Edit Queries in the ribbon.
The next target is to convert every column to numbers.
Power BI with decimal point
Choose all columns and let Power BI Detect Data Type from the Transform tab in the ribbon in order to get the correct data types. This could be done by clicking Detect Data Type. Repeat these steps for both datasets.
The final step is to add an Index Column to the df_housing dataset. This can be done from the Add Column tab in the ribbon. When that has been done go to the Home tab and click Close & Apply.
Create custom visualizations using Python:
We go back to the Reports view to start working on our visualizations. The plan is to visualize the clusters on a chart using the principal components as axes. A scatter plot is a good alternative for this visualization. Click on the Scatter chart to add it to the page. Drag the df_housing columns to the visualizations pane in the following way:
•	Drag PC1 to the X Axis.
•	Drag PC2 to the Y Axis.
•	Drag CLUSTER to the Legend to visualize how the clusters are grouped.
•	Drag Index to Details as this will remove the aggregation in the chart.
You might have a different cluster order (coloring) due to the random selection of initial centroids in the k-means clustering.
Next we will visualize how each feature affects each of the principal components. This information is available in the df_pca dataset. We will show this information through a heatmap. The Seaborn Python library provides an easy way to create a heatmap so we'll add a Python visual to the page. Power BI might warn you about script visuals, click Enable to continue. Each feature needs to be separately dragged to the Data fields area. Drag each column from the dataset except VARRATIO. Copy the following code snippet to the code area and click Run script:

import matplotlib.pyplot as plt

import seaborn as snsdataset.index = ['PC1', 'PC2']

plt.figure(figsize=(8, 2))

plt.xticks(rotation=45)

data = dataset.loc['PC1', :].to_frame().sort_values(by='PC1').transpose()

sns.heatmap(data,
            cmap='plasma',
            square=True,
            annot=True,
            cbar=False,
            yticklabels='')
            
plt.show()
You should now see a heatmap. Depending on your screen resolution you might have to hide the script pane to see the visual.
Repeat the same step to create a heatmap for the second principal component but use below code snippet instead to use the data from the second principal component and make a vertical visualization that can be placed on the left side of the scatter plot:

import matplotlib.pyplot as plt

import seaborn as snsdataset.index = ['PC1', 'PC2']

plt.figure(figsize=(2, 8))

data = dataset.loc['PC2', :].to_frame().sort_values(by='PC2', ascending=False)

sns.heatmap(data,
            cmap='plasma',
            square=True,
            annot=True,
            cbar=False,
            xticklabels='')
            
plt.show()
We have now visualized the identified clusters of data with regards to the two principal components that together explain the majority of the variance in the data. The heat maps display which features affects each principal component in a positive or negative manner with regards to the principal component value. Let’s make some final touches to make the report look a little nicer:
•	Change one of the gray cluster colors to purple.
•	Add a heading and reference to data.
•	Add a visualization that shows the average median price per cluster.
•	Add a text box explaining the features.
•	Add a visualization stating the variance explained for each principal component.
This is how my version ended up looking:
 


The Distance feature has the biggest impact for lowering the PC1 value by a small margin, so let's have a closer look at that to see how it behaves in the different clusters. We create a new page and do the following changes:
•	Add a descriptive heading.
•	Add a Scatter chart with X Axis value is DIS column, Y Axis value is MEDV and Legend and Details fields are set up like the previous scatter plot.
•	Right click the DIS column and choose New group and create bins of size 1.
•	Add a Stacked column chart showing the distribution of distance values by using the newly created binned column of DIS. Sort the chart in ascending order by the Axis instead of the count.
This is how my version of the report turned out:
 

