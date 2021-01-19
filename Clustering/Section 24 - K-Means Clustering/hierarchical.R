#Hierarchical Clustering

#Importing the dataset
dataset = read.csv("Mall_Customers.csv")
X = dataset[4:5]

#Using the dendrogram to find the optimal number of clusters
dendrogram = hclust(dist(X, method = "euclidean"), method = "ward.D")
plot(dendrogram,
     main = paste("Dendrogram"),
     xlab = "Customers",
     ylab = "Euclidean distances")

#Fitting the hierarchical clustering to the mall dataset
hc = hclust(dist(X, method = "euclidean"), method = "ward.D")
y_hc = cutree(hc, 5)
#plot(y_hc,
#     main = paste("Dendrogram"),
#     xlab = "Customers",
#     ylab = "Euclidean distances")

#Visualizing the clusters
library(cluster)
clusplot(X,
         y_hc,
         lines = 1,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("clusters of customers"),
         xlab = "annual income",
         ylab = "spending score")
