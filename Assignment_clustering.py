import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report

# Load the data
data = pd.read_csv('IRIS.csv')
data.rename(columns={'species': 'class'}, inplace=True)

# Remove duplicates
data.drop_duplicates(inplace=True)

# Distributions of Features
for feature in data.drop(columns='class'):
    plt.figure(figsize=(6, 4))
    sns.histplot(data=data, x=feature, kde=True)
    plt.xlabel(f'{feature}')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature}')
    plt.show()

# Outlier Analysis
plt.figure(figsize=(6, 4))
sns.boxplot(x=data['sepal_width'])
plt.title('Box Plot for Sepal Width')
plt.show()

# Remove outliers for 'sepal_width'
lower, upper = data['sepal_width'].quantile([0.02, 0.98]).to_list()
data = data[data['sepal_width'].between(lower, upper)]
plt.figure(figsize=(6, 4))
sns.boxplot(x=data['sepal_width'])
plt.title('Box Plot for Sepal Width (Outlier Removed)')
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 5))
sns.scatterplot(data=data, x='petal_length',y='petal_width', hue='class')
plt.title('Effect of class and corelation between petal width and petal length');
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# Correlation between features
correlation = data.drop(columns='class').corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='YlGnBu')
plt.title('Correlation Heatmap')
plt.show()

# Splitting Data
target = 'class'
X = data.drop(columns=target)
y = data[target]

# Model Building - K Means
k_means = Pipeline([
    ('scaler', MinMaxScaler()),
    ('kmeans', KMeans(n_clusters=3, random_state=42))
])

# Model Training
k_means.fit(X)
labels = k_means.named_steps['kmeans'].labels_
centroids = k_means.named_steps['kmeans'].cluster_centers_

# Transform X using the scaler
X_transformed = k_means.named_steps['scaler'].fit_transform(X)

# Scatter plot for K-Means Clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_transformed[:, 1], y=X_transformed[:, 2], hue=labels, palette='deep')
plt.scatter(x=centroids[:, 1], y=centroids[:, 2], color='gray', marker='*', s=150)
plt.title('K-Means Clustering: Sepal Width vs Sepal Length')
plt.xlabel('Scaled Sepal Width')
plt.ylabel('Scaled Sepal Length')
plt.show()

# Plot 2D scatter plot using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
color_palette = sns.color_palette('tab10', n_colors=len(set(labels)))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette=color_palette)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title("PCA Plot: Clustering by K-Means")
plt.show()

# Building KNN model
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])

# Define the parameter grid with the hyperparameters and their values to search
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
}

# Create the GridSearchCV object with the pipeline and parameter grid
grid_search = GridSearchCV(knn, param_grid, scoring='accuracy', cv=5)

# Fit the GridSearchCV object on the data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found by grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model found by grid search
best_knn = grid_search.best_estimator_

# Best model choosing
best_knn.fit(X_train, y_train)

# Model evaluation
y_train_pred = best_knn.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))

disp = ConfusionMatrixDisplay.from_estimator(best_knn, X_train, y_train, colorbar=False, cmap='viridis')
disp.plot()
plt.title('Confusion Matrix - Training Data')
plt.show()

# Make prediction
y_test_pred = best_knn.predict(X_test)

disp = ConfusionMatrixDisplay.from_estimator(best_knn, X_test, y_test, colorbar=False, cmap='viridis')
disp.plot()
plt.title('Confusion Matrix - Test Data')
plt.show()

print(classification_report(y_test, y_test_pred))

# Create a scatter plot of the data points colored by labels for testing data
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test['petal_length'], y=X_test['petal_width'], hue=y_test_pred, palette='coolwarm')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('KNN Classification for Test Data')
plt.show()

# PCA Plot for Original Classes
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='summer')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Iris data in 2D Using PCA Labeled by Original Classes')
plt.show()
