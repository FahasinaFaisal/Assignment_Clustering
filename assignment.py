import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, silhouette_score

def load_and_preprocess_data(file_path):
    """
    Load and preprocess data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - data (pd.DataFrame): Preprocessed DataFrame.
    """
    data = pd.read_csv(file_path)
    if 'species' in data.columns:
        data.rename(columns={'species': 'class'}, inplace=True)
    data.drop_duplicates(inplace=True)
    return data

def clean_and_transpose_data(data):
    """
    Clean and transpose the input data.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.

    Returns:
    - cleaned_data (pd.DataFrame): Cleaned DataFrame.
    - transposed_data (pd.DataFrame): Transposed DataFrame.
    """
    cleaned_data = data.dropna()
    transposed_data = cleaned_data.transpose()
    return cleaned_data, transposed_data

def visualize_feature_distributions(data):
    """
    Visualize feature distributions.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.
    """
    for feature in data.drop(columns='class'):
        plt.figure(figsize=(6, 4))
        sns.histplot(data=data, x=feature, kde=True)
        plt.xlabel(f'{feature}')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {feature}')
        plt.show()

def analyze_outliers(data, feature):
    """
    Analyze outliers for a specific feature.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.
    - feature (str): Name of the feature to analyze.
    """
    if feature in data.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=data[feature])
        plt.title(f'Box Plot for {feature}')
        plt.show()
    else:
        print(f"Warning: Feature '{feature}' not found in the dataframe.")

def remove_outliers(data, feature):
    """
    Remove outliers for a specific feature.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.
    - feature (str): Name of the feature to remove outliers from.

    Returns:
    - data (pd.DataFrame): DataFrame after removing outliers.
    """
    if feature in data.columns:
        lower, upper = data[feature].quantile([0.02, 0.98]).to_list()
        return data[data[feature].between(lower, upper)]
    else:
        print(f"Warning: Feature '{feature}' not found in the dataframe.")
        return data

def scatter_plot(x, y, hue, title, data, xlabel='', ylabel=''):
    """
    Create a scatter plot.

    Parameters:
    - x (str): Name of the feature for the x-axis.
    - y (str): Name of the feature for the y-axis.
    - hue (str): Name of the feature for coloring.
    - title (str): Title of the plot.
    - data (pd.DataFrame): Input DataFrame.
    - xlabel (str, optional): Label for the x-axis. Defaults to an empty string.
    - ylabel (str, optional): Label for the y-axis. Defaults to an empty string.
    """
    if x in data.columns and y in data.columns:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=data, x=x, y=y, hue=hue)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    else:
        print(f"Warning: One or more specified features not found in the dataframe.")

def visualize_correlation(data):
    """
    Visualize correlation heatmap.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.
    """
    correlation = data.drop(columns='class').corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='YlGnBu')
    plt.title('Correlation Heatmap')
    plt.show()

def visualize_k_means_clusters(X_transformed, labels, centroids):
    """
    Visualize K-Means clusters.

    Parameters:
    - X_transformed (array-like): Transformed data.
    - labels (array-like): Cluster labels.
    - centroids (array-like): Cluster centroids.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_transformed[:, 1], y=X_transformed[:, 2], hue=labels, palette='deep')
    plt.scatter(x=centroids[:, 1], y=centroids[:, 2], color='gray', marker='*', s=150)
    plt.title('K-Means Clustering: Sepal Width vs Sepal Length')
    plt.xlabel('Scaled Sepal Width')
    plt.ylabel('Scaled Sepal Length')
    plt.show()

def visualize_pca_plot(X_pca, labels):
    """
    Visualize PCA plot.

    Parameters:
    - X_pca (array-like): PCA-transformed data.
    - labels (array-like): Labels for coloring.
    """
    plt.figure(figsize=(8, 6))
    color_palette = sns.color_palette('tab10', n_colors=len(set(labels)))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette=color_palette)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("PCA Plot: Clustering by K-Means")
    plt.show()

def build_knn_model():
    """
    Build a KNN model using a pipeline.

    Returns:
    - model (Pipeline): KNN model.
    """
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=3))
    ])

def perform_grid_search(knn, X_train, y_train):
    """
    Perform grid search for hyperparameter tuning.

    Parameters:
    - knn (Pipeline): KNN model.
    - X_train (array-like): Training data.
    - y_train (array-like): Training labels.

    Returns:
    - best_knn (Pipeline): Best KNN model.
    - best_params (dict): Best hyperparameters.
    """
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9],
        'knn__weights': ['uniform', 'distance'],
    }
    grid_search = GridSearchCV(knn, param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)
    best_knn = grid_search.best_estimator_
    return best_knn, best_params

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the KNN model.

    Parameters:
    - model (Pipeline): KNN model.
    - X_train (array-like): Training data.
    - y_train (array-like): Training labels.
    - X_test (array-like): Testing data.
    - y_test (array-like): Testing labels.
    """
    y_train_pred = model.predict(X_train)
    print("Training Accuracy:", accuracy_score(y_train, y_train_pred))

    disp_train = ConfusionMatrixDisplay.from_estimator(model, X_train, y_train, colorbar=False, cmap='viridis')
    plt.title('Confusion Matrix - Training Data')
    plt.show()

    disp_test = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, colorbar=False, cmap='viridis')
    plt.title('Confusion Matrix - Test Data')
    plt.show()

    # Get predictions for the test set
    y_test_pred = model.predict(X_test)

    # Assuming X_test is a NumPy array
    plt.figure(figsize=(8, 6))
    
    # Assuming 'petal_length' and 'petal_width' are the first two columns in X_test
    petal_length_idx = 0
    petal_width_idx = 1
    
    sns.scatterplot(x=X_test[:, petal_length_idx], y=X_test[:, petal_width_idx], hue=y_test_pred, palette='coolwarm')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('KNN Classification for Test Data')
    plt.show()

def visualize_pca_original_classes(X_pca, y):
    """
    Visualize PCA plot for original classes.

    Parameters:
    - X_pca (array-like): PCA-transformed data.
    - y (array-like): Original labels for coloring.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='summer')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Iris data in 2D Using PCA Labeled by Original Classes')
    plt.show()

# Load and preprocess data
file_path = 'IRIS.csv'
data = load_and_preprocess_data(file_path)

# Clean and transpose the data
cleaned_data, transposed_data = clean_and_transpose_data(data)

# Visualize feature distributions
visualize_feature_distributions(cleaned_data)

# Analyze outliers for 'sepal_width'
analyze_outliers(cleaned_data, 'sepal_width')

# Remove outliers for 'sepal_width'
cleaned_data = remove_outliers(cleaned_data, 'sepal_width')

# Scatter Plots
scatter_plot('sepal_length', 'sepal_width', 'class', 'Effect of class and correlation between sepal length and sepal width', cleaned_data, xlabel='Sepal Length', ylabel='Sepal Width')
scatter_plot('petal_length', 'petal_width', 'class', 'Effect of class and correlation between petal width and petal length', cleaned_data, xlabel='Petal Length', ylabel='Petal Width')

# Visualize correlation between features
visualize_correlation(cleaned_data)

# Splitting Data
target = 'class'
X = cleaned_data.drop(columns=target)
y = cleaned_data[target]

# Normalize the data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Model Building - K Means
k_means = Pipeline([
    ('scaler', MinMaxScaler()),
    ('kmeans', KMeans(n_clusters=2, random_state=42))
])

# Model Training
X_normalized = scaler.fit_transform(X)
k_means.fit(X_normalized)
labels = k_means.named_steps['kmeans'].labels_
centroids = k_means.named_steps['kmeans'].cluster_centers_

# Calculate silhouette score
silhouette_avg = silhouette_score(X_normalized, labels)
print(f"Silhouette Score for K-Means Clustering: {silhouette_avg}")

# Visualize K-Means Clustering
visualize_k_means_clusters(X_normalized, labels, centroids)

# Plot 2D scatter plot using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)

# Visualize PCA Plot
visualize_pca_plot(X_pca, labels)

# Building KNN model
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42, stratify=y)
knn = build_knn_model()

# Perform Grid Search for hyperparameter tuning
best_knn, best_params = perform_grid_search(knn, X_train, y_train)

# Evaluate the KNN model
evaluate_model(best_knn, X_train, y_train, X_test, y_test)

# Visualize PCA Plot for Original Classes
visualize_pca_original_classes(X_pca, y)
