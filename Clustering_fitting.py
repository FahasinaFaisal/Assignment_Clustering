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

def load_and_preprocess_data(file_path):
    """
    Load the data from a CSV file, rename the 'species' column to 'class', and remove duplicates.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    data = pd.read_csv(file_path)
    data.rename(columns={'species': 'class'}, inplace=True)
    data.drop_duplicates(inplace=True)
    return data

def visualize_feature_distributions(data):
    """
    Visualize the distributions of features using histograms.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data.
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
    Analyze and visualize outliers for a specific feature using boxplots.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data.
    - feature (str): Feature for outlier analysis.
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data[feature])
    plt.title(f'Box Plot for {feature}')
    plt.show()

    # Remove outliers
    lower, upper = data[feature].quantile([0.02, 0.98]).to_list()
    data = data[data[feature].between(lower, upper)]

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data[feature])
    plt.title(f'Box Plot for {feature} (Outlier Removed)')
    plt.show()

def scatter_plot(x, y, hue, title, data):
    """
    Create a scatter plot.

    Parameters:
    - x, y, hue, title (str): Plotting parameters.
    - data (pd.DataFrame): DataFrame containing the data.
    """
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=data, x=x, y=y, hue=hue)
    plt.title(title)
    plt.show()

def visualize_correlation(data):
    """
    Visualize the correlation between features using a heatmap.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data.
    """
    correlation = data.drop(columns='class').corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='YlGnBu')
    plt.title('Correlation Heatmap')
    plt.show()

def create_k_means_pipeline(n_clusters=3):
    """
    Create a K-Means clustering pipeline.

    Parameters:
    - n_clusters (int): Number of clusters for K-Means.

    Returns:
    - Pipeline: K-Means clustering pipeline.
    """
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42))
    ])

def visualize_k_means_clusters(X_transformed, labels, centroids):
    """
    Visualize K-Means clustering results using a scatter plot.

    Parameters:
    - X_transformed (np.ndarray): Transformed data using MinMaxScaler.
    - labels (np.ndarray): Cluster labels.
    - centroids (np.ndarray): Cluster centroids.
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
    Visualize a 2D scatter plot using PCA.

    Parameters:
    - X_pca (np.ndarray): Transformed data using PCA.
    - labels (np.ndarray): Cluster labels.
    """
    plt.figure(figsize=(8, 6))
    color_palette = sns.color_palette('tab10', n_colors=len(set(labels)))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette=color_palette)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("PCA Plot: Clustering by K-Means")
    plt.show()

def build_knn_model(X_train, y_train):
    """
    Build a KNN model using a pipeline.

    Parameters:
    - X_train, y_train (pd.DataFrame, np.ndarray): Training data and labels.

    Returns:
    - Pipeline: KNN model pipeline.
    """
    knn = Pipeline([
        ('scaler', MinMaxScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=3))
    ])
    return knn

def perform_grid_search(knn, X_train, y_train):
    """
    Perform grid search for hyperparameter tuning.

    Parameters:
    - knn (Pipeline): KNN model pipeline.
    - X_train, y_train (pd.DataFrame, np.ndarray): Training data and labels.

    Returns:
    - Pipeline: Best KNN model from grid search.
    - dict: Best hyperparameters.
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
    Evaluate the model and display performance metrics.

    Parameters:
    - model (Pipeline): Trained model.
    - X_train, y_train (pd.DataFrame, np.ndarray): Training data and labels.
    - X_test, y_test (pd.DataFrame, np.ndarray): Test data and labels.
    """
    # Training accuracy
    y_train_pred = model.predict(X_train)
    print("Training Accuracy:", accuracy_score(y_train, y_train_pred))

    # Confusion Matrix Display for Training Data
    disp_train = ConfusionMatrixDisplay.from_estimator(model, X_train, y_train, colorbar=False, cmap='viridis')
    plt.title('Confusion Matrix - Training Data')
    plt.show()

    # Confusion Matrix Display for Test Data
    disp_test = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, colorbar=False, cmap='viridis')
    plt.title('Confusion Matrix - Test Data')
    plt.show()

    # Test accuracy and classification report
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred))

    # Scatter plot for KNN Classification on Test Data
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_test['petal_length'], y=X_test['petal_width'], hue=y_test_pred, palette='coolwarm')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('KNN Classification for Test Data')
    plt.show()

def visualize_pca_original_classes(X_pca, y):
    """
    Visualize a 2D scatter plot using PCA with original class labels.

    Parameters:
    - X_pca (np.ndarray): Transformed data using PCA.
    - y (np.ndarray): Original class labels.
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

# Visualize feature distributions
visualize_feature_distributions(data)

# Analyze outliers for 'sepal_width'
analyze_outliers(data, 'sepal_width')

# Scatter Plots
scatter_plot('sepal_length', 'sepal_width', 'class', 'Effect of class and correlation between sepal length and sepal width', data)
scatter_plot('petal_length', 'petal_width', 'class', 'Effect of class and correlation between petal width and petal length', data)

# Visualize correlation between features
visualize_correlation(data)

# Splitting Data
target = 'class'
X = data.drop(columns=target)
y = data[target]

# Model Building - K Means
k_means = create_k_means_pipeline(n_clusters=3)

# Model Training - K Means
k_means.fit(X)
labels = k_means.named_steps['kmeans'].labels_
centroids = k_means.named_steps['kmeans'].cluster_centers_

# Transform X using the scaler
X_transformed = k_means.named_steps['scaler'].fit_transform(X)

# Visualize K-Means Clustering
visualize_k_means_clusters(X_transformed, labels, centroids)

# Plot 2D scatter plot using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize PCA Plot
visualize_pca_plot(X_pca, labels)

# Building KNN model
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = build_knn_model(X_train, y_train)

# Perform Grid Search for hyperparameter tuning
best_knn, best_params = perform_grid_search(knn, X_train, y_train)

# Evaluate the KNN model
evaluate_model(best_knn, X_train, y_train, X_test, y_test)

# Visualize PCA Plot for Original Classes
visualize_pca_original_classes(X_pca, y)
