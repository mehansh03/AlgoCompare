import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt # type: ignore
import io
import base64
from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)

# Global variables to store data and results
df = None
results_table = None
model_names = ['Logistic Regression', 'Naive Bayes', 'KNN', 'Decision Tree', 'Random Forest', 'K-means']

def prepare_data(df, text_column, label_column):
    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df[text_column])

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df[label_column])

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_evaluate_classifier(classifier, X_train, X_test, y_train, y_test, name):
    # Train the model
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Calculate metrics
    metrics = {
        'Classifier': name,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted'),
        'Accuracy': accuracy_score(y_test, y_pred)
    }

    return metrics

def run_classification_analysis(df, text_column, label_column):
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, text_column, label_column)

    # Initialize classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
    }

    # Train and evaluate each classifier
    results = []
    for name, classifier in classifiers.items():
        metrics = train_evaluate_classifier(classifier, X_train, X_test, y_train, y_test, name)
        results.append(metrics)

    # Add K-means clustering
    kmeans = KMeans(n_clusters=len(np.unique(y_train)), random_state=42)
    kmeans_pred = kmeans.fit_predict(X_test.toarray())

    # Calculate metrics for K-means
    kmeans_metrics = {
        'Classifier': 'K-means',
        'Precision': precision_score(y_test, kmeans_pred, average='weighted'),
        'Recall': recall_score(y_test, kmeans_pred, average='weighted'),
        'F1-Score': f1_score(y_test, kmeans_pred, average='weighted'),
        'Accuracy': accuracy_score(y_test, kmeans_pred)
    }
    results.append(kmeans_metrics)

    # Create comparison table
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('Classifier')
    comparison_df = comparison_df.round(3)

    return comparison_df

def create_performance_chart(results_df):
    # Create a bar chart of the results
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    
    plt.figure(figsize=(12, 8))
    
    # Set up the bar chart
    bar_width = 0.15
    index = np.arange(len(results_df.index))
    
    for i, metric in enumerate(metrics):
        plt.bar(index + i * bar_width, results_df[metric], bar_width, label=metric)
    
    plt.xlabel('Classification Algorithms')
    plt.ylabel('Score')
    plt.title('Performance Comparison of Classification Algorithms')
    plt.xticks(index + bar_width * (len(metrics) - 1) / 2, results_df.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Convert plot to base64 for embedding in HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    
    try:
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Get column names for dropdown selections
        columns = df.columns.tolist()
        
        return render_template('analyze.html', columns=columns, filename=file.filename)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/analyze', methods=['POST'])
def analyze():
    global df, results_table
    
    try:
        text_column = request.form.get('text_column')
        label_column = request.form.get('label_column')
        
        if not text_column or not label_column:
            return render_template('error.html', error="Please select both text and label columns.")
        
        # Run the classification analysis
        results_table = run_classification_analysis(df, text_column, label_column)
        
        # Create chart for visualization
        plot_url = create_performance_chart(results_table)
        
        # Convert the results to HTML
        results_html = results_table.to_html(classes='table table-striped')
        
        return render_template('results.html', 
                               results=results_html, 
                               plot_url=plot_url,
                               model_names=model_names)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/model_details/<model_name>')
def model_details(model_name):
    global results_table
    
    try:
        if results_table is None:
            return redirect(url_for('index'))
        
        model_metrics = results_table.loc[model_name].to_dict()
        
        return render_template('model_details.html', 
                               model_name=model_name, 
                               metrics=model_metrics)
    
    except Exception as e:
        return render_template('error.html', error=str(e))



if __name__ == '__main__':
    app.run(debug=True)