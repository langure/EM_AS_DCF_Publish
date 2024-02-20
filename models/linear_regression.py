import os
import pickle
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

MAX_LOGISTIC_ITERATIONS = 10

ROOT_PATH = '../workdir'
LINEAR_WORKDIR_PATH = os.path.join(ROOT_PATH, 'linear_workdir')
db_file_path = os.path.join(ROOT_PATH, 'soul_emotions.db')

if os.path.exists(db_file_path):
    print("Database file found.")
else:
    print("Database file not found.")
    exit()

print(db_file_path)


def load_source_emotions_from_database(local_db_file_path):
    conn = sqlite3.connect(local_db_file_path)
    cursor = conn.cursor()
    cursor.execute('SELECT source_id, dataset_id, text, source_emotion FROM soul_emotions')
    rows = cursor.fetchall()
    conn.close()
    return rows


def load_shared_emotions_from_database(local_db_file_path):
    conn = sqlite3.connect(local_db_file_path)
    cursor = conn.cursor()
    cursor.execute('SELECT source_id, dataset_id, text, shared_emotion as source_emotion FROM soul_emotions')
    rows = cursor.fetchall()
    conn.close()
    return rows


def load_quadrant_emotions_from_database(local_db_file_path):
    conn = sqlite3.connect(local_db_file_path)
    cursor = conn.cursor()
    cursor.execute('SELECT source_id, dataset_id, text, quadrant_emotion as source_emotion FROM soul_emotions')
    rows = cursor.fetchall()
    conn.close()
    return rows


def split_data_by_dataset(data):
    datasets = {}
    for row in data:
        dataset_id = row[1]
        if dataset_id not in datasets:
            datasets[dataset_id] = []
        datasets[dataset_id].append(row)
    return datasets


def train_logistic_regression_model(dataset_id, dataset, max_iter=100):
    print(f"About to train dataset {dataset_id} with {max_iter} iterations")
    texts = [row[2] for row in dataset]
    emotions = [row[3] for row in dataset]
    emotion_labels = list(set(emotions))

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, emotions, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=max_iter)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    report = classification_report(y_test, y_pred, target_names=emotion_labels, output_dict=True)
    results = {
        'model': model,
        'report': report,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'vectorizer': vectorizer
    }
    return results


class EmotionClassifier:
    def __init__(self, models):
        self.models = models

    def predict_emotions(self, text):
        predictions = {}

        for dataset_id, model_info in self.models.items():
            model = model_info.get('model')
            vectorizer = model_info.get('vectorizer')

            if model and vectorizer:
                vectorized_text = vectorizer.transform([text])
                emotion = model.predict(vectorized_text)[0]
                predictions[f"{dataset_id}"] = emotion
            else:
                predictions[f"{dataset_id}"] = "No model found for dataset_id"

        return predictions


def cross_dataset_prediction(em, data, real_random=False):
    df = pd.DataFrame(data, columns=['source_id', 'dataset_id', 'text', 'source_emotion'])
    accuracy_results = {}

    for dataset_id, model_info in em.models.items():
        filtered_df = df[df['dataset_id'] != dataset_id]
        if real_random:
            random_state = None
        else:
            random_state = 42

        sampled_df = filtered_df.sample(n=1000, random_state=random_state)
        correct_predictions = 0
        total_predictions = 0

        for index, row in sampled_df.iterrows():
            text = row['text']
            actual_emotion = row['source_emotion']
            predicted_emotion = em.predict_emotions(text).get(dataset_id)
            if predicted_emotion == actual_emotion:
                correct_predictions += 1
            total_predictions += 1
        accuracy = (correct_predictions / total_predictions) * 100
        accuracy_results[dataset_id] = accuracy
    return accuracy_results


def train_all_datasets(datasets, max_iter=MAX_LOGISTIC_ITERATIONS):
    trained_models = {}
    results_data = []

    for dataset_id, dataset in datasets.items():
        results = train_logistic_regression_model(dataset_id, dataset, max_iter)
        trained_models[dataset_id] = results
        print(f"Dataset {dataset_id} accuracy: {results['accuracy']}")
        results_data.append({
            'Dataset ID': dataset_id,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1 Score': results['f1_score']
        })
    return results_data, trained_models


print("Loading data from database...")
source_emotions_data = load_source_emotions_from_database(db_file_path)
shared_emotions_data = load_shared_emotions_from_database(db_file_path)
quadrant_emotions_data = load_quadrant_emotions_from_database(db_file_path)

print("Splitting data by dataset...")
source_emotions_datasets = split_data_by_dataset(source_emotions_data)
shared_emotions_datasets = split_data_by_dataset(shared_emotions_data)
quadrant_emotions_datasets = split_data_by_dataset(quadrant_emotions_data)

if not os.path.exists(LINEAR_WORKDIR_PATH):
    os.makedirs(LINEAR_WORKDIR_PATH)

print("Training models for source labels...")
results_data, trained_models = train_all_datasets(source_emotions_datasets, MAX_LOGISTIC_ITERATIONS)
results_df = pd.DataFrame(results_data)
results_df.to_csv(os.path.join(LINEAR_WORKDIR_PATH, 'source_classification_results.csv'), index=False)
source_emotion_classifier = EmotionClassifier(trained_models)
with open(os.path.join(LINEAR_WORKDIR_PATH, 'source_emotion_classifier.pkl'), 'wb') as file:
    pickle.dump(source_emotion_classifier, file)

print("Training models for shared mapping...")
results_data, trained_models = train_all_datasets(shared_emotions_datasets, MAX_LOGISTIC_ITERATIONS)
results_df = pd.DataFrame(results_data)
results_df.to_csv(os.path.join(LINEAR_WORKDIR_PATH, 'shared_classification_results.csv'), index=False)
shared_emotion_classifier = EmotionClassifier(trained_models)
with open(os.path.join(LINEAR_WORKDIR_PATH, 'shared_emotion_classifier.pkl'), 'wb') as file:
    pickle.dump(shared_emotion_classifier, file)

print("Training models for quadrant mapping...")
results_data, trained_models = train_all_datasets(quadrant_emotions_datasets, MAX_LOGISTIC_ITERATIONS)
results_df = pd.DataFrame(results_data)
results_df.to_csv(os.path.join(LINEAR_WORKDIR_PATH, 'quadrant_classification_results.csv'), index=False)
quadrant_emotion_classifier = EmotionClassifier(trained_models)
with open(os.path.join(LINEAR_WORKDIR_PATH, 'quadrant_emotion_classifier.pkl'), 'wb') as file:
    pickle.dump(quadrant_emotion_classifier, file)

print("Cross dataset prediction for source labels...")
accuracy_results = cross_dataset_prediction(source_emotion_classifier, source_emotions_data, real_random=True)
for dataset_id, accuracy in accuracy_results.items():
    print(f'Dataset {dataset_id} Accuracy: {accuracy}%')

print("Cross dataset prediction for shared mapping...")
accuracy_results = cross_dataset_prediction(shared_emotion_classifier, shared_emotions_data, real_random=True)
for dataset_id, accuracy in accuracy_results.items():
    print(f'Dataset {dataset_id} Accuracy: {accuracy}%')

print("Cross dataset prediction for quadrant mapping...")
accuracy_results = cross_dataset_prediction(quadrant_emotion_classifier, quadrant_emotions_data, real_random=True)
for dataset_id, accuracy in accuracy_results.items():
    print(f'Dataset {dataset_id} Accuracy: {accuracy}%')
