import os
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
from tqdm import tqdm

ROOT_PATH = '../workdir'
FNN_WORKDIR_PATH = os.path.join(ROOT_PATH, 'fnn_workdir')
db_file_path = os.path.join(ROOT_PATH, 'soul_emotions.db')

HIDDEN_SIZE = 32
EPOCHS = 3
RANDOM_SAMPLE_SIZE_FOR_CROSS_DATASET_PREDICTION = 1000

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


class SimpleFNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


def train_emotion_classifier(dataset):
    texts = [item[2] for item in dataset]
    labels = [item[3] for item in dataset]

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(texts)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    X_train, X_val, y_train, y_val = train_test_split(X_tfidf, labels_encoded, test_size=0.2, random_state=42)

    input_size = X_tfidf.shape[1]
    hidden_size = HIDDEN_SIZE
    num_classes = len(label_encoder.classes_)

    model = SimpleFNNClassifier(input_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = EPOCHS
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(total=X_train.shape[0])
        for i in range(X_train.shape[0]):
            batch_x = X_train.getrow(i).toarray().flatten()
            batch_y = y_train[i]
            outputs = model(torch.tensor(batch_x, dtype=torch.float32))
            loss = criterion(outputs.unsqueeze(0), torch.tensor([batch_y], dtype=torch.int64))
            loss.backward()
            optimizer.step()
            pbar.update(1)
        pbar.close()

    model.eval()
    with torch.no_grad():
        val_predictions = []
        for i in range(X_val.shape[0]):
            batch_x = X_val.getrow(i).toarray().flatten()
            outputs = model(torch.tensor(batch_x, dtype=torch.float32))
            _, predicted = torch.max(outputs, 0)
            val_predictions.append(predicted.item())

    accuracy = accuracy_score(y_val, val_predictions)
    precision = precision_score(y_val, val_predictions, average="weighted")
    recall = recall_score(y_val, val_predictions, average="weighted")
    f1 = f1_score(y_val, val_predictions, average="weighted")

    results_dict = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tfidf_vectorizer': tfidf_vectorizer,
        'label_encoder': label_encoder,
        'emotion_mapping': dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
    }
    return results_dict


class EmotionClassifier:
    def __init__(self):
        self.datasets = {}

    def add_model(self, dataset_id, results_dict):
        self.datasets[dataset_id] = results_dict

    def predict_emotions(self, text):
        predictions = {}
        for dataset_id, data in self.datasets.items():
            emotion_mapping = data['emotion_mapping']
            tfidf_vectorizer = data['tfidf_vectorizer']
            model = data['model']
            tfidf_vector = tfidf_vectorizer.transform([text])
            batch_x = tfidf_vector.toarray().flatten()
            outputs = model(torch.tensor(batch_x, dtype=torch.float32))
            predicted_label_index = torch.argmax(outputs).item()
            predicted_emotion = emotion_mapping[predicted_label_index]
            predictions[dataset_id] = predicted_emotion

        return predictions


def cross_dataset_prediction(em, data, num_samples=RANDOM_SAMPLE_SIZE_FOR_CROSS_DATASET_PREDICTION):
    audit_df = pd.DataFrame(columns=['model_label', 'text', 'source_emotion', 'predicted_emotion'])
    accuracy_results = {}
    df = pd.DataFrame(data, columns=['source_id', 'dataset_id', 'text', 'source_emotion'])
    model_labels = list(em.datasets.keys())

    for model_label in model_labels:
        result = {
            'model_label': model_label,
            'accuracy': 0,
            'num_samples': 0,
            'num_correct': 0,
            'num_incorrect': 0
        }
        accuracy_results[model_label] = result

    keep_sampling = True
    sampled_data = []
    round = 1

    while keep_sampling:
        print(f"Round {round}")
        print(f"Number of samples: {num_samples}")

        for result in accuracy_results.values():
            print(f"{result['model_label']}: {result['num_samples']}/{num_samples}")

        round += 1
        for result in accuracy_results.values():
            if result['num_samples'] < num_samples:
                keep_sampling = True
                break
            else:
                keep_sampling = False
        if keep_sampling:
            df_sample = df.sample(n=num_samples)
            for index, row in df_sample.iterrows():
                unique_id = f"{row['dataset_id']}_{row['source_id']}"
                if unique_id not in sampled_data:
                    predicted_emotions = em.predict_emotions(row['text'])
                    sampled_data.append(unique_id)
                    for result in accuracy_results.values():
                        if result['model_label'] != row['dataset_id'] and result['num_samples'] < num_samples:
                            new_row = {'model_label': result['model_label'], 'text': row['text'], 'source_emotion': row['source_emotion'], 'predicted_emotion': predicted_emotions.get(f"{row['dataset_id']}")}
                            new_row_df = pd.DataFrame([new_row])
                            audit_df = pd.concat([audit_df, new_row_df], ignore_index=True)
                            if predicted_emotions.get(f"{row['dataset_id']}") == row['source_emotion']:
                                result['num_correct'] += 1
                            else:
                                result['num_incorrect'] += 1
                            result['num_samples'] += 1

    for result in accuracy_results.values():
        result['accuracy'] = result['num_correct'] / (result['num_correct'] + result['num_incorrect'])

    return accuracy_results, audit_df


if not os.path.exists(FNN_WORKDIR_PATH):
    os.makedirs(FNN_WORKDIR_PATH)

print("Train emotion classifier with the source emotions")
source_emotions_data = load_source_emotions_from_database(db_file_path)
source_emotions_datasets = split_data_by_dataset(source_emotions_data)
dataset_ids = list(source_emotions_datasets.keys())
res_list = []
for dataset_id in dataset_ids:
    print(f"Training model for dataset {dataset_id}")
    dataset = source_emotions_datasets[dataset_id]
    res_list.append(train_emotion_classifier(dataset))
emotions_classifier = EmotionClassifier()
for dataset_id, res in zip(dataset_ids, res_list):
    emotions_classifier.add_model(dataset_id, res)
with open(os.path.join(FNN_WORKDIR_PATH, 'source_emotion_classifier.pkl'), 'wb') as f:
    pickle.dump(emotions_classifier, f)

print("Train emotion classifier with the shared emotions")
shared_emotions_data = load_shared_emotions_from_database(db_file_path)
shared_emotions_datasets = split_data_by_dataset(shared_emotions_data)
dataset_ids = list(shared_emotions_datasets.keys())
res_list = []
for dataset_id in dataset_ids:
    print(f"Training model for dataset {dataset_id}")
    dataset = shared_emotions_datasets[dataset_id]
    res_list.append(train_emotion_classifier(dataset))
emotions_classifier = EmotionClassifier()
for dataset_id, res in zip(dataset_ids, res_list):
    emotions_classifier.add_model(dataset_id, res)
with open(os.path.join(FNN_WORKDIR_PATH, 'shared_emotion_classifier.pkl'), 'wb') as f:
    pickle.dump(emotions_classifier, f)

print("Train emotion classifier with the quadrant emotions")
quadrant_emotions_data = load_quadrant_emotions_from_database(db_file_path)
quadrant_emotions_datasets = split_data_by_dataset(quadrant_emotions_data)
dataset_ids = list(quadrant_emotions_datasets.keys())
res_list = []
for dataset_id in dataset_ids:
    print(f"Training model for dataset {dataset_id}")
    dataset = quadrant_emotions_datasets[dataset_id]
    res_list.append(train_emotion_classifier(dataset))
emotions_classifier = EmotionClassifier()
for dataset_id, res in zip(dataset_ids, res_list):
    emotions_classifier.add_model(dataset_id, res)
with open(os.path.join(FNN_WORKDIR_PATH, 'quadrant_emotion_classifier.pkl'), 'wb') as f:
    pickle.dump(emotions_classifier, f)

print("Loading the emotion classifiers from pickle")
with open(os.path.join(FNN_WORKDIR_PATH, 'source_emotion_classifier.pkl'), 'rb') as f:
    emotions_classifier_source = pickle.load(f)
with open(os.path.join(FNN_WORKDIR_PATH, 'shared_emotion_classifier.pkl'), 'rb') as f:
    emotions_classifier_shared = pickle.load(f)
with open(os.path.join(FNN_WORKDIR_PATH, 'quadrant_emotion_classifier.pkl'), 'rb') as f:
    emotions_classifier_quadrant = pickle.load(f)

round = "_w0"

print("Cross dataset predictions with source emotions. Saving actual predictions in an audit file")
accuracy_results_s1, audit_df = cross_dataset_prediction(emotions_classifier_source, source_emotions_data, num_samples=RANDOM_SAMPLE_SIZE_FOR_CROSS_DATASET_PREDICTION)
with open(os.path.join(FNN_WORKDIR_PATH, f'accuracy_results_s1{round}.txt'), 'w') as f:
    f.write(str(accuracy_results_s1))
audit_df.to_csv(os.path.join(FNN_WORKDIR_PATH, f'accuracy_results_audit_1{round}.csv'))
print(accuracy_results_s1)

print("Cross dataset predictions with shared emotions. Saving actual predictions in an audit file")
accuracy_results_s2, audit_df = cross_dataset_prediction(emotions_classifier_shared, shared_emotions_data, num_samples=RANDOM_SAMPLE_SIZE_FOR_CROSS_DATASET_PREDICTION)
with open(os.path.join(FNN_WORKDIR_PATH, f'shared_accuracy_results_s2{round}.txt'), 'w') as f:
    f.write(str(accuracy_results_s2))
audit_df.to_csv(os.path.join(FNN_WORKDIR_PATH, f'shared_accuracy_results_audit_2{round}.csv'))
print(accuracy_results_s2)

print("Cross dataset predictions with quadrant emotions. Saving actual predictions in an audit file")
accuracy_results_s3, audit_df = cross_dataset_prediction(emotions_classifier_quadrant, quadrant_emotions_data, num_samples=RANDOM_SAMPLE_SIZE_FOR_CROSS_DATASET_PREDICTION)
with open(os.path.join(FNN_WORKDIR_PATH, f'quadrant_accuracy_results_s2{round}.txt'), 'w') as f:
    f.write(str(accuracy_results_s3))
audit_df.to_csv(os.path.join(FNN_WORKDIR_PATH, f'quadrant_accuracy_results_audit_2{round}.csv'))
print(accuracy_results_s3)