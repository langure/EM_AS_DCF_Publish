import os
import sqlite3
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT_PATH = '../workdir'
BERT_WORKDIR_PATH = os.path.join(ROOT_PATH, 'bert_workdir')
db_file_path = os.path.join(ROOT_PATH, 'soul_emotions.db')

RANDOM_SAMPLE_SIZE_FOR_CROSS_DATASET_PREDICTION = 1000
EPOCHS = 3


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


def train_emotion_classifier(dataset):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(dataset))

    texts = [item[2] for item in dataset]
    labels = [item[3] for item in dataset]

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    input_ids = []
    attention_masks = []
    for text in tqdm(texts, desc="Tokenizing"):
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels_encoded = torch.tensor(labels_encoded)
    input_ids_train, input_ids_val, attention_masks_train, attention_masks_val, labels_train, labels_val = train_test_split(
        input_ids, attention_masks, labels_encoded, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    val_loader = DataLoader(val_dataset, batch_size=16)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    num_epochs = EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            input_ids_batch, attention_mask_batch, labels_batch = batch
            input_ids_batch = input_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
            loss = criterion(outputs.logits, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Training Loss: {average_loss:.4f}")

    model.eval()
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids_batch, attention_mask_batch, labels_batch = batch
            input_ids_batch = input_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
            _, predicted = torch.max(outputs.logits, 1)
            val_predictions.extend(predicted.cpu().numpy())
            val_targets.extend(labels_batch.cpu().numpy())

    accuracy = accuracy_score(val_targets, val_predictions)
    precision = precision_score(val_targets, val_predictions, average="weighted")
    recall = recall_score(val_targets, val_predictions, average="weighted")
    f1 = f1_score(val_targets, val_predictions, average="weighted")

    emotion_mapping = {label_encoder.transform([emotion])[0]: emotion for emotion in label_encoder.classes_}
    results_dict = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'emotion_mapping': emotion_mapping
    }
    return results_dict


class EmotionClassifier:
    def __init__(self):
        self.models = {}

    def add_model(self, dataset_id, res):
        self.models[dataset_id] = res

    def predict_emotions(self, text):
        predictions = {}
        for dataset_id, model_result in self.models.items():
            model = model_result['model']
            emotion_mapping = model_result['emotion_mapping']
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            inputs = {key: value.to(model.device) for key, value in
                      inputs.items()}  # Move input to the same device as the model

            with torch.no_grad():
                outputs = model(**inputs)
                _, predicted = torch.max(outputs.logits, 1)

            predicted_emotion = emotion_mapping.get(predicted.item(), 'Unknown')
            predictions[f'dataset_{dataset_id}'] = predicted_emotion

        return predictions


def cross_dataset_prediction(em, data, num_samples=RANDOM_SAMPLE_SIZE_FOR_CROSS_DATASET_PREDICTION):
    audit_df = pd.DataFrame(columns=['model_label', 'text', 'source_emotion', 'predicted_emotion'])
    accuracy_results = {}

    df = pd.DataFrame(data, columns=['source_id', 'dataset_id', 'text', 'source_emotion'])
    model_labels = list(em.models.keys())

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
                            new_row = {'model_label': result['model_label'], 'text': row['text'],
                                       'source_emotion': row['source_emotion'],
                                       'predicted_emotion': predicted_emotions.get(f"dataset_{row['dataset_id']}")}
                            new_row_df = pd.DataFrame([new_row])
                            audit_df = pd.concat([audit_df, new_row_df], ignore_index=True)
                            if predicted_emotions.get(f"dataset_{row['dataset_id']}") == row['source_emotion']:
                                result['num_correct'] += 1
                            else:
                                result['num_incorrect'] += 1
                            result['num_samples'] += 1

    for result in accuracy_results.values():
        result['accuracy'] = result['num_correct'] / (result['num_correct'] + result['num_incorrect'])
    return accuracy_results, audit_df


if not os.path.exists(BERT_WORKDIR_PATH):
    os.makedirs(BERT_WORKDIR_PATH)

print("Loading source emotions data from database")
source_emotions_data = load_source_emotions_from_database(db_file_path)
source_emotions_datasets = split_data_by_dataset(source_emotions_data)
dataset_ids = list(source_emotions_datasets.keys())
print("Training emotion classifiers with source emotions")
res_list = []
for dataset_id in dataset_ids:
    print(f"Training model for dataset {dataset_id}")
    dataset = source_emotions_datasets[dataset_id]
    res = train_emotion_classifier(dataset)
    res_list.append(res)
emotions_classifier = EmotionClassifier()
for dataset_id, res in zip(dataset_ids, res_list):
    emotions_classifier.add_model(dataset_id, res)
with open(os.path.join(BERT_WORKDIR_PATH, 'emotions_classifier_source.pkl'), 'wb') as f:
    pickle.dump(emotions_classifier, f)

print("Loading shared emotions data from database")
shared_emotions_data = load_shared_emotions_from_database(db_file_path)
shared_emotions_datasets = split_data_by_dataset(shared_emotions_data)
dataset_ids = list(shared_emotions_datasets.keys())
print("Training emotion classifiers with shared emotions mappings")
res_list = []
for dataset_id in dataset_ids:
    print(f"Training model for dataset {dataset_id}")
    dataset = shared_emotions_datasets[dataset_id]
    res = train_emotion_classifier(dataset)
    res_list.append(res)
emotions_classifier = EmotionClassifier()
for dataset_id, res in zip(dataset_ids, res_list):
    emotions_classifier.add_model(dataset_id, res)
with open(os.path.join(BERT_WORKDIR_PATH, 'emotions_classifier_shared.pkl'), 'wb') as f:
    pickle.dump(emotions_classifier, f)

print("Loading quadrant emotions data from database")
quadrant_emotions_data = load_quadrant_emotions_from_database(db_file_path)
quadrant_emotions_datasets = split_data_by_dataset(quadrant_emotions_data)
dataset_ids = list(quadrant_emotions_datasets.keys())
print("Training emotion classifiers with quadrant emotions mappings")
res_list = []
for dataset_id in dataset_ids:
    print(f"Training model for dataset {dataset_id}")
    dataset = quadrant_emotions_datasets[dataset_id]
    res = train_emotion_classifier(dataset)
    res_list.append(res)
emotions_classifier = EmotionClassifier()
for dataset_id, res in zip(dataset_ids, res_list):
    emotions_classifier.add_model(dataset_id, res)
with open(os.path.join(BERT_WORKDIR_PATH, 'emotions_classifier_quadrant.pkl'), 'wb') as f:
    pickle.dump(emotions_classifier, f)

print("Loading trained emotion classifiers from pickle")
with open(os.path.join(BERT_WORKDIR_PATH, 'emotions_classifier_source.pkl'), 'rb') as f:
    source_emotions_classifier = pickle.load(f)
with open(os.path.join(BERT_WORKDIR_PATH, 'emotions_classifier_shared.pkl'), 'rb') as f:
    shared_emotions_classifier = pickle.load(f)
with open(os.path.join(BERT_WORKDIR_PATH, 'emotions_classifier_quadrant.pkl'), 'rb') as f:
    quadrant_emotions_classifier = pickle.load(f)

source_emotions_data = load_source_emotions_from_database(db_file_path)
shared_emotions_data = load_shared_emotions_from_database(db_file_path)
quadrant_emotions_data = load_quadrant_emotions_from_database(db_file_path)

print("Cross dataset prediction for source labels...")
accuracy_results_s1, audit_df = cross_dataset_prediction(source_emotions_classifier, source_emotions_data, num_samples=RANDOM_SAMPLE_SIZE_FOR_CROSS_DATASET_PREDICTION)
with open(os.path.join(BERT_WORKDIR_PATH, 'accuracy_results_s1.txt'), 'w') as f:
    f.write(str(accuracy_results_s1))
audit_df.to_csv(os.path.join(BERT_WORKDIR_PATH, 'accuracy_results_audit_1.csv'))

print("Cross dataset prediction for shared labels...")
accuracy_results_s2, audit_df = cross_dataset_prediction(shared_emotions_classifier, shared_emotions_data, num_samples=RANDOM_SAMPLE_SIZE_FOR_CROSS_DATASET_PREDICTION)
with open(os.path.join(BERT_WORKDIR_PATH, 'shared_accuracy_results_s2.txt'), 'w') as f:
    f.write(str(accuracy_results_s2))
audit_df.to_csv(os.path.join(BERT_WORKDIR_PATH, 'shared_accuracy_results_audit_2.csv'))

print("Cross dataset prediction for quadrant labels...")
accuracy_results_s3, audit_df = cross_dataset_prediction(quadrant_emotions_classifier, quadrant_emotions_data, num_samples=RANDOM_SAMPLE_SIZE_FOR_CROSS_DATASET_PREDICTION)
with open(os.path.join(BERT_WORKDIR_PATH, 'quadrant_accuracy_results_s2.txt'), 'w') as f:
    f.write(str(accuracy_results_s3))
audit_df.to_csv(os.path.join(BERT_WORKDIR_PATH, 'quadrant_accuracy_results_audit_2.csv'))
