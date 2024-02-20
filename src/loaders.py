import abc
import os
import pandas as pd
import re
from database import DBDriver
from tqdm import tqdm
import json
import xml.etree.ElementTree as ET
import csv


class DBRow:
    def __init__(self, source_id, dataset_id, text, source_emotion, shared_emotion, quadrant_emotion, valence,
                 arousal):
        self.source_id = source_id
        self.dataset_id = dataset_id
        self.text = self.clean_text(text)
        self.source_emotion = source_emotion
        self.shared_emotion = shared_emotion
        self.quadrant_emotion = quadrant_emotion
        self.valence = valence
        self.arousal = arousal

    @staticmethod
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)
        cleaned_text = re.sub(r'[^\w\s@#😀-🙏]', '', text)
        return cleaned_text


class AbstractDataLoader(metaclass=abc.ABCMeta):
    def __init__(self, file_path, db_driver: DBDriver):
        self.file_path = file_path
        self.db_driver = db_driver
        self.dataset_id = self.__class__.__name__
        self.verify_file_exists()

    def verify_file_exists(self):
        file_paths = self.file_path.split('|')
        for path in file_paths:
            if not os.path.exists(path.strip()):
                raise FileNotFoundError(f"The file {path} does not exist.")

    @abc.abstractmethod
    def load_file(self):
        pass

    @staticmethod
    def parse(df: pd.DataFrame):
        db_rows = []
        for index, row in df.iterrows():
            source_id = row['source_id']
            text = row['content']
            source_emotion = row['sentiment']
            dataset_id = None
            shared_emotion = None
            quadrant_emotion = None
            valence = None
            arousal = None
            db_row = DBRow(source_id, dataset_id, text, source_emotion, shared_emotion, quadrant_emotion,
                           valence, arousal)
            db_rows.append(db_row)

        return db_rows

    def persist(self, rows):
        if not self.db_driver:
            raise ValueError("DB driver is not set. Cannot persist data to the database.")

        existing_rows = self.db_driver.get_rows_by_dataset_id(self.dataset_id)
        existing_source_ids = set(row[1] for row in existing_rows)
        rows_to_persist = [row for row in rows if row.source_id not in existing_source_ids]
        total_rows = len(rows_to_persist)

        progress_bar = tqdm(total=total_rows, desc=f"Persisting {self.dataset_id}", unit=" row")

        for row in rows_to_persist:
            row.dataset_id = self.dataset_id
            self.db_driver.write(row)
            progress_bar.update(1)
        progress_bar.close()

        return f"Persisted {total_rows} rows"


class DataLoaderFactory:
    _data_loader_registry = {}

    @classmethod
    def register_data_loader(cls, data_loader_cls):
        cls._data_loader_registry[data_loader_cls.__name__] = data_loader_cls
        return data_loader_cls

    @classmethod
    def create_data_loader(cls, loader_name, file_path, db_driver: DBDriver):
        data_loader_cls = cls._data_loader_registry.get(loader_name)
        if data_loader_cls:
            return data_loader_cls(file_path, db_driver)  # Pass db_driver to the data loader constructor
        else:
            raise ValueError(f"Data loader '{loader_name}' not found")


@DataLoaderFactory.register_data_loader
class DatasetOneLoader(AbstractDataLoader):
    def load_file(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        df = pd.read_csv(self.file_path)
        expected_columns = ['tweet_id', 'sentiment', 'content']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The CSV file is missing columns: {', '.join(missing_columns)}")
        df.rename(columns={'tweet_id': 'source_id'}, inplace=True)
        return df


@DataLoaderFactory.register_data_loader
class DatasetTwoLoader(AbstractDataLoader):
    def load_file(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        df = pd.read_csv(self.file_path)
        expected_columns = ['content', 'sentiment']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The CSV file is missing columns: {', '.join(missing_columns)}")
        df.insert(0, 'source_id', range(1, 1 + len(df)))
        return df


@DataLoaderFactory.register_data_loader
class DatasetThreeLoader(AbstractDataLoader):
    def load_file(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        with open(self.file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        label_mapping = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }
        df = pd.DataFrame(data)
        df.rename(columns={'text': 'content'}, inplace=True)
        df['sentiment'] = df['label'].map(label_mapping)
        df.drop(columns=['label'], inplace=True)
        expected_columns = ['content', 'sentiment']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The CSV file is missing columns: {', '.join(missing_columns)}")
        df.insert(0, 'source_id', range(1, 1 + len(df)))
        return df


@DataLoaderFactory.register_data_loader
class DatasetFourLoader(AbstractDataLoader):
    def load_file(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        df = pd.read_csv(self.file_path, sep='|', usecols=range(43))
        expected_columns = ['ID', 'Field1', 'SIT']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The CSV file is missing columns: {', '.join(missing_columns)}")
        df = df.rename(columns={'ID': 'source_id', 'Field1': 'sentiment', 'SIT': 'content'})
        return df


@DataLoaderFactory.register_data_loader
class DatasetFiveLoader(AbstractDataLoader):
    def load_file(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        df = pd.read_csv(self.file_path)
        df = df[df['example_very_unclear'] != 'TRUE']
        expected_columns = ["text",
                            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
                            'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
                            'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness',
                            'surprise', 'neutral'
                            ]

        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The CSV file is missing columns: {', '.join(missing_columns)}")
        df = df.rename(columns={"text": "content"})
        sentiment_columns = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
            'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
            'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness',
            'surprise', 'neutral'
        ]
        df['sentiment'] = df[sentiment_columns].idxmax(axis=1)
        df = df[['content', 'sentiment']]
        df.insert(0, 'source_id', range(1, 1 + len(df)))
        return df


@DataLoaderFactory.register_data_loader
class DatasetSixLoader(AbstractDataLoader):
    def __init__(self, file_path, db_driver: DBDriver):
        super().__init__(file_path, db_driver)

    def load_file(self):
        file_paths = self.file_path.split('|')
        if len(file_paths) != 2:
            raise ValueError("Expected two file paths separated by '|', but received an invalid format.")
        xml_file_path = file_paths[0]
        xml_dataframe = pd.DataFrame()
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            data = []
            for instance in root.findall(".//instance"):
                source_id = instance.get("id")
                content = instance.text
                data.append({"source_id": source_id, "content": content})
            xml_dataframe = pd.DataFrame(data)
        except ET.ParseError as e:
            raise ValueError(f"Error parsing XML file 1: {str(e)}")

        txt_file_path = file_paths[1]
        txt_dataframe = pd.DataFrame()
        try:
            with open(txt_file_path, 'r') as txt_file:
                data = []
                for line in txt_file:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    source_id = parts[0]
                    emotions = list(map(int, parts[1:]))
                    sentiment = ["anger", "disgust", "fear", "joy", "sadness", "surprise"][
                        emotions.index(max(emotions))]
                    data.append({"source_id": source_id, "sentiment": sentiment})
                txt_dataframe = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Error loading and processing TXT file 2: {str(e)}")
        combined_df = pd.merge(xml_dataframe, txt_dataframe, on="source_id", how="inner")
        return combined_df


@DataLoaderFactory.register_data_loader
class DatasetSevenLoader(AbstractDataLoader):
    def load_file(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        data = []
        source_id_counter = 1

        with open(self.file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) != 2:
                    raise ValueError("CSV file should have exactly two columns: text and emotion")
                content, emotion = row[0].strip(), row[1].strip()
                data.append({'source_id': source_id_counter, 'content': content, 'sentiment': emotion})
                source_id_counter += 1
        df = pd.DataFrame(data)
        expected_columns = ['source_id', 'content', 'sentiment']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The DataFrame is missing columns: {', '.join(missing_columns)}")
        return df


@DataLoaderFactory.register_data_loader
class DatasetEightLoader(AbstractDataLoader):
    def load_file(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip().split('\t')
                if len(line) >= 3:
                    source_id = int(line[0])
                    content = line[1]
                    sentiment = line[2]
                    data.append({'source_id': source_id, 'sentiment': sentiment, 'content': content})
        df = pd.DataFrame(data)
        expected_columns = ['source_id', 'sentiment', 'content']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The DataFrame is missing columns: {', '.join(missing_columns)}")
        return df
