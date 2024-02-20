import time
import os
import traceback
from database import DBDriver
from loaders import DataLoaderFactory
from emotion_model_transformers import EmotionModelTransformer
from emotion_mappings import EMOTION_MAPPING_11, EMOTION_VALENCE_AROUSAL

LOAD_DATA_SOURCES = True
WORK_DIR = '../workdir'
DATA_DIR = '../data'


def transform_data():
    db_driver = DBDriver(os.path.join(WORK_DIR, 'soul_emotions.db'))
    transformer = EmotionModelTransformer(db_driver)
    transformer.show_statistics()
    transformer.reduce_emotions(EMOTION_MAPPING_11)
    transformer.transform_to_russell(EMOTION_VALENCE_AROUSAL)
    transformer.reduce_to_quadrant_emotions()
    db_driver.close()


def load_datasources(datasources):
    db_driver = DBDriver(os.path.join(WORK_DIR, 'soul_emotions.db'))
    db_driver.create_tables()

    total_start_time = time.time()

    for datasource in datasources:
        print(f"Loading datasource: {datasource['name']}")

        try:
            loader = DataLoaderFactory.create_data_loader(datasource['name'], datasource['file_path'], db_driver)
            df = loader.load_file()
            db_rows = loader.parse(df)
            print(loader.persist(db_rows))
        except Exception as e:
            with open('errors.log', 'a') as error_file:
                error_file.write(f"Error loading {datasource['name']}: {str(e)}\n")
            traceback.print_exc()

    total_end_time = time.time()
    elapsed_time = total_end_time - total_start_time

    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total time taken: {minutes} minutes and {seconds} seconds")

    db_driver.close()


if __name__ == '__main__':

    dataset_6_instances_path = os.path.join(DATA_DIR, 'dataset_6', 'instances.xml')
    dataset_6_composed_path = os.path.join(DATA_DIR, 'dataset_6', 'encoded_emotions.txt')

    data = [
        {'name': 'DatasetOneLoader', 'file_path': os.path.join(DATA_DIR, 'dataset_1', 'tweet_emotions.csv')},
        {'name': 'DatasetTwoLoader', 'file_path': os.path.join(DATA_DIR, 'dataset_2', 'ALL_DATA.csv')},
        {'name': 'DatasetThreeLoader', 'file_path': os.path.join(DATA_DIR, 'dataset_3', 'data.jsonl')},
        {'name': 'DatasetFourLoader', 'file_path': os.path.join(DATA_DIR, 'dataset_4', 'isear.csv')},
        {'name': 'DatasetFiveLoader', 'file_path': os.path.join(DATA_DIR, 'dataset_5', 'goemotions_1.csv')},
        {'name': 'DatasetSixLoader', 'file_path': f"{dataset_6_instances_path}|{dataset_6_composed_path}"},
        {'name': 'DatasetSevenLoader', 'file_path': os.path.join(DATA_DIR, 'dataset_7', 'emotions_formatted.csv')},
        {'name': 'DatasetEightLoader', 'file_path': os.path.join(DATA_DIR, 'dataset_8', 'emotions_ratings.txt')},
    ]

    if LOAD_DATA_SOURCES:
        print('Process execution started!')
        load_datasources(data)
        print('Process execution finished!')
    else:
        print('Process execution skipped!')

    transform_data()
