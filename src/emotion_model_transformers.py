from tqdm import tqdm

from src.database import DBDriver
from prettytable import PrettyTable


class EmotionModelTransformer:
    def __init__(self, db_driver: DBDriver):
        self.db_driver = db_driver

    def reduce_to_quadrant_emotions(self):
        update_query = """
            UPDATE soul_emotions
            SET quadrant_emotion = 
              CASE
                WHEN valence > 0 AND arousal > 0 THEN 'Q1'
                WHEN valence > 0 AND arousal < 0 THEN 'Q2'
                WHEN valence < 0 AND arousal < 0 THEN 'Q3'
                WHEN valence < 0 AND arousal > 0 THEN 'Q4'
                ELSE 'Q0'  -- Handle the case when both valence and arousal are 0
              END
        """

        self.db_driver.execute_query(update_query)
        print("Reduction to Quadrant Emotions complete. Data updated in the database.")

    def reduce_emotions(self, mapping):
        data = self.db_driver.get_all_rows_as_dataframe()
        unique_source_emotions = data['source_emotion'].unique()
        if not set(unique_source_emotions).issubset(set(mapping.keys())):
            missing_emotions = set(unique_source_emotions) - set(mapping.keys())
            raise ValueError(f"Mapping dictionary is missing emotions: {missing_emotions}")

        progress_bar = tqdm(mapping.items(), total=len(mapping), desc="Reducing emotions")
        for source_emotion, shared_emotion in progress_bar:
            update_query = """
                UPDATE soul_emotions
                SET shared_emotion = ?
                WHERE source_emotion = ?
            """
            self.db_driver.execute_query(update_query, (shared_emotion, source_emotion))
        print("Transformation complete. Data updated in the database.")

    def transform_to_russell(self, mapping):
        data = self.db_driver.get_all_rows_as_dataframe()
        unique_source_emotions = data['source_emotion'].unique()

        if not set(unique_source_emotions).issubset(set(mapping.keys())):
            missing_emotions = set(unique_source_emotions) - set(mapping.keys())
            raise ValueError(f"Mapping dictionary is missing emotions: {missing_emotions}")

        total_emotions = len(mapping)
        with tqdm(total=total_emotions, desc="Transforming to Russell's model") as pbar:
            for source_emotion, (valence, arousal) in mapping.items():
                update_query = """
                    UPDATE soul_emotions
                    SET valence = ?, arousal = ?
                    WHERE source_emotion = ?
                """
                self.db_driver.execute_query(update_query, (valence, arousal, source_emotion))
                pbar.update(1)  # Update the progress bar
        print("Transformation to Russell's model complete. Data updated in the database.")

    def show_statistics(self):
        data = self.db_driver.get_all_rows_as_dataframe()
        distinct_source_emotions = data['source_emotion'].unique()
        distinct_source_ids = data['source_id'].unique()
        total_rows = len(data)
        num_distinct_source_emotions = len(distinct_source_emotions)
        distinct_source_emotions.sort()
        num_distinct_source_ids = len(distinct_source_ids)

        table = PrettyTable()
        table.field_names = ["Emotions", "Sources", 'Total Rows']
        table.add_row([num_distinct_source_emotions, num_distinct_source_ids, total_rows])
        print(table)
        print("\nDistinct source emotions:" + str(distinct_source_emotions))
