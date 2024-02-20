import sqlite3
from sqlite3 import Error
import pandas as pd
from tqdm import tqdm


class DBDriver:
    def __init__(self, db_file_path):
        # Initialize the database connection based on the provided file path
        self.db_file_path = db_file_path
        self.connection = self.create_connection()

    def create_connection(self):
        try:
            connection = sqlite3.connect(self.db_file_path)
            return connection
        except Error as e:
            print(f"Error: {e}")
            return None

    def create_tables(self):
        # Define the table structure
        create_table_query = """
            CREATE TABLE IF NOT EXISTS soul_emotions (
                id INTEGER PRIMARY KEY,
                source_id INTEGER,
                dataset_id INTEGER,
                text TEXT,
                source_emotion TEXT,
                shared_emotion TEXT,
                quadrant_emotion TEXT,
                valence REAL,
                arousal REAL
            )
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(create_table_query)
            self.connection.commit()
        except Error as e:
            print(f"Error creating table: {e}")

    def write(self, db_row):
        # Insert the provided DBRow into the database
        insert_query = """
            INSERT INTO soul_emotions (
                source_id, dataset_id, text, source_emotion, shared_emotion,
                quadrant_emotion, valence, arousal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(insert_query, (
                db_row.source_id,
                db_row.dataset_id,
                db_row.text,
                db_row.source_emotion,
                db_row.shared_emotion,
                db_row.quadrant_emotion,
                db_row.valence,
                db_row.arousal
            ))
            self.connection.commit()
        except Error as e:
            print(f"Error inserting data: {e}")

    def get_rows_by_dataset_id(self, dataset_id_param):
        # Retrieve rows from the database where dataset_id matches the parameter
        select_query = """
            SELECT * FROM soul_emotions WHERE dataset_id = ?
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(select_query, (dataset_id_param,))
            rows = cursor.fetchall()
            return rows
        except Error as e:
            print(f"Error retrieving data: {e}")
            return []

    def close(self):
        # Close the database connection
        if self.connection:
            self.connection.close()

    def get_all_rows_as_dataframe(self):
        # Retrieve all rows from the database and return them as a DataFrame
        select_query = """
            SELECT * FROM soul_emotions
        """
        try:
            return pd.read_sql_query(select_query, self.connection)
        except Error as e:
            print(f"Error retrieving data as DataFrame: {e}")
            return pd.DataFrame()

    def update_dataframe_to_database(self, df):
        # Check if the columns in the DataFrame match the database columns
        if not set(df.columns).issubset({'id', 'source_id', 'dataset_id', 'text', 'source_emotion', 'shared_emotion',
                                         'quadrant_emotion', 'valence', 'arousal'}):
            print("Error: DataFrame columns do not match the database columns.")
            raise ValueError("DataFrame columns do not match the database columns.")

        rows_updated = 0
        total_rows = len(df)

        # Use tqdm to create a progress bar
        for _, row in tqdm(df.iterrows(), total=total_rows, desc="Updating rows"):
            source_id = row['source_id']

            # Check if the source_id exists in the database
            select_query = """
                SELECT COUNT(*) FROM soul_emotions WHERE source_id = ?
            """
            cursor = self.connection.cursor()
            cursor.execute(select_query, (source_id,))
            row_count = cursor.fetchone()[0]

            if row_count == 1:
                # Update the row in the database with the data from the DataFrame
                update_query = """
                    UPDATE soul_emotions
                    SET dataset_id = ?, text = ?, source_emotion = ?, shared_emotion = ?,
                        quadrant_emotion = ?, valence = ?, arousal = ?
                    WHERE source_id = ?
                """
                cursor.execute(update_query, (row['dataset_id'], row['text'], row['source_emotion'],
                                              row['shared_emotion'], row['quadrant_emotion'], row['valence'],
                                              row['arousal'], source_id))
                self.connection.commit()
                rows_updated += 1

        print(f"Total rows in DataFrame: {total_rows}")
        print(f"Total rows updated in the database: {rows_updated}")
        print(f"Total orphan rows (source_id not found in the database): {total_rows - rows_updated}")

    def execute_query(self, query, params=None):
        try:
            cursor = self.connection.cursor()
            if params is None:
                cursor.execute(query)
            else:
                cursor.execute(query, params)
            self.connection.commit()
        except Error as e:
            print(f"Error executing query: {e}")
