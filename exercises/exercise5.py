import zipfile
import urllib.request
import pandas as pd
import sqlite3

def download_and_extract_gtfs_data(url, zip_file_path, extract_folder):
    urllib.request.urlretrieve(url, zip_file_path)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

 
def load_and_filter_stops_data(csv_path):
    columns_to_select = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'zone_id']
    stops_df = pd.read_csv(csv_path, usecols=columns_to_select, dtype={'stop_id': 'str', 'stop_name': 'str', 'stop_lat': 'float', 'stop_lon': 'float', 'zone_id': 'int'}, encoding='utf-8')
    stops_df = stops_df[stops_df['zone_id'] == '2001']
    stops_df = stops_df[(stops_df['stop_lat'] >= -90) & (stops_df['stop_lat'] <= 90) & (stops_df['stop_lon'] >= -90) & (stops_df['stop_lon'] <= 90)]
    return stops_df


def create_database_and_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stops (
            stop_id TEXT PRIMARY KEY,
            stop_name TEXT,
            stop_lat  REAL ,
            stop_lon  REAL ,
            zone_id TEXT
        )
    ''')
    conn.commit()

def transfer_data_to_database(conn, stops_df):
    stops_df.to_sql("stops", conn, index=False, if_exists="replace", dtype={'stop_id': 'TEXT', 'stop_name': 'TEXT', 'stop_lat': 'REAL', 'stop_lon': 'REAL', 'zone_id': 'REAL'})
    conn.commit()

def main():
    url = "https://gtfs.rhoenenergie-bus.de/GTFS.zip"
    zip_file_path = "GTFS.zip"
    extract_folder = "gtfs_data"
    csv_path = f"{extract_folder}/stops.txt"
    database_name = "gtfs.sqlite"

    # Download and extract GTFS data
    download_and_extract_gtfs_data(url, zip_file_path, extract_folder)

    # Load and filter stops data
    stops_df = load_and_filter_stops_data(csv_path)

    # Create SQLite database and table
    conn = sqlite3.connect(database_name)
    create_database_and_table(conn)

    # Transfer data to the database
    transfer_data_to_database(conn, stops_df)

    # Close the connection
    conn.close()

if __name__ == "__main__":
    main()
