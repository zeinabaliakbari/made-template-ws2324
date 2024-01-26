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
    stops_df = stops_df[(stops_df['zone_id'] == 2001) & 
                        (stops_df['stop_lat'] >= -90) & (stops_df['stop_lat'] <= 90) & 
                        (stops_df['stop_lon'] >= -90) & (stops_df['stop_lon'] <= 90)]
    print("Number of rows after filtering:", len(stops_df))
    print(stops_df.head())

    return stops_df

def connect_to_database(db_name):
    conn = sqlite3.connect(db_name)
    return conn, conn.cursor()

def create_sqlite_table(cursor, table_name, sqlite_types):
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {', '.join([f'{col} {col_type}' for col, col_type in sqlite_types.items()])}
    );
    """
    cursor.execute(create_table_query)

def write_to_sqlite(df, conn, table_name, sqlite_types):
    df.to_sql(table_name, conn, if_exists='replace', index=False, dtype=sqlite_types)

def close_connection(conn):
    conn.commit()
    conn.close()
    print("Connection closed.")

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
    db_name = "gtfs.sqlite"
    conn, cursor = connect_to_database(db_name)

    # Create SQLite table with appropriate types
    sqlite_types = {'stop_id': 'TEXT', 'stop_name': 'TEXT', 
                    'stop_lat': 'REAL', 'stop_lon': 'REAL', 
                    'zone_id': 'INTEGER'}
    table_name = 'stops'
    create_sqlite_table(cursor, table_name, sqlite_types)

    # Write DataFrame to SQLite database
    write_to_sqlite(stops_df, conn, table_name, sqlite_types)

    # Close the connection
    close_connection(conn)

if __name__ == "__main__":
    main()
