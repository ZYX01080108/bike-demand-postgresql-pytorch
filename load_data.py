from project_lib import connect, load_data

if __name__ == "__main__":
    with connect() as conn:
        load_data(conn)
