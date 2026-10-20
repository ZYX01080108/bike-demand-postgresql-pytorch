from datetime import datetime
from project_lib import connect, load_data, explore, train, predict_one

if __name__ == "__main__":
    with connect() as conn:
        load_data(conn)
        explore(conn)
        train(conn)
        predict_one(conn, datetime(2012, 12, 30, 17))
