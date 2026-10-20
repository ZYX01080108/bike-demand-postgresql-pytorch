import argparse
from datetime import datetime
from project_lib import connect, predict_one

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict for a recorded hourly weather scenario.")
    parser.add_argument("--at", default="2012-12-30 17:00:00")
    args = parser.parse_args()
    with connect() as conn:
        predict_one(conn, datetime.fromisoformat(args.at))
