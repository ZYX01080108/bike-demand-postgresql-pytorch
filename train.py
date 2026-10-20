from project_lib import connect, train

if __name__ == "__main__":
    with connect() as conn:
        train(conn)
