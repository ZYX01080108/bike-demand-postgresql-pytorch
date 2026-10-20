from project_lib import connect, explore

if __name__ == "__main__":
    with connect() as conn:
        explore(conn)
