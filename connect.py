import wrds
import os

def get_connection():
    conn = wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME"), wrds_password=os.getenv("WRDS_PASSWORD"))
    return conn





