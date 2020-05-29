import json
import datetime
import sqlite3

class Adapter:

    def __init__(self, filename):
        self.conn = sqlite3.connect(filename)

    def __del__(self):
        self.conn.close()
    
    def generate_id(self, setting) -> str:
        experiment_id = "-".join([str(val) for item in sorted(setting.items(), key=lambda item: item[0]) for val in item])
        return experiment_id

    def put_experiment(self, setting, result):
        sql = """
        """

    def get_experiment_by_id(self, experiment_id):
        sql = """
        """

    