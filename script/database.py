import json
import sqlite3
from tinydb import TinyDB, Query


class db:
    def __init__(self, filename):
        self.conn = sqlite3.connect(filename)
    
    def __del__(self):
        self.conn.close()
    
    def put_file(self, filename):
        assert("/" not in filename)

        insert_sql = """
        INSERT INTO file (filename, content, num_vertices, num_edges, num_anchors)
            VALUES (:filename, :content, :num_vertices, :num_edges, :num_anchors);
        """
        mapping = {
            "filename": filename
        }

        with open(filename, "r") as fp:
            lines = [line for line in fp.readlines() if line != ""]
            mapping["content"] = "\n".join(lines)
        
        with self.conn:
            cursor = self.conn.cursor()
            try:
                vert_ind = lines.index("P")
                num_vertices = int(lines[vert_ind + 1])
            except ValueError:
                raise ValueError("num_vertices = 0")

            try:
                edge_ind = lines.index("E")
                num_edges = int(lines[edge_ind + 1])
            except ValueError:
                raise ValueError("num_edges = 0")

            try:
                anchor_ind = lines.index("anchors")
                num_anchors = int(lines[anchor_ind + 1])
            except ValueError:
                num_anchors = 0

            mapping["num_vertices"] = num_vertices
            mapping["num_edges"] = num_edges
            mapping["num_anchors"] = num_anchors

            cursor.execute(insert_sql, mapping)
        
            return cursor.lastrowid
    
    def put_opt_setting(self, setting):

        insert_sql = """
        INSERT OR REPLACE INTO opt_setting (epsilon, cost, file_id, json)
        VALUES (:epsilon, :cost, :file_id, :json);
        """

        mapping = {
            "epsilon": setting["epsilon"],
            "cost": setting["cost"],
            "file_id": setting["file_id"],
            "json": json.dumps(setting)
        }

        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(insert_sql, mapping)

            return cursor.lastrowid

    def put_result(self, result):
        "SLE"