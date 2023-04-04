import sqlite3

class DB(object):
    def __init__(self, path) -> None:
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()
        
    def execute(self, query, params=None):
        self.cursor.execute(query, params)
        self.conn.commit()
        return self.cursor.fetchall()