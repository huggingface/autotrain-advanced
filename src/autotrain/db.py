import sqlite3


class AutoTrainDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()
        self.create_jobs_table()

    def create_jobs_table(self):
        self.c.execute(
            """CREATE TABLE IF NOT EXISTS jobs
            (id INTEGER PRIMARY KEY, pid INTEGER)"""
        )
        self.conn.commit()

    def add_job(self, pid):
        sql = f"INSERT INTO jobs (pid) VALUES ({pid})"
        self.c.execute(sql)
        self.conn.commit()

    def get_running_jobs(self):
        self.c.execute("""SELECT pid FROM jobs""")
        running_pids = self.c.fetchall()
        running_pids = [pid[0] for pid in running_pids]
        return running_pids

    def delete_job(self, pid):
        sql = f"DELETE FROM jobs WHERE pid={pid}"
        self.c.execute(sql)
        self.conn.commit()
