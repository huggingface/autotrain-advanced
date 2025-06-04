import sqlite3
import logging

logger = logging.getLogger(__name__)


class AutoTrainDB:
    """
    A class to manage job records in a SQLite database.

    Attributes:
    -----------
    db_path : str
        The path to the SQLite database file.
    conn : sqlite3.Connection
        The SQLite database connection object.
    c : sqlite3.Cursor
        The SQLite database cursor object.

    Methods:
    --------
    __init__(db_path):
        Initializes the database connection and creates the jobs table if it does not exist.

    create_jobs_table():
        Creates the jobs table in the database if it does not exist.

    add_job(pid):
        Adds a new job with the given process ID (pid) to the jobs table.

    get_running_jobs():
        Retrieves a list of all running job process IDs (pids) from the jobs table.

    delete_job(pid):
        Deletes the job with the given process ID (pid) from the jobs table.
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()
        self.create_jobs_table()

    def create_jobs_table(self):
        """
        Create the jobs table if it doesn't exist.
        If table exists but structure is different, migrate it.
        """
        try:
            # First check if table exists
            self.c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
            if self.c.fetchone():
                # Table exists, check structure
                self.c.execute("PRAGMA table_info(jobs)")
                columns = {row[1] for row in self.c.fetchall()}
                
                # If start_time column is missing, add it
                if 'start_time' not in columns:
                    self.c.execute("ALTER TABLE jobs ADD COLUMN start_time TEXT")
                    self.conn.commit()
            else:
                # Create new table with all columns
                self.c.execute("""
                    CREATE TABLE jobs (
                        id INTEGER PRIMARY KEY,
                        pid INTEGER,
                        start_time TEXT
                    )
                """)
                self.conn.commit()
        except Exception as e:
            logger.error(f"Error creating/migrating jobs table: {str(e)}")
            raise

    def add_job(self, pid):
        """
        Add a job to the database.
        
        Args:
            pid (int): Process ID of the job
        """
        try:
            # First check if job exists
            self.c.execute("SELECT pid FROM jobs WHERE pid = ?", (pid,))
            if self.c.fetchone():
                # Job exists, remove it first
                self.delete_job(pid)
            
            # Add new job
            sql = "INSERT INTO jobs (pid, start_time) VALUES (?, datetime('now'))"
            self.c.execute(sql, (pid,))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error adding job to database: {str(e)}")
            raise

    def get_running_jobs(self):
        self.c.execute("""SELECT pid FROM jobs""")
        running_pids = self.c.fetchall()
        running_pids = [pid[0] for pid in running_pids]
        return running_pids

    def delete_job(self, pid):
        sql = f"DELETE FROM jobs WHERE pid={pid}"
        self.c.execute(sql)
        self.conn.commit()
