import sqlite3
from datetime import datetime
from feed_entry import FeedEntry


class RSSRepo:
    """Class to handle SQLite database operations."""

    def __init__(self, db_path: str):
        """Initialize the Database.

        Args:
            db_path: The path to the SQLite database file.
        """
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            print(f"An error occurred when connecting to the database: {e}")

        self.__create_tables()

    def __create_tables(self):
        """Create the entries, related_links, and related_bios tables in the database if they do not already exist."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS entries
            (id INTEGER PRIMARY KEY, title TEXT, link TEXT, pubDate DATETIME, description TEXT, description_tokens INTEGER, dc_subject TEXT, dc_creator TEXT, summary TEXT, summary_tokens INTEGER)
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS related_links
            (id INTEGER PRIMARY KEY, entry_id INTEGER, url TEXT, FOREIGN KEY(entry_id) REFERENCES entries(id))
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS related_bios
            (id INTEGER PRIMARY KEY, entry_id INTEGER, name TEXT, url TEXT, FOREIGN KEY(entry_id) REFERENCES entries(id))
        ''')
        self.cursor.execute('''
                   CREATE TABLE IF NOT EXISTS entry_ner
                   (id INTEGER PRIMARY KEY, entry_id INTEGER, entity_type TEXT, entity TEXT, FOREIGN KEY(entry_id) REFERENCES entries(id))
               ''')

    def close(self):
        """Close the connection to the database."""
        self.conn.close()

    def have_link(self, url: str):
        self.cursor.execute('''
                   SELECT id FROM entries WHERE link = ?
               ''', (url,))

        return self.cursor.fetchone() is not None

    def add_entry(self, title: str, link: str, pub_date: datetime, description: str, subject: str, creator: str, description_tokens: int):
        self.cursor.execute('''
                   INSERT INTO entries (title, link, pubDate, description, dc_subject, dc_creator, description_tokens, summary, summary_tokens)
                   VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 0)
               ''', (title, link, pub_date, description, subject, creator, description_tokens))

        return self.cursor.lastrowid

    def add_related_link(self, entry_id: int, url: str) -> None:
        """Add a related link to the database.

        Args:
            entry_id: The id of the entry the link is related to.
            url: The URL of the related link.
        """
        self.cursor.execute('''
            INSERT INTO related_links (entry_id, url)
            VALUES (?, ?)
        ''', (entry_id, url))

    def add_related_bio(self, entry_id: int, name: str, url: str) -> None:
        """Add a related biography to the database.

        Args:
            entry_id: The id of the entry the bio is related to.
            name: The name of the person.
            url: The URL of the biography.
        """
        self.cursor.execute('''
            INSERT INTO related_bios (entry_id, name, url)
            VALUES (?, ?, ?)
        ''', (entry_id, name, url))

    def entries_needing_summaries(self):
        self.cursor.execute('''
             SELECT * FROM entries WHERE summary IS NULL
         ''')

        return list(map(lambda row: FeedEntry(*row), self.cursor.fetchall()))

    def entries_without_ner(self):
        """Fetch entries that don't have any related bios."""
        self.cursor.execute('''
            SELECT entries.* FROM entries 
            LEFT JOIN entry_ner ON entries.id = entry_ner.entry_id 
            WHERE entry_ner.entry_id IS NULL
        ''')

        entries = list(map(lambda row: FeedEntry(*row), self.cursor.fetchall()))
        return entries

    def commit(self):
        self.conn.commit()

    def update_summary(self, entry):
        self.cursor.execute('''
                            UPDATE entries SET summary = ?, summary_tokens = ? WHERE id = ?
                        ''', (entry.summary, entry.summary_tokens, entry.id))

    def add_ner(self, entry, type, entity):
        self.cursor.execute('''
                    INSERT INTO entry_ner (entry_id, entity_type, entity)
                    VALUES (?, ?, ?)
                ''', (entry.id, type, entity))

    def delete_all_ner(self) -> None:
        self.cursor.execute('''
            DELETE FROM entry_ner
        ''')
        self.conn.commit()