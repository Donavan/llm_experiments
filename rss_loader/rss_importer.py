import json
import nltk
import openai
import logging
import asyncio
import argparse
import tiktoken
import feedparser
from dateutil import parser
from rss_repo import RSSRepo
from datetime import datetime
from asyncio import Semaphore
from bs4 import BeautifulSoup
from rss_loader.feed_entry import FeedEntry


class FeedProcessor:
    """Class to process RSS feeds and store entries in a database."""

    def __init__(self, db: RSSRepo, **kwargs) -> None:
        """Initialize the FeedProcessor.

        Args:
            db: The Database object to use.
        """
        self.db = db
        self.model_name: str = kwargs.get("model_name", "gpt-3.5-turbo-16k")
        self.temperature: float = kwargs.get("temperature", 0.0)
        self.max_delay: int = kwargs.get("max_delay", 10)
        self.concurrency_limit: int = kwargs.get("concurrency_limit", 3)
        self.semaphore: Semaphore = asyncio.Semaphore(self.concurrency_limit)
        self.encoder = tiktoken.encoding_for_model('gpt-4')

    def __exponential_backoff(self, delay):
        asyncio.sleep(min(2 * delay, self.max_delay))

    def _chat_one_shot(self, prompt: str, user_content: str, model_name: str, max_tokens=0):
        delay = 1  # Initial delay between retries
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ]

        while True:
            try:
                if max_tokens > 0:
                    response = openai.ChatCompletion.create(model=model_name, temperature=self.temperature, messages=messages, max_tokens=max_tokens)
                else:
                    response = openai.ChatCompletion.create(model=model_name, temperature=self.temperature, messages=messages)
                response_message = response["choices"][0]["message"]
                return response_message["content"].strip(), response['usage']['completion_tokens']

            except openai.error.OpenAIError as e:
                logging.exception("OpenAIError occurred: %s", e)
                self.__exponential_backoff(delay)
                delay *= 2
            except Exception as e:
                logging.exception("Error occurred during chat completion: %s", e)
                raise

    def parse_feed(self, url: str) -> None:
        """Fetch and parse an RSS feed and add new entries to the database.

        Args:
            url: The URL of the RSS feed to parse.

        Raises:
            Exception: If an error occurs when fetching or parsing the feed.
        """
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                self.add_entry(entry)

            self.update_summaries()
            self.ner_extraction()
        except Exception as e:
            print(f"An error occurred when parsing the feed: {e}")
            raise e

    def add_entry(self, entry: feedparser.FeedParserDict) -> None:
        """Add an entry to the database if it does not already exist.

        Args:
            entry: The entry to add.

        Raises:
            ValueError: If an error occurs when parsing the date.
        """
        if self.db.have_link(entry.link):
            print(f"Skipping {entry.link} as it already exists...")
            return

        print(f"Adding {entry.link}")
        dc_subject = None
        if 'tags' in entry:
            dc_subject = ', '.join([tag.term for tag in entry.tags])

        desc_tokens = len(self.encoder.encode(entry.description))
        try:
            pub_date = parser.parse(entry.published)
        except ValueError:
            print("An error occurred when parsing the date. Using the current date and time instead.")
            pub_date = datetime.now()

        entry_id = self.db.add_entry(entry.title, entry.link, pub_date, entry.description, dc_subject, entry.author, desc_tokens)

        soup = BeautifulSoup(entry.description, 'html.parser')
        for a in soup.find_all('a', href=True):
            if 'name' in a.get('class', []):
                self.db.add_related_bio(entry_id, a.text, a['href'])
            else:
                self.db.add_related_link(entry_id, a['href'])

        self.db.commit()

    def update_summaries(self):
        entries = self.db.entries_needing_summaries()

        for entry in entries:
            entry.summary, entry.summary_tokens = self.summarize(entry)
            self.db.update_summary(entry)
            self.db.commit()

    def summarize(self, entry: FeedEntry):
        print(f"Summarizing {entry.link}")
        soup = BeautifulSoup(entry.description, features="html.parser")
        description_text = soup.get_text()

        prompt = 'Produce a concise summary of the news article in the user message.'
        return self._chat_one_shot(prompt, description_text, self.model_name)

    def ner_extraction(self):
        entries = self.db.entries_without_ner()
        for entry in entries:

            ner_data = self.perform_ner(entry)

            for ner in ner_data:
                self.db.add_ner(entry, ner['type'], ner['entity'])

            self.db.commit()

    def perform_ner(self, entry):
        soup = BeautifulSoup(entry.description, features="html.parser")
        description_text = soup.get_text()
        chunks = self.split_text_for_ner(description_text, 4000)
        entities = []
        chunk_no = 1
        for chunk in chunks:
            print(f"Running NER on {entry.link}.  Chunk {chunk_no} of {len(chunks)}.")
            chunk_no = chunk_no + 1
            entities.extend(self.perform_ner_chunk(chunk))

        unique_entities = set()  # Create an empty set to store unique entities
        no_duplicates = []  # Array to store dicts with no duplicate entities

        for dict_ in entities:
            if dict_["entity"] not in unique_entities:  # Check if entity is unique
                no_duplicates.append(dict_)  # If it is, add dict to no_duplicates
                unique_entities.add(dict_["entity"])  # And add entity to unique_entities

        return no_duplicates



    def split_text_for_ner(self, text, token_limit):
        # Split text into paragraphs
        paragraphs = nltk.tokenize.blankline_tokenize(text)

        # Array to hold result
        result = []

        # Temporary variable to hold paragraphs
        temp_paragraphs = ""

        for paragraph in paragraphs:
            # Calculate tokens in the current paragraph
            tokens = len(self.encoder.encode(paragraph))

            # If tokens in the current paragraph and the temporary paragraphs exceed the limit
            # add the temporary paragraphs to the result and start a new temporary paragraphs
            if tokens + len(self.encoder.encode(temp_paragraphs)) > token_limit:
                result.append(temp_paragraphs.strip())
                temp_paragraphs = paragraph
            else:
                # Otherwise, add the current paragraph to the temporary paragraphs
                temp_paragraphs += "\n" + paragraph

        # Don't forget to add the last batch of paragraphs
        result.append(temp_paragraphs.strip())

        return result

    def perform_ner_chunk(self, chunk):
        prompt = 'extract all named entities in JSON format as an array of objects with "type" and "entity" fields from the text in the user message. Before including each entity, verify that it has not already been included.  Your output should consist of only json.'
        while True:
            try:
                js_text, tokens = self._chat_one_shot(prompt, chunk, 'gpt-4-0613')
                return json.loads(js_text)
            except:
                print(f"NER failure retrying\n{js_text}")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Parse an RSS feed and store new entries in a database.")
    arg_parser.add_argument("url", help="The URL of the RSS feed to parse.")
    args = arg_parser.parse_args()

    db = RSSRepo('rss.db')
    # db.delete_all_ner()

    processor = FeedProcessor(db)
    processor.parse_feed(args.url)

    db.close()
