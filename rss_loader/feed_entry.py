from datetime import datetime
from dataclasses import dataclass


@dataclass
class FeedEntry:
    id: int
    title: str
    link: str
    pubDate: datetime
    description: str
    description_tokens: int
    dc_subject: str
    dc_creator: str
    summary: str
    summary_tokens: int

