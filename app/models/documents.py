from pydantic import BaseModel
from typing import Optional


class Product(BaseModel):
    """Product document model."""
    title: str
    description: str
    link: str


class Page(BaseModel):
    """Page document model."""
    title: str
    description: str
    link: str
