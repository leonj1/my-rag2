from pydantic import BaseModel
from typing import Optional


class Product(BaseModel):
    """Product document model."""
    title: str
    description: str
    link: str
    type: str = "product"


class Page(BaseModel):
    """Page document model."""
    description: str
    link: str
    type: str = "page"
