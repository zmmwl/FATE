from typing import List
from fate.interface import Cache as CacheInterface

class Cache(CacheInterface):
    def __init__(self) -> None:
        self.cache: List = []
