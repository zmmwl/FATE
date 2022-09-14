from typing import Protocol


class Summary(Protocol):
    def save(self):
        ...
