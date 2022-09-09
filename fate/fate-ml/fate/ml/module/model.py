import tempfile


class ModelSaver:
    def __init__(self) -> None:
        self._states = {}

    def save_int(self, key: str, value: int):
        self._states[key] = value

    def save_file(self, key: str):
        """
        with saver.savefile("meta") as f:
            write_to_file(f)
        """
        return _SaverFileState(self, key)

    def save_dir(self, key: str):
        ...

    def save_bytes(self, key: str, bytes_value):
        self._states[key] = bytes_value


class ModelLoader:
    def __init__(self, states) -> None:
        self.states = states

    def load_int(self, key: str):
        return self.states[key]

    def load_file(self, key: str):
        """
        with loader.load_file("meta") as f:
            read_from_file(f)
        """
        return _LoaderFileState(self, self.states[key])

    def load_bytes(self, key: str):
        return self.states[key]


class _SaverFileState:
    def __init__(self, saver: ModelSaver, key) -> None:
        self.saver = saver
        self.key = key
        self.temperate = tempfile.NamedTemporaryFile()

    def __enter__(self):
        return self.temperate.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.temperate.seek(0)
        self.saver.save_bytes(self.key, self.temperate.read())
        self.temperate.__exit__(exc_type, exc_val, exc_tb)


class _LoaderFileState:
    def __init__(self, loader: ModelLoader, model_bytes) -> None:
        self.loader = loader
        self.model_bytes = model_bytes
        self.temperate = tempfile.NamedTemporaryFile()

    def __enter__(self):
        f = self.temperate.__enter__()
        f.write(self.model_bytes)
        f.seek(0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.temperate.__exit__(exc_type, exc_val, exc_tb)
