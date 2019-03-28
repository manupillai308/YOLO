
class BaseModel(object):
    """Base model class."""

    def init(self):
        raise NotImplementedError

    def evaluate(self, task):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
