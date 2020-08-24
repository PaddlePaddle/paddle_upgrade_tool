import os

class FileNotFoundException(Exception):
    def __init__(self, message):
        super(FileNotFoundException, self).__init__(message)

class Query:
    def __init__(self, path):
        self.path = path
        self.transformer = []
        if not os.path.exists(self.path):
            raise FileNotFoundException('path: "{}" does not exist.'.format(self.path))

    def __str__(self):
        return 'Query("{}")'.format(self.path)
