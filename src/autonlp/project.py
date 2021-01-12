import requests


class Project:
    def __init__(self, name, org, user):
        self.name = name
        self.org = org
        self.user = user

    def upload(self, files, split, col_mapping):
        pass

    def train(self):
        pass
