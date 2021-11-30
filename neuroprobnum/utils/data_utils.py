import os
import pickle


def make_dir(path):
    if not os.path.exists(path): os.makedirs(path)


def save_var(x, file):
    with open(file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)


def load_var(file):
    with open(file, 'rb') as f:
        x = pickle.load(f)
    return x


def file_exists(file):
    return os.path.isfile(file)
