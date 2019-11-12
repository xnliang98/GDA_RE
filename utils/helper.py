'''
Helper functions.
'''

import os
import json
import time

# Functions about directory or files
def check_dir(d):
    ''' Exit if directory d does not exist. 
        Args:
            d (str): directory's name or path.
    '''
    if not os.path.exists(d):
        print("Directory {} does not exist, exit.".format(d))
        exit(1)

def check_files(files):
    ''' Exit if any one file in files do not exist.
        Args:
            files (list): list of filenames.
    '''
    for f in files:
        if f is not None and not os.path.exists(f):
            print("File {} does not exist, exit.".format(f))
            exit(1)

def ensure_dir(d, verbose=True):
    ''' Make sure directory exist, if directory not exists,
    create the directory.
        Args:
            d (str): directory's name or path.
            verbose (boolean): if true, print verbose info.
    '''
    if not os.path.exists(d):
        print("Directory {} do not exist, creating ...".format(d))
        os.makedirs(d)

# Functions about config's save, load, print
def save_config(config, path, verbose=True):
    ''' Save config to json file with path.
        Args:
            config (dict): config of model and train.
            path (str): path to save config file.
            verbose (boolean): if true, print verbose info.
    '''
    with open(path, 'w') as fout:
        json.dump(config, fout, indent=2)
    if verbose:
        print("Config saved to file {}.".format(path))
    return config

def load_config(path, verbose=True):
    ''' Load config from json file with path and return config.
        Args:
            path (str): path to load config file.
            verbose (boolean): if true, print verbose info.
    '''
    with open(path, 'r') as fin:
        config = json.load(fin)
    if verbose:
        print("Config has been loaded from file {}.".format(path))
    return config

def print_config(config):
    ''' Print config.
        Args:
            config (dict): config need to be printed.
    '''
    info = "Running with the following configs:\n"
    for k,v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    print("\n" + info + "\n")
    return None

class FileLogger(object):
    """ A file logger that opens the file periodically and write to it."""
    def __init__(self, filename='logs', header=None):
        # Use time of creating FileLogger object as filename
        self.filename = filename + '_' + time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)
    
    def log(self, message):
        with open(self.filename, 'a') as out:
            print(message, file=out)
