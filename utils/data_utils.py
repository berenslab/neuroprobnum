import os
import pickle
import numpy as np

############################################################################
def clean_folder(folder, prefix='', exceptions=[], verbose=True):
    if not(isinstance(exceptions, list)): exceptions = [exceptions]
    
    files = os.listdir(folder)
    for file in sorted(files):
        if file[0:len(prefix)] == prefix and not(os.path.isdir(folder+'/'+file)):
            if (file not in exceptions) and (file not in [exception + '.pkl' for exception in exceptions]):
                if verbose: print('\t Delete ' + file)
                os.remove(folder+'/'+file)
    
##########################################################################
def is_in_any_file(string, files):
    if not isinstance(files, list): files = [files]
    
    found = False
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if string in line:
                found = True
                break;
                            
    return found

##########################################################################
def make_dir(path):
    if not os.path.exists(path): os.makedirs(path)

############################################################################
def save_var(x, file):
    with open(file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

############################################################################
def load_var(file):
    try:
        with open(file, 'rb') as f:
            x = pickle.load(f)
    except:
        import pickle5
        with open(file, 'rb') as f:
            x = pickle5.load(f)
    return x

def file_exists(file):
    return os.path.isfile(file)