import pandas as pd
import numpy as np
import os
from pathlib import Path
from os import path
import datetime
import subprocess
import re
import csv
import _pickle as cPickle

def disp(df):
    [m,n]= df.shape
    pd.set_option('display.max_rows', m,'display.max_columns',n)
    display(df)
    pd.reset_option('display.max_rows')
    
    
def tuplizer(x):
    return tuple(x) if isinstance(x, (np.ndarray, list)) else x

  
  

def free():
    # Get process info
    ps = subprocess.Popen(['ps', '-caxm', '-orss,comm'], stdout=subprocess.PIPE).communicate()[0].decode()
    vm = subprocess.Popen(['vm_stat'], stdout=subprocess.PIPE).communicate()[0].decode()

    # Iterate processes
    processLines = ps.split('\n')
    sep = re.compile('[\s]+')
    rssTotal = 0 # kB
    for row in range(1,len(processLines)):
        rowText = processLines[row].strip()
        rowElements = sep.split(rowText)
        try:
            rss = float(rowElements[0]) * 1024
        except:
            rss = 0 # ignore...
        rssTotal += rss

    # Process vm_stat
    vmLines = vm.split('\n')
    sep = re.compile(':[\s]+')
    vmStats = {}
    for row in range(1,len(vmLines)-2):
        rowText = vmLines[row].strip()
        rowElements = sep.split(rowText)
        vmStats[(rowElements[0])] = int(rowElements[1].strip('\.')) * 4096

    print (f'Wired Memory:\t\t {np.round(vmStats["Pages wired down"]/2**30,2):10} GB')
    print (f'Active Memory:\t\t {np.round(vmStats["Pages active"]/2**30,2):10} GB')
    print (f'Inactive Memory:\t {np.round(vmStats["Pages inactive"]/2**30,2):10} GB')
    print (f'Free Memory:\t\t {np.round(vmStats["Pages free"]/2**30,2):10} GB')
    print (f'Total Real Memory:\t {np.round(rssTotal/2**30,2):10} GB')
    
    
def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return np.array([bind.get(itm, None) for itm in a])  
  
  
def nnz(a):
  return np.count_nonzero(a)


def rename_categories(idx,mapping):
    keys = np.array(list(mapping.keys()))
    values = np.array(list(mapping.values()))
    mapping_array = np.zeros(keys.max()+1, dtype=values.dtype)
    mapping_array[keys] = values
    return mapping_array[idx]

def list_to_csv(my_list,file_name,header):
    np.savetxt(file_name, my_list, delimiter=",", fmt='%s', header=header)

def file_parts(file):
    head_tail = os.path.split(file)
    path = head_tail[0]
    file_name, file_type = head_tail[1].split('.')
    return path, file_name, file_type


def find_duplicate(L):
    seen,duplicate = set(),set()
    index = np.zeros(len(L), dtype=bool)
    seen_add, duplicate_add = seen.add, duplicate.add
    for idx,item in enumerate(L):
        if item in seen:
            duplicate_add(item)
            index[idx] = True
        else:
            seen_add(item)
    
    return ismember(L, list(duplicate)) != None, duplicate


def file_date(path_to_file):
    stat = os.stat(path_to_file)
    date = datetime.datetime.fromtimestamp(stat.st_mtime)
    return date, stat


def save_pickle(pickle_file, data):
    if not path.exists(pickle_file):
            filepath, _, _ = file_parts(pickle_file)
            Path(filepath).mkdir(parents=True, exist_ok=True)
    output_pickle = open(pickle_file, "wb")
    cPickle.dump(data, output_pickle)
    output_pickle.close()

def load_pickle(pickle_file):
    if path.exists(pickle_file) and path.getsize(pickle_file) > 0:
        input_pickle = open(pickle_file, 'rb')
        data = cPickle.load(input_pickle)
        input_pickle.close()
        return data
    return False


def get_mapper(value,key):
    return pd.Series(value,index=key).to_dict()
