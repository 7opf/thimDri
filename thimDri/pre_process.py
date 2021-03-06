import numpy as np
import pandas as pd
import time
import sys
import zipfile
import os
from pathlib import Path
from os import path
import _pickle as cPickle
import datetime
import subprocess
import re
import csv




def tidy_obs_domains(_df):
    '''
    Separate the large observation Dataframe to domain specific subsets 
    i.e. location, sleep and physiology
    
    INPUT
    =====
    _df = Pandas Dataframe parsed from the Observation.csv contained within 
    the thim zip files 

    RETURNS
    ======
    df = a dictionary containing the three dataframes representing the three domains 
    '''
    df = {}
    df['location'] = _df[_df.location.notnull()]
    df['location'] = df['location'].drop(columns = ["datetimeReceived","provider","valueQuantity",                                                                 "valueUnit","valueDatetimeStart","valueDatetimeEnd"])
    df['sleep'] = _df[_df.type.isin(['258158006','29373008','248218005','60984000','89129007','307155000',                                          '67233009','421355008'])]
    df['sleep'] = df['sleep'].drop(columns = ["datetimeReceived","provider","location","valueBoolean",                                                       "valueState"])
    idx1,idx2 = df['sleep'].type.unique(),df['location'].type.unique()
    df['physiology']  = _df[(~_df.type.isin(idx1)) & (~_df.type.isin(idx2))]
    df['physiology']  = df['physiology'] .drop(columns = ["datetimeReceived","provider","location",                                              "valueBoolean","valueState","valueDatetimeStart","valueDatetimeEnd"])
    return df


def parse_observation(df,zip_file,pickle_file,verbose=False):
    '''
    checks if observation file already exists and if not parses the 
    one contained within the zip_file which is then stored as a pickle locally 
    
    INPUT
    =====
    df = Either an already preprocessed dictionary or a boolean
    zip_file = a string directing to a zip file 
    pickle_file = a string direction to store the processed data
    verbose = indicating whether to print out steps timming
    
    RETURNS
    ======
    a dictionary containing observation data frames 
    ''' 
    date,_ = file_date(zip_file)
    _, name, _ = file_parts(zip_file)
    if isinstance(df, bool) or df['date']<date:
        if verbose:tic = time.perf_counter()
        _zip = zipfile.ZipFile(zip_file)
        _df = pd.read_csv(_zip.open('Observations.csv'),encoding = 'unicode_escape',low_memory=False)
        if verbose:        
            print(f"Loading csv file took : {time.perf_counter()-tic:0.2f} seconds")
            tic = time.perf_counter()
        _df = pid_mapping(_df, _zip)
        _df['datetimeObserved'] = pd.to_datetime(_df['datetimeObserved'])
        _df = _df.sort_values(by=['datetimeObserved'])
        _df = _df.reset_index(drop=True)
        _df['project'] = name
        _df['project'] = pd.Categorical(_df['project'])
        df = tidy_obs_domains(_df)
        df['date'] = date
        if verbose:        
            print(f"Processing file took : {time.perf_counter()-tic:0.2f} seconds")
            tic = time.perf_counter()
        save_pickle(pickle_file, df)
        if verbose:        
            print(f"Saving pickle: {time.perf_counter()-tic:0.2f} seconds")
    return df

def load_observation(zip_file,output_path='Data/pkl/',verbose=True):
    '''
    Given an thim zip_file this function compares the date of any 
    already existing processed data and if it finds none it will parse 
    the Observation.csv file and store the processed data locally under 
    the output_path
        
    INPUT
    =====
    zip_file = a string directing to a thim zip file 
    output_path = a string direction to local path
    verbose = indicating whether to print out steps timming
    
    RETURNS
    ======
    three Pandas Dataframes in the following order 
    df['location'],df['sleep'],df['physiology']
    ''' 
    if verbose:tic = time.perf_counter()
    _, name, _ = file_parts(zip_file)
    date,_ = file_date(zip_file)
    date = date.strftime("%Y%m%d")
    pickle_file = f"{output_path}{name}_{date}_Obs.pkl"
    df = load_pickle(pickle_file)
    df = parse_observation(df,zip_file,pickle_file,verbose)
    if verbose:        
        print(
            f"Elapsed time to load observation {name} file: {time.perf_counter()-tic:0.2f} seconds")
    return df['location'],df['sleep'],df['physiology']  


def pid_mapping(_df, _zip):
    '''
    Given a semi-processed pandas DataFrame _df and a zipfile object 
    associated with a thim zip file this function will try to use the 
    # internal Patients.csv file to convert anonymised patient id to internal numeric id 
        
    INPUT
    =====
    _df = a semi-processed pandas DataFrame containing a subject column
    _zip = a zipfile object  containing a Patients.csv file
    
    RETURNS
    ======
    _df = a semi-processed pandas DataFrame containing the old subject 
    column under project_id and a new remapped subject column
    ''' 
    try:
        _pid = pd.read_csv(_zip.open('Patients.csv'))
    except:
        _tmp = zipfile.ZipFile('Data/tihmdri.zip')
        _pid = pd.read_csv(_tmp.open('Patients.csv'))
        del _tmp
    _df.subject = pd.Categorical(_df.subject)
    _pid = _pid.iloc[ismember(_df.subject.cat.categories, _pid.subjectId)]
    mapping = pd.Series(_pid.sabpId.values,
                        index=_pid.subjectId.values).to_dict()
    _df['project_id'] = _df.subject
    _df.subject = _df.subject.cat.rename_categories(mapping)
    return _df


def merge_observations(files,output_path='Data/pkl/',verbose=True):
    '''
    Given a list of paths to zip files this function will iteratively 
    pre-process all observation.csv files and merge them to 
    one unified Dataframe per experimental domains 
            
    INPUT
    =====
    files = a list of file paths to thim zip files 
    output_path = a place where already pre-processed data can be 
    either stored or retrived 
    verbose = indicating whether to print out steps timming
    
    RETURNS
    ======
    three Pandas Dataframes in the following order 
    df['location'],df['sleep'],df['physiology']
    ''' 
    if verbose:tic = time.perf_counter()
    pickle_file = f"{output_path}merged_Observations.pkl"
    df = load_pickle(pickle_file)
    if isinstance(df, bool):
        k = ['location', 'sleep', 'physiological']
        df = {key: ['0']*len(files) for key in k}

        for idx,fid in enumerate(files):
            df[k[0]][idx], df[k[1]][idx], df[k[2]][idx] = load_observation(fid,output_path=output_path, verbose=verbose)

        df = {key: pd.concat([_df for _df in df[key]]) for key in df.keys()}
        
        df['location'] = pd.get_dummies(df['location'], columns=['location'], prefix='', prefix_sep='')

        for key in df.keys():
            df[key].subject = pd.Categorical(df[key].subject)

        save_pickle(pickle_file, df)
    
    if verbose:        
        print(f"Elapsed time to load merged observation file: {time.perf_counter()-tic:0.2f} seconds")  
    return df['location'],df['sleep'],df['physiological']



def load_flags(zip_file,output_path='Data/pkl/',verbose=True):
    '''
    Given an thim zip_file this function compares the date of any 
    already existing processed data and if it finds none it will parse 
    the Flags.csv file and store the processed data locally under 
    the output_path
        
    INPUT
    =====
    zip_file = a string directing to a thim zip file 
    output_path = a string direction to local path
    verbose = indicating whether to print out steps timming
    
    RETURNS
    ======
    a Pandas Dataframe of project specific flags 
    ''' 
    if verbose:tic = time.perf_counter()
    _, name, _ = file_parts(zip_file)
    date,_ = file_date(zip_file)
    date = date.strftime("%Y%m%d")
    pickle_file = f"{output_path}{name}_{date}_flags.pkl"
    df = load_pickle(pickle_file)
    df = parse_flags(df,zip_file,pickle_file,verbose)
    if verbose:        
        print(
            f"Elapsed time to load flags {name} file: {time.perf_counter()-tic:0.2f} seconds")
    return df 


def parse_flags(_df,zip_file,pickle_file,verbose):
    '''
    checks if flags file already exists and if not parses the 
    one contained within the zip_file which is then stored as a pickle locally 
    
    INPUT
    =====
    df = Either an already preprocessed dictionary or a boolean
    zip_file = a string directing to a zip file 
    pickle_file = a string direction to store the processed data
    verbose = indicating whether to print out steps timming
    
    RETURNS
    ======
    a dataFrame containing flags events in tabular form
    ''' 
    date,_ = file_date(zip_file)
    _, name, _ = file_parts(zip_file)
    if isinstance(_df, bool):
        if verbose:tic = time.perf_counter()
        _zip = zipfile.ZipFile(zip_file)
        _df = pd.read_csv(_zip.open('Flags.csv'))
        _type = pd.read_csv(_zip.open('Flag-type.csv'))
        _cat = pd.read_csv(_zip.open('Flag-category.csv'))
        _val = pd.read_csv(_zip.open('FlagValidations.csv'))
        _df = pd.merge(_df, _val, how='outer', on=None, 
                left_on="flagId", right_on="flag",
                suffixes=('_df', '_val'), copy=True)
        _df.category = pd.Categorical(_df.category)
        mapping = get_mapper(_cat.display.values,_cat.code.values)
        _df.category = _df.category.cat.rename_categories(mapping)
        idx = find_duplicate(_type.display.values)[0]
        if any(idx):
            values = list(_type.code.values[idx])
            key = list(np.where(idx)[0])
            for key,val in dict(zip(key[0:-1],values[0:-1])).items():
                _df.type[_df.type == val] = values[-1]
                _type = _type.drop(key)
        mapping = get_mapper(_type.display.values,_type.code.values)
        _df.type  = pd.Categorical(_df.type)
        _df.type  = _df.type.cat.rename_categories(mapping)
        _df['project'] = name
        _df['project'] = pd.Categorical(_df['project'])
        _df.rename(columns={'subject_df':'subject'}, inplace=True)
        _df = pid_mapping(_df, _zip)
        if verbose:        
            print(f"Processing flags took : {time.perf_counter()-tic:0.2f} seconds")
        save_pickle(pickle_file, _df)
    return _df 


def merge_flags(files,output_path='Data/pkl/',verbose=True):
    '''
    Given a list of paths to zip files this function will iteratively 
    pre-process all Flags.csv files and merge them to 
    one unified Dataframe in tabular form
            
    INPUT
    =====
    files = a list of file paths to thim zip files 
    output_path = a place where already pre-processed data can be 
    either stored or retrived 
    verbose = indicating whether to print out steps timming
    
    RETURNS
    ======
    a dataFrame containing flags events in tabular form
    '''     
    if verbose:tic = time.perf_counter()
    pickle_file = f"{output_path}merged_flags.pkl"
    df = load_pickle(pickle_file)
    if isinstance(df, bool):
        df = ['0']*len(files)
        for idx,fid in enumerate(files):
            df[idx] = load_flags(fid,output_path=output_path, verbose=verbose)

        df = pd.concat([df[ii] for ii in range(len(files))],ignore_index=True) 
        
        df.subject = pd.Categorical(df.subject)

        save_pickle(pickle_file, df)
    
    if verbose:        
        print(f"Elapsed time to load merged flags file: {time.perf_counter()-tic:0.2f} seconds")  
    return df


def load_wellbeing(zip_file,output_path='Data/pkl/',verbose=True):
    '''
    Given an thim zip_file this function compares the date of any 
    already existing processed data and if it finds none it will parse 
    the QuestionnaireResponses.csv file and store the processed data 
    locally under the output_path
        
    INPUT
    =====
    zip_file = a string directing to a thim zip file 
    output_path = a string direction to local path
    verbose = indicating whether to print out steps timming
    
    RETURNS
    ======
    a dataFrame containing wellbeing events in tabular form

    ''' 
    if verbose:tic = time.perf_counter()
    _, name, _ = file_parts(zip_file)
    date,_ = file_date(zip_file)
    date = date.strftime("%Y%m%d")
    pickle_file = f"{output_path}{name}_{date}_well.pkl"
    df = load_pickle(pickle_file)
    df = parse_wellbeing(df,zip_file,pickle_file,verbose)
    if verbose:        
        print(
            f"Elapsed time to load wellbeing {name} file: {time.perf_counter()-tic:0.2f} seconds")
    return df


def parse_wellbeing(_df,zip_file,pickle_file,verbose=True):
    '''
    checks if wellbeing file already exists and if not parses the 
    one contained within the zip_file which is then stored as a pickle locally 
    
    INPUT
    =====
    _df = Either an already preprocessed dictionary or a boolean
    zip_file = a string directing to a zip file 
    pickle_file = a string direction to store the processed data
    verbose = indicating whether to print out steps timming
    
    RETURNS
    ======
    a dataFrame containing flags events in tabular form
    ''' 
    date,_ = file_date(zip_file)
    _, name, _ = file_parts(zip_file)
    if isinstance(_df, bool):
        if verbose:tic = time.perf_counter()
        _zip = zipfile.ZipFile(zip_file)
        _df = pd.read_csv(_zip.open('QuestionnaireResponses.csv'))
        _df['datetimeAnswered'] = pd.to_datetime(_df['datetimeAnswered'])
        _df = _df.sort_values(by=['datetimeAnswered'])
        _df = _df.drop(columns=["questionnaire", "datetimeReceived"])
        _df.question, questions = pd.factorize(_df.question)
        _df = _df.drop_duplicates()
        _df = pid_mapping(_df, _zip)
        _df = _df.dropna().reset_index(drop=True)
        index = pd.MultiIndex.from_tuples(zip(_df.subject, _df.datetimeAnswered,
                _df.question), names=['subject', 'datetimeAnswered', 'question'])
        _df = pd.DataFrame(_df.answer.values, index=index,columns=['answer']).unstack()
        _df.columns = _df.columns.droplevel()
        _df.columns = questions
        _df = _df.reset_index()
        _df['project'] = name
        _df['project'] = pd.Categorical(_df['project'])
        for col in _df.columns:
            _df[col] = pd.Categorical(_df[col])

        if verbose:        
            print(f"Processing flags took : {time.perf_counter()-tic:0.2f} seconds")
        save_pickle(pickle_file, _df)
    return _df 


def merge_wellbeing(files,output_path='Data/pkl/',verbose=True):
    '''
    Given a list of paths to zip files this function will iteratively 
    pre-process all QuestionnaireResponses.csv files and merge them to 
    one unified Dataframe in tabular form
            
    INPUT
    =====
    files = a list of file paths to thim zip files 
    output_path = a place where already pre-processed data can be 
    either stored or retrived 
    verbose = indicating whether to print out steps timming
    
    RETURNS
    ======
    a dataFrame containing flags events in tabular form
    '''     
    if verbose:tic = time.perf_counter()
    pickle_file = f"{output_path}merged_well.pkl"
    df = load_pickle(pickle_file)
    if isinstance(df, bool):
        df = ['0']*len(files)
        for idx,fid in enumerate(files):
            df[idx] = load_wellbeing(fid,output_path=output_path, verbose=verbose)

        df = pd.concat([df[ii] for ii in range(len(files))],ignore_index=True) 
        
        df.subject = pd.Categorical(df.subject)

        save_pickle(pickle_file, df)
    
    if verbose:        
        print(f"Elapsed time to load merged well-being file: {time.perf_counter()-tic:0.2f} seconds")  
    return df

def disp(df):
    '''
    Used in notebboks to temporarily allow pandas to show all columns and rows 
    of the dataframe df - use with care!!!
    '''
    [m,n]= df.shape
    pd.set_option('display.max_rows', m,'display.max_columns',n)
    display(df)
    pd.reset_option('display.max_rows')
    
    
def tuplizer(x):
    '''
    convert cell contents to tupel for the odd cases where a cell in a dataframe
    contains either an array or a list 
    '''
    return tuple(x) if isinstance(x, (np.ndarray, list)) else x

  
  

def free_memory():
    '''
    gives info regarding the amount of free memory in linux systems 
    '''
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
    '''
    mimic's Matlabs ismemeber function (should be removed in later versions)
    '''
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return np.array([bind.get(itm, None) for itm in a])  
  
  
def nnz(a):
    '''
    shortcut to count non zeros 
    '''
    return np.count_nonzero(a)


def rename_categories(idx,mapping):
    '''
    remapping numpy arrays efficiently 
    '''
    keys = np.array(list(mapping.keys()))
    values = np.array(list(mapping.values()))
    mapping_array = np.zeros(keys.max()+1, dtype=values.dtype)
    mapping_array[keys] = values
    return mapping_array[idx]

def list_to_csv(my_list,file_name,header):
    '''
    saving out lists to files quickely
    '''
    np.savetxt(file_name, my_list, delimiter=",", fmt='%s', header=header)

def file_parts(file):
    '''
    splits a string file to path, file_name, file_type
    '''
    head_tail = os.path.split(file)
    path = head_tail[0]
    file_name, file_type = head_tail[1].split('.')
    return path, file_name, file_type


def find_duplicate(L):
    '''
    identifies duplicates in a list and returns their index
    '''   
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
    '''
    returns a files time stamp as a datetime object
    '''
    stat = os.stat(path_to_file)
    date = datetime.datetime.fromtimestamp(stat.st_mtime)
    return date, stat


def save_pickle(pickle_file, data):
    '''
    Save data as pickle requires full path 
    '''
    if not os.path.exists(pickle_file):
            filepath, _, _ = file_parts(pickle_file)
            Path(filepath).mkdir(parents=True, exist_ok=True)
    output_pickle = open(pickle_file, "wb")
    cPickle.dump(data, output_pickle)
    output_pickle.close()

def load_pickle(pickle_file):
    '''
    loads data from a pickle file 
    '''
    if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0:
        input_pickle = open(pickle_file, 'rb')
        data = cPickle.load(input_pickle)
        input_pickle.close()
        return data
    return False


def get_mapper(value,key):
    '''
    creates a mapping dictionary 
    '''
    return pd.Series(value,index=key).to_dict()
