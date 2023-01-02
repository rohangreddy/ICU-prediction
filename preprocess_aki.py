import os
from multiprocessing import Pool
from collections import deque
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from functools import partial

def label_aki(path: str):
    """
    Labels patient data based on whether they develop AKI during their hospital stay
    
    Onset of AKI is indicated by two conditions:

    Condition 1 -> An increase of creatinine of 0.3 within a 48 hour window
    Condition 2 -> An increase in creatinine of 1.5 * baseline value, where 
                   baseline is defined as the lowest value seen in the prior 168 hours 
    Returns: 
        (patient_ID, hour of aki onset) if the patient developed AKI, otherwise return patient_id 
    """
    # Read patient csv
    df = pd.read_csv(path)
    id = path.split('/')[-1].split('.')[0]
    # filter out non-NaN creatinine values
    df = df[df['Creatinine'].notna()][['Creatinine', 'Hours_Since_Admission']]
    # Initialize hash table for fast lookups of creatinine levels based on hour
    levels = {}
    # Initialize tuple to keep track of lowest value in 48hr window
    lowest = (0,0)
    # Initialize tuple to keep track of lowest value in 168hr window
    lowest_7 = (0,0)
    # Initialize deque for 48hr window
    window_48 = deque()
    # Initialize deque for 168hr window
    window_7 = deque()
    # Loop through patient's valid (non-NaN) creatinine datapoints
    for i in range(len(df.index)):
        # find current creatinine level
        level = df.loc[df.index[i], 'Creatinine']
        # find current time point
        hours = df.loc[df.index[i], 'Hours_Since_Admission']
        # add to hash table
        levels[hours] = level
        # remove time values from 48 hour window if outside of 48 hours of current timepoint 
        while window_48 and window_48[0] < hours - 48:
            window_48.popleft()
        # remove time values from 168 hour window if outside of 168 hours of current timepoint 
        while window_7 and window_7[0] < hours - 168:
            window_7.popleft()
        # Initialize tuples for lowest values to current value if not initialized already
        if not lowest[0]:
            lowest = (level,hours)
        if not lowest_7[0]:
            lowest_7 = (level,hours)
        # Append current timepoint to the windows
        window_48.append(hours)
        window_7.append(hours)
        # update lowest value seen so far in the 48-hour window
        future = list(window_48)
        curr_lowest = (levels[future[0]], future[0])
        for t in future:
            curr = levels[t]
            if curr < curr_lowest[0]:
                curr_lowest = (curr, t)
        lowest = curr_lowest
        # update lowest value seen so far in the 168-hour window
        future = list(window_7)
        curr_lowest = (levels[future[0]], future[0])
        for t in future:
            curr = levels[t]
            if curr < curr_lowest[0]:
                curr_lowest = (curr, t)
        lowest_7 = curr_lowest
        # check for AKI using condition 1 and condition 2
        if level >= lowest[0] + 0.3 or level >= lowest_7[0] * 1.5:
            return (id, hours)
    return (id, -1)

def create_observation_window(path, test=False):
    """
    Creates observation windows of 24, 48, and 72 hours with
    accompanying labels depending on whether the patient develops
    AKI 24 hours following the observation window.

    The data in each observation window will be aggregated such that each
    individual row in the matrix represents the aggregation of the patient's
    data during the observation window. 

    For training data, any patient that develops AKI during the observation window will be excluded.
    For testing data, this restriction will not apply. 

    Returns:
        [ 
            [observation_window_24hrs (Pandas Series), label, ID], 
            [observation_window_48hrs (Pandas Series), label, ID], 
            [observation_window_72hrs (Pandas Series), label, ID] 
        ]
    """
    # Read in patient from path
    pt = pd.read_csv(path)
    # Initialize return value of 24 hr, 48 hr, 72 hr observation windows
    ans = [[], [], []]
    # Get the ID and hour of onset of AKI
    id, onset = label_aki(path)
    # Find the starting hour of the patient's hospital stay
    start_hour = pt.iloc[pt.index.min(), -1]
    # Find the ending index of the smaller of the first 24 hours of the patient's stay or max index
    end_idx = min(pt.index.min() + 23, pt.index.max())
    # Find the ending hour of the current observation window
    end_hour = pt.iloc[end_idx, -1]
    # Initialize a count for the 3 observation windows
    window = 0
    # Iterate 3 times for each of the observation windows while onset of AKI is outside the current window
    while window < 3 and (test or not (start_hour <= onset <= end_hour)):
        # Initialize label with default value of 0 corresponding to control
        label = 0
        # Modify label to 1 for case if onset of AKI is within 24 hours after the current observation window
        if end_hour < onset <= end_hour+24:
            label = 1
        # Add [aggregated data during observation window, label, id] to the appropriate index in the return value
        ans[window] = [pt.iloc[:end_idx+1,:-1].mean(), label, id]
        # Expand the current observation window by 24 hours for the next iteration
        end_idx = min(end_idx+23, pt.index.max())
        end_hour = pt.iloc[end_idx, -1]
        window += 1
    return ans

def impute_median(data):
    copy = data.copy()
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imputed = imp.fit_transform(copy)
    return imputed

def preprocess_observation_windows(path, test=False):
    """
    Calls create_observation_window on all patient CSV files using Pool.
    Then aggregates all observation window and label results. Missing values are imputed 
    using the median value.

    Returns:
        numpy arrays of observation windows and labels  
    """
    
    print('running preprocess_observation_windows . . .')

    files = os.listdir(path)
    file_list = [path+filename for filename in files if filename.split('.')[1]=='csv']
    
    with Pool(processes=8) as pool:
        mapfunc = partial(create_observation_window, test=test)
        windows = pool.map(mapfunc, file_list)

    pt_windows = [[patient[i][0] for patient in windows if patient[i]] for i in range(3)]
    pt_labels = [[patient[i][1] for patient in windows if patient[i]] for i in range(3)]
    pt_ids = [[patient[i][2] for patient in windows if patient[i]] for i in range(3)]
    
    values = []
    name = 'test' if test else 'train'
    
    for i in range(3):
        window = pd.concat(pt_windows[i], axis=1).T if pt_windows[i] else []
        window_hours = str(24 + 24*i)
        window.to_csv(f'data/{name}_window_{window_hours}.csv')
        window = impute_median(window)
        values.append(window)
        labels = {'id': pt_ids[i], 'label': pt_labels[i]} if pt_labels and pt_ids else {}
        labels = pd.DataFrame(labels)
        labels.to_csv(f'data/{name}_labels_{window_hours}.csv')
        labels = labels['label'].to_numpy()
        values.append(labels)
        
    return values

if __name__ == '__main__':
    train_path = 'data/dataset_AKI_prediction/dataset_train/'
    test_path = 'data/dataset_AKI_prediction/dataset_test/'
    w_24, l_24, w_48, l_48, w_72, l_72 = preprocess_observation_windows(train_path)
    w_24, l_24, w_48, l_48, w_72, l_72 = preprocess_observation_windows(test_path, test=True)

    