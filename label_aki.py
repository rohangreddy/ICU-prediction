import pandas as pd
import os
from multiprocessing import Pool
from collections import deque

def label_aki(path):
    """
    Labels patient data based on whether they develop AKI during their hospital stay
    
    Onset of AKI is indicated by two conditions:

    Condition 1 -> An increase of creatinine of 0.3 within a 48 hour window
    Condition 2 -> An increase in creatinine of 1.5 * baseline value, where 
                   baseline is defined as the lowest value seen in the prior 168 hours 
    Params: 
        data -> concatenated aki patient data (including patient ids as a column 'ID')
        ids -> list of patient ids present in the data
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

def generate_aki_labels(path: str):
    files = os.listdir(path)
    file_list = [path+filename for filename in files if filename.split('.')[1]=='csv']
    with Pool(processes=8) as pool:
        labels_list = pool.map(label_aki, file_list)
        aki_labels = {'id': [x[0] for x in labels_list],
                      'aki': [0 if x[1] == -1 else 1 for x in labels_list]}
        return pd.DataFrame(aki_labels).sort_values('id').reset_index(drop=True)

if __name__ == '__main__':
    aki_train_path = './data/dataset_AKI_prediction/dataset_train/'
    aki_test_path = './data/dataset_AKI_prediction/dataset_test/'
    sepsis_path = './data/dataset_sepsis_prediction/dataset_train/'
    
    aki_train_labels = generate_aki_labels(aki_train_path)
    aki_test_labels = generate_aki_labels(aki_test_path)
    
    aki_train_labels.to_csv('data/aki_train_labels.csv', index=False)
    aki_test_labels.to_csv('data/aki_test_labels.csv', index=False)

    