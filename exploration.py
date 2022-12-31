import pandas as pd
import os
from multiprocessing import Pool

def read_csv(filename: str):
    return pd.read_csv(filename)

def combine_df(path: str):
    pt_type = path.split('/')[2].split('_')[1]
    print(f'processing {pt_type} csvs. . .')
    files = os.listdir(path)
    file_list = [path+filename for filename in files if filename.split('.')[1]=='csv']
    print(f'number of {pt_type} patient csvs:', len(file_list))
    with Pool(processes=8) as pool:
        df_list = pool.map(read_csv, file_list)
        combined_df = pd.concat(df_list, ignore_index=True)
        print('complete')
        return combined_df

def generate_statistics(df):
    statistics = {
        'features': df.columns,
        'min': df.min().reset_index(drop=True),
        'max': df.max().reset_index(drop=True),
        'mean': df.mean().reset_index(drop=True),
        'median': df.median().reset_index(drop=True),
        'std': df.std().reset_index(drop=True),
        'missing': df.isna().sum().reset_index(drop=True)
    }
    return pd.DataFrame(statistics)

if __name__ == '__main__':
    aki_path = './data/dataset_AKI_prediction/dataset_train/'
    sepsis_path = './data/dataset_sepsis_prediction/dataset_train/'
    
    sepsis_df = combine_df(sepsis_path)
    aki_df = combine_df(aki_path)

    print('size of combined sepsis df:', sepsis_df.shape)
    print('size of combined aki df:', aki_df.shape)

    stats_sepsis = generate_statistics(sepsis_df)
    stats_sepsis
    stats_aki = generate_statistics(aki_df)

    print(stats_aki)
    print(stats_sepsis)

    stats_aki.to_csv('data/aki_features.csv', index=False)
    stats_sepsis.to_csv('data/sepsis_features.csv', index=False)