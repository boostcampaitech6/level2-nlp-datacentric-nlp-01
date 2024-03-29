
from g2pk import G2p
from tqdm import tqdm
from py_hanspell.hanspell import spell_checker
import re
import os 
from settings import *
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

def noise_sentence_filter(df):
    g2p = G2p()
    noise_list = []
    for idx, sen in tqdm(enumerate(df['text']), total = len(df), desc = 'prescriptive g2p'):
        after_g2p = g2p(sen)
        if sen == after_g2p:
            noise_list.append(idx)

        
    return noise_list


def noise_sentence_filter_descriptive(df):
    g2p = G2p()
    noise_list = []
    for idx, sen in tqdm(enumerate(df['text']), total = len(df), desc = 'descriptive g2p'):
        after_g2p = g2p(sen, descriptive = True)
        if sen == after_g2p:
            noise_list.append(idx)
        
    return noise_list

def get_full_noise_list(df):
    pre_list = noise_sentence_filter(df)
    des_list = noise_sentence_filter_descriptive(df)
    full_list = list(set(pre_list + des_list))
    print(f"prescriptive noise list length: {len(pre_list)}")
    print(f"descriptive noise list length: {len(des_list)}")
    print(f"concat full noise list length: {len(full_list)}")
    return full_list


def spell_check(df):
    for idx, row in tqdm(df.iterrows(), desc = 'Spell checking...', total = len(df)):
        text = row['text']
        res = spell_checker.check(text).as_dict()
        clean_res = re.sub('…|(\.\.\.)',' ',res['checked']).rstrip()
        df.at[idx,'text'] = clean_res
    return df

def load_data(file_name, g2p_type = True):
    if not os.path.exists(os.path.join(DATA_DIR, f"{file_name}.csv")):
        cleaned = pd.read_csv(os.path.join(DATA_DIR, 'cleaned_data.csv'))
        if g2p_type:
            full_noise = get_full_noise_list(cleaned)
            denoise_list = [i for i in cleaned.index.tolist() if i not in full_noise]
            cleaned = cleaned.iloc[denoise_list]
        cleaned = spell_check(cleaned)
        cleaned.to_csv(os.path.join(DATA_DIR, f"{file_name}.csv"), index = False)
    df = pd.read_csv(os.path.join(DATA_DIR, f"{file_name}.csv"))
    return df


def plot_scatter2D(dataframe, sampling=10_000, save=False, fname='scatter_plot'):
    color_map = ['#8B0000', '#FF0000', '#BDB76B', '#7CFC00', '#008080', '#4169E1', '#FF69B4']
    if sampling:
        idx = np.random.randint(0, len(dataframe), sampling)    
        samples = dataframe.iloc[idx]
    else:
        samples = dataframe.copy()
    
    x = samples.iloc[:, 0]
    y = samples.iloc[:, 1]
    label = samples.loc[:, 'label']
    
    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()
    ax.scatter(x, y, color=label.apply(lambda x: color_map[x]), alpha=0.5)
    
    if save:
        plt.savefig(os.path.join(FIG_DIR, f'{fname}.png'), dpi=200)
    plt.show()
    
def plot_scatter3D(dataframe, sampling=10_000, save=False, fname='scatter_plot'):
    color_map = ['#8B0000', '#FF0000', '#BDB76B', '#7CFC00', '#008080', '#4169E1', '#FF69B4']
    if sampling:
        idx = np.random.randint(0, len(dataframe), sampling)    
        samples = dataframe.iloc[idx]
    else:
        samples = dataframe.copy()
    
    x = samples.iloc[:, 0]
    y = samples.iloc[:, 1]
    z = samples.iloc[:, 2]
    label = samples.loc[:, 'label']
    
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color=label.apply(lambda x: color_map[x]), alpha=0.5)
    
    if save:
        plt.savefig(os.path.join(FIG_DIR, f'{fname}.png'), dpi=200)
    plt.show()