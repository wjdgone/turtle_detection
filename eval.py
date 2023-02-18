# additional evaluation for res.csv
import pandas as pd
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import os 
from preprocess_data import create_dir


def find_mid_scores(res): 
    pivot_idx = len(res)-1
    thres = res['thres'].iloc[0]
    for idx, data in res.iterrows(): 
        cur_score = data['score']
        if cur_score < thres: 
            pivot_idx = idx
            break

    low, high = pivot_idx-4, pivot_idx+5
    if low < 0: 
        low, high = 0, 9
    elif high > len(res): 
        low, high = len(res)-9, len(res)

    print(low, high)
    return res.iloc[low:high]


def plot_9x9(res, mode, label): 
    ''' mode from {top, bot, mid}'''

    print(mode)

    if mode=='top': 
        res = res.sort_values(by=['score'], ascending=False, ignore_index=True)
    elif mode=='bot': 
        res = res.sort_values(by=['score'], ascending=True, ignore_index=True)
    elif mode=='mid': 
        res = find_mid_scores(res.sort_values(by=['score'], ascending=False, ignore_index=True))

    plt.clf(), plt.close()
    plt.figure(figsize=(8,8))
    for i in range(9): 
        try: 
            data = res.iloc[i]
        except: 
            data = res.iloc[6]
        # input(data)
        im_path = os.path.join(im_dir, data['name'])
        im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        patch = im[data['y1']:data['y2'], data['x1']:data['x2']]
        plt.subplot(3,3,i+1), plt.imshow(patch), plt.title(f"({label}) score={round(data['score'])}")
    # plt.show()
    plt.savefig(os.path.join(save_dir, f'{label}_{mode}.png'))


def show_by_scores(): 

    res_pos = res_all[res_all['clbl']==1]
    res_neg = res_all[res_all['clbl']==0]

    # show images
    for res, label in zip([res_all, res_pos, res_neg], ['all', 'pos', 'neg']):
        for mode in ['top', 'mid', 'bot']: 
            plot_9x9(res, mode=mode, label=label)

    # show histogram of score distribution
    plt.clf(), plt.close()
    plt.figure(figsize=(10,7))
    x_min = res_all['score'].min()
    x_max = res_all['score'].max()
    plt.subplot(2,1,1), plt.hist(np.ravel(res_pos['score']), bins=50), plt.xlim([x_min, x_max]), plt.title('score distribution of pos samples')
    plt.subplot(2,1,2), plt.hist(np.ravel(res_neg['score']), bins=50), plt.xlim([x_min, x_max]), plt.title('score distribution of neg samples')
    # plt.show()
    plt.savefig(os.path.join(save_dir, f'score_distribution.png'))
    

if __name__ == '__main__': 

    im_dir = '../data/patches/ds14/x'
    root_dir = '../results/ds14/model_3 (epoch 25)/fold_0/test/' 
    save_dir = os.path.join(root_dir, 'score_distrib_seperate')
    create_dir(save_dir)

    csv_path = os.path.join(root_dir, 'res.csv')
    res_all = pd.read_csv(csv_path)

    show_by_scores()