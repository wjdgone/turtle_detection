import cv2 
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold


def resize_bg(): 

    in_dir = '../data/raw/bg'
    save_dir = '../data/raw/bg_resized'

    s = 180

    if not os.path.isdir(save_dir): 
        os.mkdir(save_dir)
        print(f'directory created at {save_dir}')

    in_paths = [os.path.join(in_dir, name) for name in os.listdir(in_dir)]

    for in_path in in_paths: 
        name = os.path.basename(in_path).split('.')[0]
        im = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        out = cv2.resize(im, (s,s))
        cv2.imwrite(os.path.join(save_dir, name+'.jpg'), out)
            
            
def generate_neg_patch(x_min, x_max, y_min, y_max, size_bg_x, size_tar_x, size_bg_y, size_tar_y):
    x_neg_min = np.random.randint(0, size_bg_x-size_tar_x)
    y_neg_min = np.random.randint(0, size_bg_y-size_tar_y)
    x_neg_max = x_neg_min + size_tar_x
    y_neg_max = y_neg_min + size_tar_y

    # check overlap with given coords
    x_min_overlap = (x_neg_min >= x_min) and (x_neg_min <= x_max)
    x_max_overlap = (x_neg_max >= x_min) and (x_neg_max <= x_max)
    y_min_overlap = (y_neg_min >= y_min) and (y_neg_min <= y_max)
    y_max_overlap = (y_neg_max >= y_min) and (y_neg_max <= y_max)

    # regenerate coords if there is overlap
    while x_min_overlap or x_max_overlap or y_min_overlap or y_max_overlap: 
        x_neg_min = np.random.randint(0, size_bg_x-size_tar_x)
        y_neg_min = np.random.randint(0, size_bg_y-size_tar_y)
        x_neg_max = x_neg_min + size_tar_x
        y_neg_max = y_neg_min + size_tar_y

        # check overlap with given coords
        x_min_overlap = (x_neg_min >= x_min) and (x_neg_min <= x_max)
        x_max_overlap = (x_neg_max >= x_min) and (x_neg_max <= x_max)
        y_min_overlap = (y_neg_min >= y_min) and (y_neg_min <= y_max)
        y_max_overlap = (y_neg_max >= y_min) and (y_neg_max <= y_max)

    return x_neg_min, x_neg_max, y_neg_min, y_neg_max


def create_ds(): 
    ''' create dataset by overlaying target on bg and recording bounding box labels. '''

    # each bg gives n_pos_per_bg number of positive samples (and same number of negative samples)
    n_pos_per_bg = 10
    
    # ratio of pos to neg patches (neg/pos)
    n_neg_per_bg = 1

    # augment
    augment = False
    augment_translation = True

    out_dir = f'../data/patches/{ds}'

    x_dir = os.path.join(out_dir, 'x')

    create_dir(x_dir)

    # get image paths
    im_tar_paths = [os.path.join(in_tar_dir, name) for name in os.listdir(in_tar_dir)]
    im_bg_paths = [os.path.join(in_bg_dir, name) for name in os.listdir(in_bg_dir)]

    # initialize labels csv
    labels = []

    for im_tar_path in im_tar_paths: 
        print('-'*10)
        print('starting:', im_tar_path)
        for im_bg_path in im_bg_paths: 

            for idx in range(n_pos_per_bg):
                # print(im_tar_path, im_bg_path)

                im_tar = cv2.imread(im_tar_path, cv2.IMREAD_UNCHANGED)
                im_bg = cv2.imread(im_bg_path, cv2.IMREAD_UNCHANGED)
                im_bg = np.concatenate((im_bg, np.ones((im_bg.shape[0], im_bg.shape[0], 1), dtype=np.uint8)*255), axis=-1)

                im_tar = im_tar.astype(np.float64)/255.0
                im_bg = im_bg.astype(np.float64)/255.0

                im = im_bg.copy()

                if augment: 
                    # random rotation of target
                    if np.random.choice([0,1,2,3]) != 0: 
                        im_tar = cv2.rotate(im_tar, np.random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]))

                    # random flip of target (horizontally only)
                    if np.random.randint(0,3) in (0,1): 
                        im_tar = cv2.flip(im_tar, np.random.choice([0,1]))

                # random coordinates for placing target on background
                size_tar_y, size_tar_x = im_tar.shape[0], im_tar.shape[1]
                size_bg_y, size_bg_x = im_bg.shape[0], im_bg.shape[1]
                y_min = np.random.randint(0, size_bg_y-size_tar_y)
                x_min = np.random.randint(0, size_bg_x-size_tar_x)
                y_max = y_min + size_tar_y
                x_max = x_min + size_tar_x

                # overlay images with alpha channel
                alpha_s = im_tar[:, :, 3]
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    im[y_min:y_max, x_min:x_max, c] = (alpha_s * im_tar[:, :, c] + alpha_l * im_bg[y_min:y_max, x_min:x_max, c])

                # save image
                im = (im*255.0).astype(np.uint8)[..., :3]
                tar_name = os.path.basename(im_tar_path).split('.')[0]
                bg_name = os.path.basename(im_bg_path).split('.')[0]
                out_name = f'{tar_name}_{bg_name}_{idx}.png'
                print(out_name)
                im_path = os.path.join(x_dir, out_name)
                cv2.imwrite(im_path, im)
                
                # save positive labels
                if augment_translation: 
                    x_trans_aug = np.random.randint(-size_tar_x//2, size_tar_x//2)
                    y_trans_aug = np.random.randint(-size_tar_y//2, size_tar_y//2)
                    x_min += x_trans_aug
                    x_max += x_trans_aug
                    y_min += y_trans_aug
                    y_max += y_trans_aug

                    # enforce coordinates to be within the image
                    if x_min < 0: 
                        x_min, x_max = 0, size_tar_x
                    if x_max >= size_bg_x: 
                        x_min, x_max = size_bg_x-size_tar_x-1, size_bg_x-1
                    if y_min < 0: 
                        y_min, y_max = 0, size_tar_y
                    if y_max >= size_bg_y: 
                        y_min, y_max = size_bg_y-size_tar_y-1, size_bg_y-1
                    
                labels.append([out_name, x_min, x_max, y_min, y_max, 1])

                # save negative patch label(s)
                for _ in range(n_neg_per_bg): 
                    x_neg_min, x_neg_max, y_neg_min, y_neg_max = generate_neg_patch(x_min, x_max, y_min, y_max, size_bg_x, size_tar_x, size_bg_y, size_tar_y)
                    labels.append([out_name, x_neg_min, x_neg_max, y_neg_min, y_neg_max, 0])
            
    labels_df = pd.DataFrame(labels, columns=['name', 'x_min', 'x_max', 'y_min', 'y_max', 'class'])
    labels_df.to_csv(os.path.join(out_dir, 'labels.csv'), index=None)


def create_dir(*args): 

    for d in args:
        if not os.path.isdir(d): 
            os.makedirs(d)
            print(f'new directory created at {d}')


def train_test_split(): 

    in_path = f'../data/patches/{ds}/labels.csv'
    labels = pd.read_csv(in_path)

    pos_labels = labels[labels['class']==1]
    neg_labels = labels[labels['class']==0]

    kf_pos, kf_neg = KFold(n_splits=5, shuffle=True), KFold(n_splits=5, shuffle=True)
    kf_pos.get_n_splits(pos_labels)
    kf_neg.get_n_splits(neg_labels)

    save_dir = os.path.split(in_path)[0]
    for fold_idx in range(5): 
        train_pos_idx, test_pos_idx = next(kf_pos.split(pos_labels))
        train_neg_idx, test_neg_idx = next(kf_neg.split(neg_labels))

        train_pos = pos_labels.iloc[train_pos_idx]
        train_neg = neg_labels.iloc[train_neg_idx]
        test_pos = pos_labels.iloc[test_pos_idx]
        test_neg = neg_labels.iloc[test_neg_idx]

        train_ds = pd.concat((train_pos, train_neg))
        test_ds = pd.concat((test_pos, test_neg))

        save_path_train = os.path.join(save_dir, f'labels_fold{fold_idx+1}_train.csv')
        save_path_test = os.path.join(save_dir, f'labels_fold{fold_idx+1}_test.csv')
        train_ds.to_csv(save_path_train, index=None)
        test_ds.to_csv(save_path_test, index=None)


if __name__ == '__main__': 


    # resize_bg()

    ## augmentation, overlay, and generate positive and negative patch labels
    ds = 'ds15'
    
    # get in/out directories
    in_tar_dir = '../data/raw/tar_cropped/28x28' 
    in_bg_dir = '../data/raw/bg_ds15'
    create_ds()

    train_test_split()