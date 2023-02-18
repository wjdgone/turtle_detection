import os 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import pandas as pd
from data import PatchLoader
from preprocess_data import create_dir


class BaseModel: 

    def __init__(self, device, data_dir, res_dir, fold_idx): 
        self.device = device
        self.data_dir = data_dir 
        self.res_root_dir = res_dir
        create_dir(self.res_root_dir)

        self.fold_idx = fold_idx

        self.train_data, self.test_data = None, None
        self.dataset = None
        self.patch_size = 28
        # self.stride = 14

        # self.score_thres = 0.7 # threshold for detection score (softmax thres for nn)
        # self.overlap_thres = 0.5 # threshold for a patch from a scene to be considered a positive label
        self.nms_thres = 0.3

        self.model_type = None

    def get_iou(self, bbox1, bbox2):

        x_left = np.max([bbox1['x_min'], bbox2['x_min']])
        y_top = np.max([bbox1['y_min'], bbox2['y_min']])
        x_right = np.min([bbox1['x_max'], bbox2['x_max']])
        y_bottom = np.min([bbox1['y_max'], bbox2['y_max']])

        if (x_right < x_left) or (y_bottom < y_top):
            return 0.0

        # intersection of two bounding boxes
        intersection = (x_right - x_left) * (y_bottom - y_top)

        # union of two bounding boses
        bbox1_area = (bbox1['x_max'] - bbox1['x_min']) * (bbox1['y_max'] - bbox1['y_min'])
        bbox2_area = (bbox2['x_max'] - bbox2['x_min']) * (bbox2['y_max'] - bbox2['y_min'])
        union = bbox1_area + bbox2_area - intersection

        return intersection / union
    
    
    def compute_metrics(self): 

        # confusion matrix
        ys = np.array(self.clbls)
        zs = np.array(self.cpreds)

        tp = (ys*zs).sum()
        tn = np.where((ys+zs)==0, 1, 0).sum()
        fp = np.where((zs-ys)==1, 1, 0).sum()
        fn = np.where((ys-zs)==1, 1, 0).sum()

        self.acc = (tp+tn) / (tp+tn+fp+fn)
        self.prc = tp / (tp+fp)
        self.rcl = tp / (tp+fn)
        self.f1s = (self.prc*self.rcl*2) / (self.prc+self.rcl)

        # PR curve
        p,r,_= precision_recall_curve(self.clbls, self.cpreds, pos_label=1)
        disp = PrecisionRecallDisplay(precision=p, recall=r)
        disp.plot()
        plt.savefig(os.path.join(self.res_dir, 'pr_curve.png'))


    def save_res_csv(self, **kwargs): 

        res_data = {}
        for kw in kwargs: 
            res_data[kw] = kwargs[kw]

        self.res_csv_path = os.path.join(self.res_dir, 'res.csv')

        self.res_csv = pd.DataFrame(res_data)
        self.res_csv.to_csv(self.res_csv_path, index=None)


    def test(self, stride, overlap_thres): 

        # bayesian opt
        self.stride = int(stride)
        # self.score_thres = score_thres # threshold for detection score (softmax thres for nn)
        self.overlap_thres = overlap_thres # threshold for a patch from a scene to be considered a positive label
        # self.nms_thres = nms_thres

        self.res_dir = os.path.join(self.res_root_dir, f'fold_{self.fold_idx}', self.data_mode)
        self.bbox_preds_dir = os.path.join(self.res_dir, 'bbox_preds')
        self.score_maps_dir = os.path.join(self.res_dir, 'score_maps')
        self.score_distrib_dir = os.path.join(self.res_dir, 'score_distrib_dir')
        create_dir(self.bbox_preds_dir)
        create_dir(self.score_maps_dir)
        create_dir(self.score_distrib_dir)

        self.cpreds = [] # class predictions
        self.clbls = [] # class labels
        self.bboxes = [] # bboxes corresponding to class predictions
        self.scores = []
        self.names = []

        prev_name = None
        for idx, self.data in tqdm(self.dataset.iterrows(), total=self.dataset.shape[0]): 

            if self.data['name'] == prev_name: 
                continue
            else: 
                prev_name = self.data['name']
            
            # load image 
            self.im_rgb = cv2.imread(os.path.join(self.data_dir, self.data['name']), cv2.IMREAD_UNCHANGED)
            self.im = self.im_rgb.copy()
            self.im = self.im.astype(np.float64) / 255.

            # find the label
            self.x_min_true, self.x_max_true = self.data['x_min'], self.data['x_max']
            self.y_min_true, self.y_max_true = self.data['y_min'], self.data['y_max']

            cpreds, clbls, bboxes, scores = [], [], [], []
            names = []
            all_scores = [] # for plotting

            # examine patches in the testing image in overlapping patches
            for self.y_min in range(0, self.im.shape[0], self.stride): 
                all_scores_x = []
                for self.x_min in range(0, self.im.shape[1], self.stride): 
                    
                    # get coordinates of patch to cut from scene
                    self.y_max, self.x_max = self.y_min+self.patch_size, self.x_min+self.patch_size
                    if self.y_max >= self.im.shape[0]: self.y_min, self.y_max = self.im.shape[0]-self.patch_size, self.im.shape[0]
                    if self.x_max >= self.im.shape[1]: self.x_min, self.x_max = self.im.shape[1]-self.patch_size, self.im.shape[1]

                    # determine label of patch
                    cur_iou = self.get_iou({'x1':self.x_min_true, 'x2':self.x_max_true, 'y1':self.y_min_true, 'y2':self.y_max_true},
                        {'x1':self.x_min, 'x2':self.x_max, 'y1':self.y_min, 'y2':self.y_max})
                    self.clbl = 1 if cur_iou > self.overlap_thres else 0

                    # cut patch
                    self.patch = self.im[self.y_min:self.y_max, self.x_min:self.x_max]

                    # get prediction and scores
                    self.test_step()

                    all_scores_x.append(self.score)

                    # save only TP, FP, FN
                    if (self.clbl==1) or (self.z==1):
                        cpreds.append(self.z)
                        scores.append(self.score)
                        bboxes.append([self.x_min, self.x_max, self.y_min, self.y_max])
                        clbls.append(self.clbl)
                        names.append(self.data['name'])

                all_scores.append(all_scores_x)

            self.all_scores = self.visualize_scores(all_scores)

            if self.model_type == '1': 
                self.thres = np.mean(scores) - np.std(scores)*2
                cpreds = np.array(cpreds)
                cpreds[np.where(scores>self.thres)] = 0
                cpreds = cpreds.tolist()

                # save only TP, FP, FN
                cpreds = np.array(cpreds)
                scores = np.array(scores)
                bboxes = np.array(bboxes)
                clbls = np.array(clbls)
                names = np.array(names)

                idx_save = np.where((cpreds+np.array(clbls))>=1)

                cpreds = cpreds[idx_save].tolist()
                scores = scores[idx_save].tolist()
                bboxes = bboxes[idx_save].tolist()
                clbls = clbls[idx_save].tolist()
                names = names[idx_save].tolist()
            else: 
                self.thres = 0.5

            # visualize result of entire scene
            self.test_scene_visualization(cpreds, bboxes, clbls)

            self.cpreds += cpreds
            self.bboxes += bboxes
            self.clbls += clbls
            self.scores += scores
            self.names += names
            
        self.bboxes = np.array(self.bboxes)
        self.save_res_csv(name=self.names, x1=self.bboxes[:,0], x2=self.bboxes[:,1], y1=self.bboxes[:,2], y2=self.bboxes[:,3], thres=self.thres, clbl=self.clbls, cpred=self.cpreds, score=self.scores)
        
        # evaluate 
        self.compute_metrics()


    def visualize_scores(self, all_scores): 

        all_scores = np.array(all_scores)
        all_scores = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min())
        all_scores = (all_scores*255).astype(np.uint8)[..., np.newaxis]
        all_scores = np.concatenate((all_scores,all_scores,all_scores), axis=-1)

        return all_scores
        

    def test_scene_visualization(self, cpreds, bboxes, clbls): 

        # create image
        im_plot = self.im_rgb.copy()
        im_plot = cv2.cvtColor(im_plot, cv2.COLOR_RGB2BGR)

        # overlay all predicted bounding boxes
        for idx in range(len(bboxes)): 

            cpred = cpreds[idx]
            bbox = bboxes[idx]
            clbl = clbls[idx]

            if (cpred==1) and (clbl==1): # tp 
                color = (0,255,0)
            elif (cpred==0) and (clbl==1): # fn
                color = (255,0,0)
            elif (cpred==1) and (clbl==0): # fp
                color = (255,255,0)
            thickness = 3 if cpred==clbl else 1
            cv2.rectangle(im_plot, (bbox[0], bbox[2]), (bbox[1], bbox[3]), color=color, thickness=thickness)

        plt.clf(), plt.close() 
        plt.figure(figsize=(10,7))
        plt.subplot(1,2,1), plt.imshow(im_plot), plt.title('bbox preds')
        plt.subplot(1,2,2), plt.imshow(self.all_scores), plt.title('scores')
        # plt.show()
        save_path = os.path.join(self.bbox_preds_dir, self.data['name'])
        plt.savefig(save_path)


class DiffVectorModel(BaseModel): 

    def __init__(self, device, data_dir, res_dir, fold_idx): 
        super().__init__(device, data_dir, res_dir, fold_idx)


    def train(self):

        # 0 - neg, 1 - pos
        patch_means = [np.zeros((self.patch_size, self.patch_size, 3)), np.zeros((self.patch_size, self.patch_size, 3))]
        for idx, data in tqdm(self.train_data.iterrows(), total=self.train_data.shape[0]): 
            im = cv2.imread(os.path.join(self.data_dir, data['name']), cv2.IMREAD_UNCHANGED)
            im = im.astype(np.float64) / 255.
            patch = im[data['y_min']:data['y_max'], data['x_min']:data['x_max']]
            patch_means[data['class']] += patch

        pos_num = len(self.train_data[self.train_data['class']==1])
        neg_num = len(self.train_data[self.train_data['class']==0])

        patch_means[1] /= pos_num
        patch_means[0] /= neg_num

        self.diff_vec = patch_means[0] - patch_means[1]
        self.diff_vec = (self.diff_vec - self.diff_vec.min()) / (self.diff_vec.max() - self.diff_vec.min())


    def test_step(self): 

        self.patch = self.patch.flatten()[np.newaxis, ...]
        self.diff_vec = self.diff_vec.flatten()[..., np.newaxis]
        self.score = np.dot(self.patch, self.diff_vec)[0,0]
        self.z = 1


class LogisticRegressionModel(BaseModel): 

    def __init__(self, device, data_dir, res_dir, fold_idx): 
        super().__init__(device, data_dir, res_dir, fold_idx)

    def train(self): 

        xs, ys = [], []

        for idx, data in tqdm(self.train_data.iterrows(), total=self.train_data.shape[0]): 
            im = cv2.imread(os.path.join(self.data_dir, data['name']), cv2.IMREAD_UNCHANGED)
            im = im.astype(np.float64) / 255.
            patch = im[data['y_min']:data['y_max'], data['x_min']:data['x_max']].flatten()
            xs.append(patch)
            ys.append(data['class'])

        self.model = LogisticRegression(max_iter=1000).fit(xs, ys)

    
    def test_step(self): 

        self.z = self.model.predict(self.patch.reshape(1,-1))[0]
        self.score = self.model.predict_proba(self.patch.reshape(1,-1))[0][1]

class FCNetwork(nn.Module): 

    def __init__(self, channel_num, batch_size): 
        super(FCNetwork, self).__init__()

        self.batch_size = batch_size
        self.patch_size = 28
        self.in_channels = self.patch_size*self.patch_size*channel_num

        divs = [10,100]
        
        self.hl1 = nn.Linear(self.in_channels, self.in_channels//divs[0]) # hidden layer 1
        self.al1 = nn.ReLU() # activation layer 1
        self.hl2 = nn.Linear(self.in_channels//divs[0], self.in_channels//divs[1])
        self.al2 = nn.ReLU()
        self.hl3 = nn.Linear(self.in_channels//divs[1], 2)
        self.out = nn.Softmax(dim=-1)


    def forward(self, x): 

        x = torch.reshape(x, (x.size()[0], -1)).float()

        x = self.hl1(x)
        x = self.al1(x)
        
        x = self.hl2(x)
        x = self.al2(x)
        
        x = self.hl3(x)
        x = self.out(x)

        return x


class FCNModel(BaseModel): 

    def __init__(self, device, data_dir, res_dir, fold_idx): 
        super().__init__(device, data_dir, res_dir, fold_idx)
        # self.device = device
        self.channel_num = 3
        self.batch_size = 8
        self.lr = 1E-3
        self.max_epoch = 100

        self.net = FCNetwork(self.channel_num, self.batch_size).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)


    def load_data(self, csv_path, mode, loader_type=None): 
        if mode == 'train': 
            self.train_data = DataLoader(PatchLoader(csv_path, self.data_dir, mode, self.res_root_dir), batch_size=self.batch_size, shuffle=True)


    def forward_step(self): 

        self.optimizer.zero_grad()
        self.zs = self.net(self.xs)
        

    def backward_step(self): 

        loss = self.loss_fn(self.zs, self.ys)
        loss.backward()
        self.optimizer.step()
        

    def train(self): 

        self.net.train()
        for self.epoch in range(1, self.max_epoch+1): 
            print(f'epoch {self.epoch}/{self.max_epoch} start!')

            for idx, (self.xs, self.ys, self.names) in enumerate(tqdm(self.train_data)): 

                self.xs = self.xs.to(self.device)
                self.ys = self.ys.to(self.device)
                self.forward_step()
                self.backward_step()


    def test_step(self): 

        self.net.eval()
        self.patch = (self.patch*255).astype(np.uint8)
        self.patch = np.moveaxis(self.patch, -1, 0)[np.newaxis, ...]
        self.xs = torch.from_numpy(self.patch).type(torch.float32).to(self.device)
        self.forward_step()

        _, self.z = torch.max(self.zs, 1)
        self.score = self.zs[0][1]

        self.z = self.z.cpu().item()
        self.score = self.score.cpu().item()
