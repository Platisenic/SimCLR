from torch.utils.data import Dataset
import torch
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def randomCrop(x: np.array, k=4, p=0.2):
    # x: [12, 2500]
    total_lens = x.shape[-1]//k
    zero_lens = int(total_lens*p)
    r = np.random.randint(0, total_lens-zero_lens, k)
    for j in range(x.shape[0]):
        for i in range(k):
            x[j][i*total_lens+r[i]:i*total_lens+r[i]+zero_lens] = 0
    return x
    

class TVGHDataset(Dataset):
    def __init__(self,
                 annotation_file,
                 ecg_dir = '/ecgdata/libos/filtered_data/',
                 threshold = 40,
                 feature = 'sd',
                 is3lead=False,
                 isRaligned=False,
                 split=True,
                 transform=None):
        self.anno = pd.read_csv(annotation_file)
        self.ecg_dir = ecg_dir
        self.threshold = threshold
        self.feature = feature
        self.is3lead = is3lead
        self.isRaligned = isRaligned
        self.split = split
        self.target = self.anno[f'{self.feature}_{self.threshold}'].to_numpy()
        self.transform = transform

        logging.info(f'dataset: {os.path.basename(annotation_file)}')
        lables = np.bincount(self.target)
        for i, label in enumerate(lables):
            logging.info(f'class {i}: {label}, {label/len(self):.3f}')

        logging.info(f'total: {len(self)}')

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        ecg_path = os.path.join(self.ecg_dir, f'{self.anno.loc[idx, "year"]}/{self.anno.loc[idx, "EcgFileName"]}')
        label = self.anno.loc[idx, f'{self.feature}_{self.threshold}']
        ecgdata = np.load(ecg_path).astype(np.float32) # [12, 5000]
        if self.isRaligned:
            s1 = self.anno.loc[idx, "rrs1"]
            s2 = self.anno.loc[idx, "rrs2"]
            part1 = ecgdata[:, s1:s1+2500] # [12, 2500]
            part2 = ecgdata[:, s2:s2+2500] # [12, 2500]
            assert part1.shape[1] == 2500
            assert part2.shape[1] == 2500
            if self.is3lead:
                part1 =  np.vstack([
                    part1[6] - part1[7],
                    part1[8] - part1[9],
                    part1[10] - part1[11],
                ]) # [3, 2500]
                part2 =  np.vstack([
                    part2[6] - part2[7],
                    part2[8] - part2[9],
                    part2[10] - part2[11],
                ]) # [3, 2500]
        else:
            part1 = ecgdata[:, :2500]
            part2 = ecgdata[:, 2500:]
            if self.is3lead:
                part1 =  np.vstack([
                    part1[6] - part1[7],
                    part1[8] - part1[9],
                    part1[10] - part1[11],
                ]) # [3, 2500]
                part2 =  np.vstack([
                    part2[6] - part2[7],
                    part2[8] - part2[9],
                    part2[10] - part2[11],
                ]) # [3, 2500]

        if self.transform:
            part1 = self.transform(part1)
            part2 = self.transform(part2)

        if self.split:
            return [part1, part2], label
        else:
            return part1, label

def draw3leadFigure(datas): # datas:(3, sample)
    fig = plt.figure(figsize=(24, 8))
    fig.set_facecolor('xkcd:light grey')
    for i, data in enumerate(datas):
        subplot = fig.add_subplot(3, 1, i+1)
        subplot.plot(data)
    plt.show()
    plt.close(fig)

if __name__ == '__main__':
    dataset = TVGHDataset('/ecgdata/libos/SimCLR/anno/40_9pa_rr/valid.csv', transform=randomCrop)
    data, label = dataset[0]
    print(data[0].shape)
    print(data[1].shape)
    print(label)
    # draw3leadFigure(data[1][:3])
