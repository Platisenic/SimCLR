from torch.utils.data import Dataset
import torch
import logging
import os
import pandas as pd
import numpy as np


class TVGHDataset(Dataset):
    def __init__(self,
                 annotation_file,
                 ecg_dir = '/ecgdata/libos/filtered_data/',
                 threshold = 40,
                 feature = 'sd',
                 is3lead=False,
                 isRaligned=False):
        self.anno = pd.read_csv(annotation_file)
        self.ecg_dir = ecg_dir
        self.threshold = threshold
        self.feature = feature
        self.is3lead = is3lead
        self.isRaligned = isRaligned
        self.target = self.anno[f'{self.feature}_{self.threshold}'].to_numpy()

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

        return [part1, part2], label


if __name__ == '__main__':
    dataset = TVGHDataset('/ecgdata/libos/SimCLR/anno/40_9pa_rr/all.csv')
    data, label = dataset[0]
    print(data[0].shape)
    print(data[1].shape)
    print(label)
