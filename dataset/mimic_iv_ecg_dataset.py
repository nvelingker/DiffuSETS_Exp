import os
import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from scipy import signal
import wfdb
from wfdb import processing


def heart_rate_from_rr_interval(rr_interval_ms, fallback_detector):
    """Use valid millisecond RR metadata, otherwise run the supplied detector."""
    try:
        rr_interval_ms = float(rr_interval_ms)
    except (TypeError, ValueError):
        rr_interval_ms = np.nan
    if np.isfinite(rr_interval_ms) and 300.0 <= rr_interval_ms <= 1500.0:
        return 60000.0 / rr_interval_ms
    heart_rate = fallback_detector()
    if heart_rate is None or not np.isfinite(heart_rate) or heart_rate <= 0:
        raise ValueError("invalid RR interval and XQRS could not estimate heart rate")
    return float(heart_rate)


class MIMIC_IV_ECG_Dataset(Dataset):
    def __init__(self,
                 dataset_path: str, 
                 usage: str='all', 
                 num_folds: int=10, 
                 test_fold: int=None, 
                 seed: int=42, 
                 resample_length: int=1024, 
                 demo_label=False):

        self.resample_length = resample_length
        self.dataset_path = dataset_path

        # Use all data
        self.record_list = pd.read_csv(os.path.join(self.dataset_path, 'record_list.csv'), low_memory=False)
        # Only use data having note (FUTURE)
        # self.record_list = pd.read_csv(os.path.join(self.dataset_path, 'waveform_note_links.csv'), low_memory=False)

        self.mach_mea = pd.read_csv(os.path.join(self.dataset_path, 'machine_measurements.csv'), low_memory=False)
        self.sheet = pd.merge(self.record_list, self.mach_mea, how='inner', on=['subject_id', 'study_id'])

        with open('/data/0shared/laiyongfan/data_text2ecg/exclude_list.txt', 'r') as f:
            exclude_list = [eval(x.strip()) for x in f.readlines()]
        self.sheet.drop(exclude_list, inplace=True)

        # split train and test data
        if usage in ['train', 'test']:
            if seed is not None:
                np.random.seed(seed)
            folds = np.random.randint(0, num_folds, size=len(self.sheet), dtype=np.int8)
            self.sheet['fold'] = folds

            if test_fold is None:
                test_fold = num_folds - 1
            if usage == 'train':
                sheet_mask = self.sheet['fold'] != test_fold
            else:
                sheet_mask = self.sheet['fold'] == test_fold
            self.sheet = self.sheet[sheet_mask]

        # Whether to use age and gender label
        self.demo_label = demo_label
        self.patient_table = None
        if demo_label:
            patient_table_path = '/data1_science/1shared/physionet.org/files/mimiciv/2.2/hosp/patients.csv'
            self.patient_table = pd.read_csv(patient_table_path, index_col='subject_id', low_memory=False)


    # Preprocessing function for waveform data
    def _waveform_preprocess(self, x: np.ndarray):
        # x: (L=5000, C=12)

        x = np.nan_to_num(x)
        # resample x to intended length
        if self.resample_length:
            # x: (L, C) -> (resample_length, C)
            x = signal.resample(x, self.resample_length)

        x = torch.as_tensor(x, dtype=torch.float)
        return x

    # Preprocessing function for text label
    def _text_preprocess(self, texts: list):
        # texts: list of 18 reports, where blank is parsed as np.NaN

        text_clean = []
        # wash nan value in texts
        for x in texts:
            if isinstance(x, str):
                text_clean.append(x)

        # TODO: add text embedding phase
        # a simple concat way 
        text_clean = '|'.join(text_clean)

        return text_clean

    def __getitem__(self, idx: int):
        item_path = os.path.join(self.dataset_path, self.sheet['path'].iloc[idx])

        sig, fields = wfdb.rdsamp(item_path)
        x = self._waveform_preprocess(sig)
        
        texts = [self.sheet.iloc[idx][f'report_{x}'] for x in range(18)]
        text = self._text_preprocess(texts)

        rr_interval_ms = self.sheet.iloc[idx]['rr_interval']

        def detect_heart_rate():
            heart_rate = None
            for lead in range(12):
                xqrs = processing.XQRS(sig=sig[:, lead], fs=fields['fs'])
                try:
                    xqrs.detect(verbose=False)
                except Exception:
                    continue
                qrs_inds = xqrs.qrs_inds
                if len(qrs_inds) > 1:
                    rr_intervals = np.diff(qrs_inds) / fields['fs']
                    heart_rate = 60 / np.mean(rr_intervals)
                    break
            return heart_rate

        # MIMIC rr_interval is in milliseconds. The released branch divided
        # it by 1,000 before applying millisecond bounds and then referenced an
        # undefined rr_intervals variable for valid metadata.
        heart_rate = heart_rate_from_rr_interval(rr_interval_ms, detect_heart_rate)

        label_dict = {
                'text': text, 
                'subject_id': self.sheet.iloc[idx]['subject_id'], 
                'hr': heart_rate, 
                # 'note_id': self.sheet.iloc[idx]['note_id'], 
                }
        
        if self.demo_label:
            patient_id = label_dict['subject_id']
            query = self.patient_table[self.patient_table.index == patient_id] 
            label_dict['age'] = query['anchor_age'].to_list()[0]
            label_dict['gender'] = query['gender'].to_list()[0]
            
        # x: (L, C)
        return x, label_dict

    def __len__(self) -> int:
        return len(self.sheet)
       

class VAE_MIMIC_IV_ECG_Dataset(Dataset):
    def __init__(self, path:str, usage='all'):
        self.path = path
        self.file_list = os.listdir(path)
        # Make sure every time the order of list is the same
        # so as the train and test fold if split in training
        self.file_list.sort(key=lambda x: int(x.split('.')[0]))

        if usage == 'test':
            self.file_list = self.file_list[:50000]

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index) -> tuple:
        latent_file = self.file_list[index]
        latent_dict = torch.load(os.path.join(self.path, latent_file), map_location='cpu')

        # data: (C, L) i.e. (4, 128)
        # label: dict contain keys of [text, subject_id, age, gender]
        return (latent_dict['data'], latent_dict['label'])
    

class DictDataset(Dataset):
    def __init__(self, path:str):
        self.data_dict = torch.load(path)
        self.keys = list(self.data_dict.keys()) 

    def __len__(self):
        return len(self.data_dict) 
    
    def __getitem__(self, idx):
        key = self.keys[idx] 
        latent_dict = self.data_dict[key] 

        return latent_dict['data'], latent_dict['label']
       

if __name__ == '__main__':
    # Original dataset
    # path = '/data/0shared/MIMIC/physionet.org/files/mimic-iv-ecg/1.0/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
    # data = MIMIC_IV_ECG_Dataset(dataset_path=path, resample_length=1024, demo_label=True)

    # VAE encoded dataset 
    # vae_path = '/data/0shared/laiyongfan/data_text2ecg/mimic_vae_lite.pt'
    vae_path = 'mimic_vae.pt'
    # data = VAE_MIMIC_IV_ECG_Dataset(vae_path, usage='test')
    data = DictDataset(vae_path)

    # print(len(data))
    print(data[397][1])
    # print(data[397][1])

    # print(type(data[0][1]['subject_id']))

    # dataloader = DataLoader(data, 512)
    # test reading speed
    for idx, (X, y) in enumerate(tqdm.tqdm(data)):
        pass
