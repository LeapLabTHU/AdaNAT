import os
import os.path as op

import numpy as np
import pandas as pd
import base64
from torch.utils.data import Dataset
from PIL import Image

from io import BytesIO

from loguru import logger


def generate_lineidx_file(filein, idxout):
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos!=fsize:
            tsvout.write(str(fpos)+"\n")
            tsvin.readline()
            fpos = tsvin.tell()
    os.rename(idxout_tmp, idxout)


class TSVFile(object):
    def __init__(self, tsv_file, generate_lineidx=True):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None
        # the process always keeps the process which opens the file.
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and generate_lineidx:
            generate_lineidx_file(self.tsv_file, self.lineidx)

    def __del__(self):
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[idx]
        except:
            logger.info('{}-{}'.format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            logger.info('loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logger.info('re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()


class CC3MTSV(Dataset):
    def __init__(self, root, split, transform=None, txt_only=False):

        if split == 'train':
            self.data = TSVFile(os.path.join(root, f'cc3m_{split}.tsv'))
        elif split == 'val':
            self.data = pd.read_csv(os.path.join(root, f'cc3m_{split}.tsv'), delimiter='\t')
            assert txt_only, 'We only use text for validation'
        else:
            raise ValueError(f'Invalid split: {split}')
        self.split = split
        self.len = len(self.data)

        self.transform = transform
        self.txt_only = txt_only

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.split == 'train':
            while True:
                image_path, file_idx, caption, img_bytes = self.data[idx]
                if self.txt_only:
                    return caption
                try:
                    img = Image.open(BytesIO(base64.urlsafe_b64decode(img_bytes)))
                    images = self.transform(img)
                    return images, caption
                except Exception as e:
                    logger.warning(f'Error reading for {image_path} with caption {caption}: {e}')
                    idx = np.random.randint(self.len)
        elif self.split == 'val':
            return self.data.iloc[idx].to_dict()['caption']
