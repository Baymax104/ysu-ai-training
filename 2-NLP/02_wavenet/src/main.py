# -*- coding: UTF-8 -*-
import numpy as np
import params

import os


if __name__ == '__main__':
    array = np.load(os.path.join(params.DATA_DIR, 'dataset.npz'))['arr_0']
    print(array.shape)


