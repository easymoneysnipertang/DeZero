import math
pil_available = True
try:
    from PIL import Image
except:
    pil_available = False
import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset  # 具有dataset接口的实例（实现了__len__和__getitem__方法）
        self.batch_size = batch_size
        self.shuffle = shuffle  # 是否打乱数据集
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)

        self.reset()

    def reset(self):  # 重置迭代器
        self.iteration = 0
        # 重新排列数据集
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:  # 如果迭代次数大于最大迭代次数
            self.reset()
            raise StopIteration

        # 生成batch数据
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        # 将数据转换为ndarray
        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()