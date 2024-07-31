from dezero.core import Parameter
import weakref
import numpy as np
import dezero.functions as F
from dezero import cuda
from dezero.utils import pair
import os


# =============================================================================
# Layer (base class)
# =============================================================================
class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):  # 只有Parameter实例才会被添加到_params / step45添加Layer
            self._params.add(name)  # 保存所拥有的参数的名称
        super().__setattr__(name, value)
    
    def __call__(self, *inputs):  # 接收参数并调用forward方法
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        # 通过弱引用保存输入和输出
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]  # 输出为一个值时，返回该值，否则返回元组
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):  # 取出所有参数
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):  # 如果是Layer实例，递归调用params方法
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:  # 遍历所有参数
            obj = self.__dict__[name]  # 根据名称取出参数
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):  # 如果是Layer实例，递归调用_flatten_params方法
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj  # 将参数添加到params_dict中

    def save_weights(self, path):
        self.to_cpu()  # 将参数转移到CPU

        params_dict = {}
        self._flatten_params(params_dict)  # 将所有参数保存到params_dict中
        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)  # 根据key设置参数数据
        for key, param in params_dict.items():
            param.data = npz[key]


# =============================================================================
# Linear / Conv2d / Deconv2d
# =============================================================================
class Linear(Layer):
    def __init__(self, output_size, nobias=False, dtype=np.float32, input_size=None):
        super().__init__()

        self.input_size = input_size  # 输入大小，默认为None，延迟初始化
        self.output_size = output_size
        self.dtype = dtype

        # 初始化参数
        self.W = Parameter(None, name='W')  # 延迟初始化
        if self.input_size is not None:
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(output_size, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, O = self.input_size, self.output_size
        # 初始化权重，使用标准差为1/sqrt(I)的高斯分布
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
    
    def forward(self, x):
        if self.W.data is None:  # 如果W未初始化，根据输入大小进行初始化
            self.input_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y
    

class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional convolutional layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Deconv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional deconvolutional (transposed convolution)layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.deconv2d(x, self.W, self.b, self.stride, self.pad)
        return y
    

# =============================================================================
# RNN / LSTM
# =============================================================================
class RNN(Layer):
    def __init__(self, hidden_size, input_size=None):
        """An Elman RNN with tanh.

        Args:
            hidden_size (int): The number of features in the hidden state.
            input_size (int): The number of features in the input. If unspecified
            or `None`, parameter initialization will be deferred until the
            first `__call__(x)` at which time the size will be determined.

        """
        super().__init__()
        # 将x映射到h的权重
        self.x2h = Linear(hidden_size, input_size=input_size)
        # 将上一时刻的h映射到h的权重
        self.h2h = Linear(hidden_size, input_size=input_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new
    

class LSTM(Layer):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()

        H, I = hidden_size, input_size
        self.x2f = Linear(H, input_size=I)
        self.x2i = Linear(H, input_size=I)
        self.x2o = Linear(H, input_size=I)
        self.x2u = Linear(H, input_size=I)
        self.h2f = Linear(H, input_size=H, nobias=True)
        self.h2i = Linear(H, input_size=H, nobias=True)
        self.h2o = Linear(H, input_size=H, nobias=True)
        self.h2u = Linear(H, input_size=H, nobias=True)
        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            u = F.tanh(self.x2u(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            u = F.tanh(self.x2u(x) + self.h2u(self.h))

        if self.c is None:
            c_new = (i * u)
        else:
            c_new = (f * self.c) + (i * u)

        h_new = o * F.tanh(c_new)

        self.h, self.c = h_new, c_new
        return h_new