import numpy as np
import dezero
from dezero import utils
from dezero.core import Function, Variable, as_variable, as_array


# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)  # 因为计算Variable，所以要使用dezero的cos函数
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx
    

def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx
    

def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


# =============================================================================
# tensor operations: reshape / transpose
# =============================================================================
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape  # 记录输入的形状
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)  # 将梯度reshape成输入的形状


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)  # 确保返回的是Variable
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes  # 记录转置的轴，默认为None表示逆序

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))  # 计算逆序
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


# =============================================================================
# tensor operations: sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================
class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis  # 记录求和的轴
        self.keepdims = keepdims  # 是否保持维度

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis,
                                        self.keepdims)  # 为了支持Reshape和Sum函数的反向传播
        gx = broadcast_to(gy, self.x_shape)  # 广播梯度
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)  # 广播梯度，使梯度的形状和输入的形状一致
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape  # 广播的形状

    def forward(self, x):
        self.x_shape = x.shape  # 记录输入的形状
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)  # 复制了，梯度回传时要求和
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def average(x, axis=None, keepdims=False):
    x = as_variable(x)
    y = sum(x, axis, keepdims)
    return y * (y.data.size / x.data.size)


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)  # 矩阵乘法
        return y

    def backward(self, gy):
        x, W = self.inputs  # 按照公式反向传播
        gx = matmul(gy, W.T)  # 矩阵乘法的反向传播
        gW = matmul(x.T, gy)  # 矩阵乘法的反向传播
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)  # ndarray的矩阵乘法
        if b is not None:  # 如果有偏置，加上偏置
            y += b
        return y

    def backward(self, gy):  # 加法和矩阵乘法的反向传播
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


# =============================================================================
# loss functions: mean_squared_error
# =============================================================================
def mean_squared_error_simple(x0, x1):
    x0, x1 = as_variable(x0), as_variable(x1)
    diff = x0 - x1  # 中间Variable变量diff
    y = sum(diff ** 2) / len(diff)
    return y


class MeanSquaredError(Function):  # 封装成Function减少中间变量内存
    def forward(self, x0, x1):
        diff = x0 - x1  # 这里的diff的类型是ndarray
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


# =============================================================================
# activation functions: sigmoid
# =============================================================================
def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        # y = np.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)