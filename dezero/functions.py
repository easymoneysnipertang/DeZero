import numpy as np
import dezero
from dezero import utils
from dezero.core import Function, Variable, as_variable, as_array


# =============================================================================
# Basic functions: sin / cos / tanh
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
# tensor operations: sum / sum_to / broadcast_to
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