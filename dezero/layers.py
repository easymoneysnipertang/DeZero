from dezero.core import Parameter
import weakref
import numpy as np
import dezero.functions as F


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


# =============================================================================
# Linear
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

    def _init_W(self):
        I, O = self.input_size, self.output_size
        # 初始化权重，使用标准差为1/sqrt(I)的高斯分布
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
    
    def forward(self, x):
        if self.W.data is None:  # 如果W未初始化，根据输入大小进行初始化
            self.input_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y