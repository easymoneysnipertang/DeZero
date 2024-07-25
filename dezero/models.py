from dezero import Layer
from dezero import utils
import dezero.functions as F
import dezero.layers as L


# =============================================================================
# Model / Sequential / MLP
# =============================================================================
class Model(Layer):  # 直接继承Layer类，更方便管理layers
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)  # 可视化计算图
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)
    

class Sequential(Model):
    def __init__(self, *layers):
        super().__init__()
        self.layers = []  # 保存所有层
        for i, layer in enumerate(layers):  # 为每一层设置名称
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):  # 依次调用每一层的forward方法
        for layer in self.layers:
            x = layer(x)
        return x
    

class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        # fc_output_sizes可以是元组或列表，表示每一层的输出大小
        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:  # 前n-1层使用激活函数
            x = self.activation(l(x))
        return self.layers[-1](x)  # 最后一层不使用激活函数