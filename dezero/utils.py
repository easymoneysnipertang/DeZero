import os
import subprocess
import numpy as np
import urllib.request


# =============================================================================
# Visualize for computational graph
# =============================================================================
def _dot_var(v, verbose=False):
    # 将Variable实例赋给函数，返回DOT语言编写的表示实例信息的字符串
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name = '' if v.name is None else v.name
    # verbose为True时，添加变量的形状和数据类型
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)  # 使用python内置函数id()获取对象的内存地址


def _dot_func(f):
    # 将Function实例赋给函数，返回DOT语言编写的表示实例信息的字符串
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)
    # 用DOT记述函数与输入变量之间的连接，以及函数与输出变量之间的连接
    dot_edge = '{} -> {}\n' 
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # y是weakref
    return txt


def get_dot_graph(output, verbose=True):
    # 参考backward函数的实现，实现获取计算图的函数
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:  # 不需要按辈分顺序添加函数
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)  # 输出变量

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)  # 函数
        for x in func.inputs:
            txt += _dot_var(x, verbose)  # 输入变量

            if x.creator is not None:
                add_func(x.creator)  # 添加函数

    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    '''
    使用Graphviz将计算图保存为图像文件
    output: 输出变量
    verbose: 是否添加变量的形状和数据类型
    to_file: 图像文件保存路径
    '''
    dot_graph = get_dot_graph(output, verbose)

    # 保存为dot文件
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):  # 创建目录
        os.mkdir(tmp_dir)
    dot_file = os.path.join(tmp_dir, 'tmp.dot')
    with open(dot_file, 'w') as f:
        f.write(dot_graph)

    # 调用Graphviz的dot命令
    ext = os.path.splitext(to_file)[1][1:]  # 获取文件扩展名
    cmd = 'dot {} -T {} -o {}'.format(dot_file, ext, to_file)
    subprocess.run(cmd, shell=True)

    # 在juptyer notebook中显示图像
    try:
        from PIL import Image
        img = Image.open(to_file)
        img.show()
    except ImportError:
        pass


# =============================================================================
# Download functions
# =============================================================================
def show_progress(block_num, block_size, total_size):
    '''
    显示下载进度条
    '''
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')


# cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')
cache_dir = os.path.join(os.path.dirname(__file__), 'cache')


def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/.dezero`.

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.

    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


# =============================================================================
# Utility functions for numpy
# =============================================================================
def sum_to(x, shape):
    '''
    将x的各轴的元素加起来，使其形状变为shape
    '''
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    '''
    为Reshape和Sum函数的反向传播提供支持
    '''
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def logsumexp(x, axis=1):
    # xp = cuda.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    # xp.exp(y, out=y)
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    # xp.log(s, out=s)
    np.log(s, out=s)
    m += s
    return m


def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape


# =============================================================================
# Other utility functions
# =============================================================================
def pair(x):
    '''
    将x转换为长度为2的元组
    '''
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError