import os
import subprocess


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