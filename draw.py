def get_point(data_points):
    x_values = [point[0] for point in data_points]
    assert len(x_values) == len(set(x_values)), "Error: x values must be unique."
    data_points.sort(key=lambda point: point[0])
    x_values = [point[0] for point in data_points]
    y_values = [point[1] for point in data_points]
    return x_values, y_values


def draw_graph(num_batch, **kwargs):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for index, (name, data_points) in enumerate(kwargs.items()):
        x_values, y_values = get_point(data_points=data_points)
        plt = single(plt, x_values, y_values, linestyle='-', index=index, label=name)
        if 'epoch' in name:
            ticks = [x * num_batch for x in x_values]
            plt.xticks(ticks=ticks, labels=x_values)
            savefig_name = name

    plt.xlabel('Batch/Epoch')
    plt.ylabel('Loss')
    plt.legend()  # 显示图例
    plt.grid(True)
    plt.savefig(f'{savefig_name}.png')


def single(plt, x_values, y_values, marker=None, linestyle=None, color=None, index=None, label=None):
    color_list = ['red', 'blue', 'green', 'yellow', '#FF0000', (1, 0, 0)]
    marker_list = ['o', 's', 'x', '^', 'v', '>', '<', 'p', '*', 'D', 'h', '+', '|', '_']
    linestyle_list = ['-', '--', ':', '-.', 'None']
    if isinstance(index, int):
        if color is None: color = color_list[index]
        if marker is None: marker = marker_list[index]
        if linestyle is None: linestyle = linestyle_list[index]

    assert color in color_list
    assert marker in marker_list
    assert linestyle in linestyle_list

    plt.plot(x_values, y_values, marker=marker, linestyle=linestyle, color=color, label=label)

    return plt

# test
# data_points_1 = [(1, 5), (3, 10), (7, 20), (9, 30)]
# data_points_2 = [(2, 8), (4, 15), (6, 25)]
# epoch_trainloss_line = [(1, 4), (2, 6), (3, 9), (4, 12), (5, 18), (6, 22)]
# data_points_4 = [(5, 10), (10, 15)]
#
# draw_graph(num_batch=2, data_points_1=data_points_1, data_points_2=data_points_2,
#            epoch_trainloss_line=epoch_trainloss_line, data_points_4=data_points_4)
