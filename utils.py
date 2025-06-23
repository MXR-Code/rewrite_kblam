import time
import warnings
import argparse
import torch
import pandas as pd


def check_gpu():
    if torch.cuda.is_available():
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"GPU 数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i} 信息:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            print(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")
            print(f"  已使用显存: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
            print(f"  剩余显存: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
    else:
        print("CUDA 不可用")
    print()
    print()
    print()


check_gpu()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_args(args):
    for name, value in args.__dict__.items():
        print(f"{name} = {value}")
        if name == 'separate_query_head':
            assert isinstance(args.separate_query_head, bool)
        if name == 'debug':
            assert isinstance(args.debug, bool)
            if args.debug:
                warnings.warn("debugging, not training", UserWarning)
        if name == 'device' and args.device == "cpu":
            warnings.warn("using CPU, not CUDA", UserWarning)
        time.sleep(1)
        print()


def countdown(seconds):
    for i in range(seconds, 0, -1):
        print(i)
        time.sleep(1)
    print("start！")


def save_best_kblam(stopper, dataloader, kblam, time):
    save_best_model_path = f'{time}' + '+'
    save_best_model_path += dataloader.dataset_name + '+'
    save_best_model_path += kblam.sentence_encoder.model_name + '+'
    save_best_model_path += kblam.llm.config.pretrained_model_name_or_path + f'.pth'
    save_best_model_path = save_best_model_path.replace('/', '_')
    torch.save(stopper.best_model_parameter_state_dict, save_best_model_path)


class LossRecoder():
    def __init__(self):
        self.train_loss = pd.DataFrame()
        self.valid_loss = pd.DataFrame()

    def record(self, epoch, batch_index, batch_train_loss=None, batch_valid_loss=None):
        if batch_train_loss and batch_valid_loss is None:
            head = ['epoch', 'batch_index', 'batch_train_loss']
            row = [[epoch, batch_index, batch_train_loss]]
            row = pd.DataFrame(row, columns=head)
            print(row)
            self.train_loss = pd.concat([self.train_loss, row], ignore_index=True)
            if not self.train_loss[self.train_loss['epoch'] == epoch].empty:
                epoch_train_loss = self.train_loss[self.train_loss['epoch'] == epoch]['batch_train_loss'].mean()
                # 最下一行， 'epoch_train_loss'列，赋值epoch_train_loss
                self.train_loss.loc[self.train_loss.index[-1], 'epoch_train_loss'] = epoch_train_loss

        if batch_valid_loss and batch_train_loss is None:
            head = ['epoch', 'batch_index', 'batch_valid_loss']
            row = [[epoch, batch_index, batch_valid_loss]]
            row = pd.DataFrame(row, columns=head)
            print(row)
            self.valid_loss = pd.concat([self.valid_loss, row], ignore_index=True)
            if not self.valid_loss[self.valid_loss['epoch'] == epoch].empty:
                epoch_valid_loss = self.valid_loss[self.valid_loss['epoch'] == epoch]['batch_valid_loss'].mean()
                # 最下一行， 'epoch_valid_loss'列，赋值epoch_valid_loss
                self.valid_loss.loc[self.valid_loss.index[-1], 'epoch_valid_loss'] = epoch_valid_loss

    def draw(self, time):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        style_index = 0
        ticks = []
        ticks_name = []
        for head in self.train_loss.columns:
            if head == "batch_index": continue
            if head == 'epoch':
                sett = set(self.train_loss[head])
                for epoch in sett:
                    ticks.append(self.train_loss[self.train_loss['epoch'] == epoch].index[-1])
                    ticks_name.append(epoch)
                continue

            plt = self.draw_single(plt=plt,
                                   x_values=self.train_loss.index,
                                   y_values=self.train_loss[head],
                                   style_index=style_index,
                                   label=head)
            style_index += 1
        plt.xlabel('Epoch(Batch)')
        plt.ylabel('Loss')
        plt.xticks(ticks=ticks, labels=ticks_name)
        plt.legend()  # 显示图例
        plt.grid(True)
        plt.savefig(f"{time}_train.png")

        plt.figure(figsize=(12, 6))
        style_index = 0
        ticks = []
        ticks_name = []
        for head in self.valid_loss.columns:
            if head == "batch_index": continue
            if head == 'epoch':
                sett = set(self.valid_loss[head])
                for epoch in sett:
                    ticks.append(self.valid_loss[self.valid_loss['epoch'] == epoch].index[-1])
                    ticks_name.append(epoch)
                continue

            plt = self.draw_single(plt=plt,
                                   x_values=self.valid_loss.index,
                                   y_values=self.valid_loss[head],
                                   style_index=style_index,
                                   label=head)
            style_index += 1
        plt.xlabel('Epoch(Batch)')
        plt.ylabel('Loss')
        plt.xticks(ticks=ticks, labels=ticks_name)
        plt.legend()  # 显示图例
        plt.grid(True)
        plt.savefig(f"{time}_valid.png")

    def draw_single(self, plt, x_values, y_values, marker=None, linestyle=None, color=None, style_index=None,
                    label=None):
        color_list = ['red', 'blue', 'green', 'yellow', '#FF0000', (1, 0, 0)]
        marker_list = ['o', 's', 'x', '^', 'v', '>', '<', 'p', '*', 'D', 'h', '+', '|', '_']
        linestyle_list = ['-', '--', ':', '-.', 'None']
        if isinstance(style_index, int):
            if color is None: color = color_list[style_index]
            if marker is None: marker = marker_list[style_index]
            if linestyle is None: linestyle = linestyle_list[style_index]

        assert color in color_list
        assert marker in marker_list
        assert linestyle in linestyle_list

        plt.plot(x_values, y_values, marker=marker, linestyle=linestyle, color=color, label=label)

        return plt

    def record(self, epoch, batch_index, batch_train_loss=None, batch_valid_loss=None):
        if batch_train_loss and batch_valid_loss is None:
            head = ['epoch', 'batch_index', 'batch_train_loss']
            row = [[epoch, batch_index, batch_train_loss]]
            row = pd.DataFrame(row, columns=head)
            self.train_loss = pd.concat([self.train_loss, row], ignore_index=True)
            if not self.train_loss[self.train_loss['epoch'] == epoch].empty:
                epoch_train_loss = self.train_loss[self.train_loss['epoch'] == epoch]['batch_train_loss'].mean()
                # 最下一行， 'epoch_train_loss'列，赋值epoch_train_loss
                self.train_loss.loc[self.train_loss.index[-1], 'epoch_train_loss'] = epoch_train_loss
                print(self.train_loss.tail(1))

        if batch_valid_loss and batch_train_loss is None:
            head = ['epoch', 'batch_index', 'batch_valid_loss']
            row = [[epoch, batch_index, batch_valid_loss]]
            row = pd.DataFrame(row, columns=head)

            self.valid_loss = pd.concat([self.valid_loss, row], ignore_index=True)
            if not self.valid_loss[self.valid_loss['epoch'] == epoch].empty:
                epoch_valid_loss = self.valid_loss[self.valid_loss['epoch'] == epoch]['batch_valid_loss'].mean()
                # 最下一行， 'epoch_valid_loss'列，赋值epoch_valid_loss
                self.valid_loss.loc[self.valid_loss.index[-1], 'epoch_valid_loss'] = epoch_valid_loss
                print(self.valid_loss.tail(1))

    def get_epoch_loss(self, epoch):
        train_loss = self.train_loss[self.train_loss['epoch'] == epoch]['epoch_train_loss'].iloc[-1]
        valid_loss = self.valid_loss[self.valid_loss['epoch'] == epoch]['epoch_valid_loss'].iloc[-1]
        return train_loss, valid_loss


def get_model_hyperparameter(model=None, items=None):
    out = {}
    if model and items is None:
        out['str(model)'] = str(model)
        for name, value in model.__dict__.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                out[name] = value
            else:
                out[name] = name
        return out
    elif items and model is None:
        for name, value in items:
            if isinstance(value, (int, float, str, bool, type(None))):
                out[name] = value
            else:
                out[name] = name
        return out
    else:
        return out

# loss_recorder = LossRecoder()
# loss_recorder.record(epoch=0, batch_index=0, batch_train_loss=1)
# loss_recorder.record(epoch=0, batch_index=1, batch_train_loss=4)
# loss_recorder.record(epoch=0, batch_index=2, batch_train_loss=7)
#
# loss_recorder.record(epoch=1, batch_index=0, batch_train_loss=7)
# loss_recorder.record(epoch=1, batch_index=1, batch_train_loss=8)
# loss_recorder.record(epoch=1, batch_index=2, batch_train_loss=2)
#
# loss_recorder.record(epoch=2, batch_index=0, batch_train_loss=4)
# loss_recorder.record(epoch=2, batch_index=1, batch_train_loss=4)
# loss_recorder.record(epoch=2, batch_index=2, batch_train_loss=9)
#
# train_loss, valid_loss = loss_recorder.get_epoch_loss(epoch=1)
#
# print()
