# encoding:utf-8


class Config(object):

    def __init__(self):
        self.Batch_Size = 64
        self.epoch = 20
        self.is_cut = False  # 是否对语句进行分词


if __name__ == '__main__':
    config = Config()