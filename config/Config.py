# encoding:utf-8


class Config(object):

    def __init__(self):
        self.Batch_Size = 32
        self.epoch = 20
        self.is_cut = True  # 是否对语句进行分词


if __name__ == '__main__':
    config = Config()