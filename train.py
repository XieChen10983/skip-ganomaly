# coding=gbk
"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""

##
# LIBRARIES

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model

##
def main():
    """ Training
    """
    opt = Options().parse()
    data = load_data(opt)  # 所得到的data包括train_data和test_data，用data.train_data获取训练数据，data.valid_data获取测试数据。
    model = load_model(opt, data)
    model.train()

if __name__ == '__main__':
    main()
