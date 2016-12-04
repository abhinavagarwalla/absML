__author__ = 'abhinav'

from load_data import *
from generate_dataset import save_feature
import numpy as np

print('- Data and Modules Loaded')

if __name__ == '__main__':
    data_all = np.log10(1 + data_all)
    train = np.log10(1 + train)

    data_all = (data_all - train.mean()) / (data_all.max() - data_all.min())
    data_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_all.dropna(1, inplace=True)

    save_feature(data_all.values, 'mainFeatures_log10_normalized')
    train_target[0].values.dump('%s%s.npy' % (DATASET_PATH, 'Y_train'))
    valid_target[0].values.dump('%s%s.npy' % (DATASET_PATH, 'Y_valid'))

from configs import *
from load_data import *
from sklearn.preprocessing import PolynomialFeatures
from generate_dataset import *
import itertools

top = ["cli_tcp_win", "cli_pl_header", "srv_tcp_win", "cli_pl_tot", "srv_pl_header", "srv_pl_tot", "srv_pl_body",
       "srv_cont_len", "server_latency", "load_time", "application_latency", "srv_tcp_empty", "cli_tcp_tot_bytes",
       "srv_tcp_tot_bytes", "bytes", "client_latency", "srv_tx_time", "cli_cont_len", "cli_pl_body", "srv_tcp_full"]

if __name__ == "__main__":
    print('- Data Loaded')
    top = list(data_all.columns.values)

    for feat1, feat2 in itertools.combinations(top, 2):
        col1 = data_all[feat1].values
        col2 = data_all[feat2].values

        feat = col1 * col2
        print('%s_x_%s Generated' % (feat1, feat2))
        save_feature(feat, '%s_x_%s' % (feat1, feat2))

        feat = col1 / col2
        print('%s_by_%s Generated' % (feat1, feat2))
        save_feature(feat, '%s_by_%s' % (feat1, feat2))

    print('- Features Generated')
