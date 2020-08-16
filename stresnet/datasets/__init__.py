from __future__ import print_function
import h5py
import time
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from stresnet.preprocessing import remove_incomplete_days, timestamp2vec
def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps

def show_data_heatmap(data):
    ax = sns.heatmap(data, center=0)
    plt.show()

#get_nb_timeslot get current time is  what timeslot
def stat(fname):
    def get_nb_timeslot(f):
        s = f['date'][0]
        e = f['date'][-1]
        print(len(f['date']))
        print('s', s)
        print('e', e)
        year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
        ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, ts_str, te_str

    with h5py.File(fname) as f:
        nb_timeslot, ts_str, te_str = get_nb_timeslot(f)
        nb_day = int(nb_timeslot / 48)
        mmax = f['data'].value.max()
        mmin = f['data'].value.min()
        stat = '=' * 5 + 'stat' + '=' * 5 + '\n' + \
               'data shape: %s\n' % str(f['data'].shape) + \
               'date shape: %s\n' % str(f['date'].shape) + \
               '# of days: %i, from %s to %s\n' % (nb_day, ts_str, te_str) + \
               '# of timeslots: %i\n' % int(nb_timeslot) + \
               '# of timeslots (available): %i\n' % f['date'].shape[0] + \
               'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
               'max: %.3f, min: %.3f\n' % (mmax, mmin) + \
               '=' * 5 + 'stat' + '=' * 5
        print(stat)


def main():
    # fname='E:\yhy\DeepST-master\data\TaxiBJ\BJ16_M32x32_T30_InOut.h5'
    #fname ='E:\yhy\DeepST-master\data\TaxiBJ\HDF5_FILE1.h5'
    fname = 'E:\yhy\DeepST-master\data\TaxiBJ\HDF5_date1102_0304.h5'
    stat(fname)
    data, timestamps = load_stdata(fname)
    #data, timestamps = remove_incomplete_days(data, timestamps, 48)
    print('type data ', type(data))
    print('type timestamps ', type(timestamps))
    print ('tmiestamps : ', timestamps[0])
    print ('len timestamps : ', len(timestamps))
    print ('data shape ', data.shape)
    print (data[16,0, 20:40, 25:31])
    show_data_heatmap(data[16, 0, :, :])
    show_data_heatmap(data[16, 0, 20:40, 7:31])


if __name__ == '__main__':
    main()