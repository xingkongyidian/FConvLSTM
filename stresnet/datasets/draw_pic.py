import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
from pylab import *  # 支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
import seaborn as sns
from scipy import optimize

def f_1(x, A, b):
    return A*x + b

def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps

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

def show_data_heatmap(data):
    ax = sns.heatmap(data, center=0)
    plt.show()

def trend_office_resident_area():
    fname = 'E:\yhy\DeepST-master\data\TaxiBJ\BJ15_M32x32_T30_InOut.h5'
    stat(fname)
    data, timestamps = load_stdata(fname)
    print('len timestamps : ', len(timestamps))
    print('data shape ', data.shape)
    print('timestamp ', timestamps[96+18])
    print('timestamp ed', timestamps[48*54])
    idx = [114, 450, 786, 1098, 1386, 1702, 2038, 2374, 2710, 2998, 3334, 3670, 4006, 4294, 4630, 4966, 5278]
    trend1 = [data[id, 0, 9, 7] for id in idx]
    trend2 = [data[id, 0, 23, 26] for id in idx]
    time_date =[b'2015030319', b'2015031019', b'2015031719', b'2015032419', b'2015033119',
                b'2015040719', b'2015041419', b'2015042119', b'2015042819',
                b'2015050519', b'2015051219', b'2015051919', b'2015052619',
                b'2015060219', b'2015060919', b'2015061619', b'2015062319']
    x = [i for i in range(1, 18)]
    fig, ax = plt.subplots()
    plt.xlabel('Time')
    plt.ylabel('Inflow')
    x_ = [1, 5, 9, 13, 17]
    plt.xticks(x_, ['3.03', '3.31', '4.28', '5.26', '6.23'])
    A,b = optimize.curve_fit(f_1, x, trend1)[0]
    x1 = np.arange(1, 18, 0.5)
    y1 = A*x1+b
    ax.scatter(x, trend1, c='y')  # office area
    ax.plot(x1,y1,'blue')
    plt.show()

    fig, ax = plt.subplots()
    plt.xlabel('Time')
    plt.ylabel('Inflow')
    x_ = [1, 5, 9, 13, 17]
    plt.xticks(x_, ['3.03', '3.31', '4.28', '5.26', '6.23'])
    ax.scatter(x, trend2, c='y')  # non_office area
    A2, b2 = optimize.curve_fit(f_1, x, trend2)[0]
    x2 = np.arange(1, 18, 0.01)
    y2 = A2 * x2 + b2
    ax.plot(x2,y2,'blue')
    plt.show()


def period_office_resident_area():
    fname = 'E:\yhy\DeepST-master\data\TaxiBJ\BJ16_M32x32_T30_InOut.h5'
    stat(fname)
    data, timestamps = load_stdata(fname)
    print('len timestamps : ', len(timestamps))
    print('data shape ', data.shape)
    x = [24, 72, 120, 168, 216, 264, 312]

    fig, ax = plt.subplots()
    plt.xlabel('Time')
    plt.ylabel('Inflow')
    plt.xticks(x, ['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
    normal = data[4695:5031, 0, 9, 7]
    normal1 = data[4695:5031, 0, 23, 26]
    line1, = ax.plot(normal, linestyle='-') #office area
    ax.legend(loc='lower right')
    plt.show()
    fig, ax = plt.subplots()
    plt.xlabel('Time')
    plt.ylabel('Inflow')
    plt.xticks(x, ['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
    line2, = ax.plot(normal1, linestyle='-') # none_office area
    ax.legend(loc='lower right')
    plt.show()
    # print('timestamp ', timestamps[1776])
    # print('timestamp ed ', timestamps[1920])

def effects_holiday_office_area():
    fname = 'E:\yhy\DeepST-master\data\TaxiBJ\BJ16_M32x32_T30_InOut.h5'
    stat(fname)
    data, timestamps = load_stdata(fname)
    # Feb 15-21 [4695:5031] 左闭右开, Feb 8-14 [4359:4695]
    fig, ax = plt.subplots()
    plt.xlabel('Time')
    plt.ylabel('Inflow')
    x=[24, 72, 120, 168, 216, 264, 312]
    plt.xticks(x, ['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
    holiday= data[4359:4695, 0, 9, 7]
    non_holiday= data[4695:5031, 0, 9, 7]
    line1, = ax.plot(holiday, linestyle='-', label='holiday')
    line2, = ax.plot(non_holiday, linestyle='-', label='normal')
    ax.legend(loc='lower right')
    plt.show()
    # show_data_heatmap(data[16, 0, :, :])
    return None

def effects_thunderstorm_office_area():
    fname = 'E:\yhy\DeepST-master\data\TaxiBJ\BJ13_M32x32_T30_InOut.h5'
    stat(fname)
    data, timestamps = load_stdata(fname)
    # 2013 Aug 10-12 [1440:1584], Aug 17-19 [1776:1920]
    fig, ax = plt.subplots()
    plt.xlabel('Time')
    plt.ylabel('Inflow')
    x=[24,72, 120]
    plt.xticks(x, ['周六', '周日', '周一'])
    holiday = data[1440:1584, 0, 9, 7]
    non_holiday = data[1776:1920, 0, 9, 7]
    line1, = ax.plot(holiday, linestyle='-', label='thunderstorm')
    line2, = ax.plot(non_holiday, linestyle='-', label='normal')
    ax.legend(loc='lower right')
    plt.show()
    print('len timestamps : ', len(timestamps))
    print('data shape ', data.shape)
    print('timestamp ', timestamps[1776])
    print('timestamp ed ', timestamps[1920] )

def dropout_rate_trend():
    font_size = 16
    RMSE=[16.39, 16.24, 16.17, 16.09, 15.97, 16.07, 16.12, 16.64]
    dropout_rate =[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel('dropout rate', fontsize=font_size)
    plt.ylabel('RMSE', fontsize=font_size)
    # fig, ax = plt.subplots()
    # plt.xticks(dropout_rate)
    # line1, = ax.plot(RMSE)
    plt.plot(dropout_rate, RMSE, marker='o')
    plt.show()

def multi_step_trend():
    HA=[36.62]*12
    star =    [16.25, 18.3, 19.55, 20.43, 21.28, 21.91, 22.70, 23.40, 23.90, 24.52, 24.94, 25.26]
    ConvLSTM =[16.71, 18.43, 19.86, 21.54, 22.75, 23.82, 24.74, 25.55, 26.25, 26.87, 27.44, 27.95]
    ST_ResNet =[16.87, 18.58, 19.67, 20.57, 21.45, 22.16, 22.81, 23.53, 24.02, 24.79, 25.03, 25.55]
    FDCN = [16.95, 19.10, 20.77, 22.52, 24.44, 26.53, 28.81, 31.09, 33.37, 35.58, 37.66, 39.57]
    FConvLSTM =[16.03, 17.88, 19.06, 19.94, 20.71, 21.40, 22.03, 22.64, 23.23, 23.81, 24.40, 24.99]

    font_size = 16
    step = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    plt.xlabel('step',fontsize = font_size)
    plt.ylabel('RMSE',fontsize = font_size)
    # fig, ax = plt.subplots()
    plt.xticks(range(0,6), step[0:6], fontsize = font_size)
    line1, = plt.plot(star[0:6], marker='o',  label='STAR')
    line2, = plt.plot(ConvLSTM[0:6], marker='.', label='ConvLSTM')
    line3, = plt.plot(ST_ResNet[0:6], marker='D', label='ST-ResNet')
    line4, = plt.plot(FDCN[0:6], marker='*', label='FDCN')
    line5, = plt.plot(FConvLSTM[0:6], marker='^', label='FConvLSTM')
    plt.legend(fontsize = font_size)
    plt.grid(linestyle='-.')
    plt.show()

def n_rules_trend():

    font_size = 25
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    carse_label=['0', '10', '20', '30', '50', '100']
    carse =[17.44, 15.99, 16.18, 16.34, 16.28, 16.38]
    finner_label=['1','3', '5', '7', '9', '10', '11', '13', '15', '17', '19']
    finner = [16.2, 16.17, 16.38, 16.29, 16.26, 16.01, 16.48, 16.22, 16.16, 16.54, 16.31]

    fig, ax = plt.subplots()
    plt.xticks(range(0,6), carse_label, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel('n_neurons', fontsize=font_size)
    plt.ylabel('RMSE', fontsize= font_size)
    line1, = plt.plot(carse, marker='o')
    # plt.legend(fontsize=font_size)
    plt.grid(linestyle='-.')

    fig, ax = plt.subplots()
    plt.xticks(range(0, 11), finner_label, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel('n_neurons', fontsize=font_size)
    plt.ylabel('RMSE', fontsize=font_size)
    line1, = plt.plot(finner, marker='o')
    # plt.legend(fontsize=font_size)
    plt.grid(linestyle='-.')
    plt.show()

def cpt_trend():
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    c_rmse=[30.8,16.35, 16.12, 16.01, 16.26, 16.39, 16.40, 16.34, 16.43]
    c_ticks=['0', '1', '2', '3', '4', '5', '6', '7', '8']
    p_rmse =[17.02, 16.04, 16.58, 16.81, 17.23]
    p_ticks =['0', '1', '2', '3', '4']
    t_rmse =[16.98, 16.02, 16.58, 16.79]
    t_ticks = ['0', '1', '2', '3']
    plt.figure(1)
    font_size=20
    fig, ax = plt.subplots()
    plt.xlabel('lc', fontsize=font_size)
    plt.ylabel('RMSE', fontsize=font_size)
    plt.ylim(15.5, 16.5)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.bar(range(1,9), c_rmse[1:], tick_label= c_ticks[1:])
    # plt.show()
    plt.figure(2)
    fig, ax = plt.subplots()
    plt.xlabel('ld', fontsize=font_size)
    plt.ylabel('RMSE', fontsize=font_size)
    plt.ylim(15.5, 17.5)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # plt.bar(range(0, 4), p_rmse, tick_label=p_ticks)
    plt.bar(range(len(p_rmse)), p_rmse, tick_label=p_ticks)
    # plt.show()
    plt.figure(3)
    fig, ax = plt.subplots()
    plt.xlabel('lw', fontsize=font_size)
    plt.ylabel('RMSE', fontsize=font_size)
    plt.ylim(15.5, 17.5)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.bar(range(len(t_rmse)), t_rmse, tick_label=t_ticks)
    plt.show()



if __name__ == '__main__':
    # effects_holiday_office_area()
    # effects_thunderstorm_office_area()
    # period_office_resident_area()
    # trend_office_resident_area()
    # dropout_rate_trend()
    # multi_step_trend()
    n_rules_trend()
    # cpt_trend()