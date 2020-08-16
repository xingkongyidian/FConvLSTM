from __future__ import print_function
import os
import pandas as pd
import numpy as np

from stresnet.utils import string2timestamp


class STMatrix(object):
	"""docstring for STMatrix"""

	def __init__(self, data, timestamps, T=48, CheckComplete=True):
		super(STMatrix, self).__init__()
		assert len(data) == len(timestamps)
		self.data = data
		self.timestamps = timestamps
		self.T = T
		self.pd_timestamps = string2timestamp(timestamps, T=self.T)
		if CheckComplete:
			self.check_complete()
		# index
		self.make_index()

	def make_index(self):
		self.get_index = dict()
		for i, ts in enumerate(self.pd_timestamps):
			self.get_index[ts] = i

	def check_complete(self):
		missing_timestamps = []
		offset = pd.DateOffset(minutes=24 * 60 // self.T)
		pd_timestamps = self.pd_timestamps
		i = 1
		while i < len(pd_timestamps):
			if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
				missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
			i += 1
		for v in missing_timestamps:
			print(v)
		assert len(missing_timestamps) == 0

	def get_matrix(self, timestamp):
		return self.data[self.get_index[timestamp]]

	def save(self, fname):
		pass

	def check_it(self, depends):
		for d in depends:
			if d not in self.get_index.keys():
				return False
		return True

	def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1,
					   channels=False, C_in_P = 1):
		"""current version
		"""
		# offset_week = pd.DateOffset(days=7)
		offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
		XC = []
		XP = []
		XT = []
		Y = []
		timestamps_Y = []

		C_in_T = C_in_P

		depends = [range(1, len_closeness + 1),
				   # [PeriodInterval * self.T * j for j in range(1, len_period+1)],
				   # [TrendInterval * self.T * j for j in range(1, len_trend+1)]]
				   [i + PeriodInterval * self.T * j for j in range(1, len_period + 1) for i in range(0, C_in_P)],
				   [i + TrendInterval * self.T * j for j in range(1, len_trend + 1) for i in range(0, C_in_T)]]
		i = max(self.T * TrendInterval * len_trend + C_in_T - 1, self.T * PeriodInterval * len_period + C_in_P -1, len_closeness)

		while i < len(self.pd_timestamps):
			Flag = True
			for depend in depends:
				if Flag is False:
					break
				Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

			if Flag is False:
				i += 1
				continue
			x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]  # [x-1, x-2, x-3]
			x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]  # [x-10, x-20, x-30]
			x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]  # [x-100, x-200, x-300]
			y = self.get_matrix(self.pd_timestamps[i])
			if channels:    # 按照时间顺序，重新排列 输入
				for x_seq in [x_c, x_p, x_t]:  #反向后 [x-3,x-2,x-1], [x-30,x-20,x-10]
					x_seq.reverse()

			if len_closeness > 0:
				XC.append(np.vstack(x_c))
			if len_period > 0:
				XP.append(np.vstack(x_p))
			if len_trend > 0:
				XT.append(np.vstack(x_t))
			Y.append(y)
			timestamps_Y.append(self.timestamps[i])
			i += 1
		XC = np.asarray(XC)
		XP = np.asarray(XP)
		XT = np.asarray(XT)
		Y = np.asarray(Y)
		if channels:
			XC = XC.reshape((XC.shape[0], len_closeness, 2, 32, 32))
			XP = XP.reshape((XP.shape[0], len_period*C_in_P, 2, 32, 32))
			XT = XT.reshape((XT.shape[0], len_trend*C_in_T, 2, 32, 32))
		print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)

		return XC, XP, XT, Y, timestamps_Y
	

	def create_dataset_new(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1, channels=False,C_in_P=1):
		"""current version ,这个函数是将lc,lp,lt合在一起
		"""
		# offset_week = pd.DateOffset(days=7)
		offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
		X = []
		Y = []
		timestamps_Y = []
		C_in_T = C_in_P
		depends = [range(1, len_closeness + 1),
				   # [PeriodInterval * self.T * j for j in range(1, len_period+1)],
				   # [TrendInterval * self.T * j for j in range(1, len_trend+1)]]
				   [i + PeriodInterval * self.T * j for j in range(1, len_period + 1) for i in range(0, C_in_P)],
				   [i + TrendInterval * self.T * j for j in range(1, len_trend + 1) for i in range(0, C_in_T)]]
		i = max(self.T * TrendInterval * len_trend + C_in_T - 1, self.T * PeriodInterval * len_period + C_in_P - 1,
				len_closeness)

		while i < len(self.pd_timestamps):
			Flag = True
			for depend in depends:
				if Flag is False:
					break
				Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

			if Flag is False:
				i += 1
				continue
			x =[]
			for depend in depends:
				for j in depend:
					x.append(self.get_matrix(self.pd_timestamps[i] - j * offset_frame))
			x.reverse()  # 反转后： [301, 300, 201, 200, 101, 100, 31, 30, 21, 20, 11, 10, 3, 2, 1]
			y = self.get_matrix(self.pd_timestamps[i])
			X.append(np.vstack(x))
			Y.append(y)
			timestamps_Y.append(self.timestamps[i])
			i += 1
		X = np.asarray(X)
		Y = np.asarray(Y)
		if channels:
			X = X.reshape((X.shape[0], len_closeness+len_period*C_in_P+len_trend*C_in_T, 2, 32, 32))
		print("X shape: ", X.shape,  "Y shape:", Y.shape)
		
		return X , Y, timestamps_Y
				


class SDMatrix(object):
	def __init__(self, data, timestamps, step, T=48, CheckComplete=True):
		super(SDMatrix, self).__init__()
		assert len(data[0]) == len(timestamps)
		self.data = data
		self.timestamps = timestamps
		self.step = step
		self.T = T
		self.pd_timestamps = string2timestamp(timestamps, T=self.T)
		if CheckComplete:
			self.check_complete()
		# index
		self.make_index()

	def make_index(self):
		self.get_index = dict()
		for i, ts in enumerate(self.pd_timestamps):
			self.get_index[ts] = i

	def check_complete(self):
		missing_timestamps = []
		offset = pd.DateOffset(minutes=24 * 60 // self.T)
		pd_timestamps = self.pd_timestamps
		i = 1
		while i < len(pd_timestamps):
			if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
				missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
			i += 1
		for v in missing_timestamps:
			print(v)
		assert len(missing_timestamps) == 0

	def get_matrix(self, timestamp, index=0):
		return self.data[index][self.get_index[timestamp]]

	def save(self, fname):
		pass

	def check_it(self, depends):
		for d in depends:
			if d not in self.get_index.keys():
				return False
		return True

	def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
		"""current version
		"""
		# offset_week = pd.DateOffset(days=7)
		offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
		XC = []
		XP = []
		XT = []
		Y = []
		timestamps_Y = []
		timestamps_mark = np.zeros(len(self.timestamps))  # timestamps 对应的mask
		depends = [range(1, len_closeness + 1),
				   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
				   [TrendInterval * self.T * j for j in range(1, len_trend + 1)]]

		i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness) + self.step
		i_start = i
		count = 0
		while i < len(self.pd_timestamps):
			Flag = True
			for depend in depends:
				if Flag is False:
					break
				Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

			if Flag is False:
				i += 1
				count += 1
				continue

			if (self.step == 0):
				x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame, 0) for j in depends[0]]
			if (self.step == 1):
				x_c = [
					self.get_matrix(self.pd_timestamps[i] - offset_frame, 1),
					self.get_matrix(self.pd_timestamps[i] - 2 * offset_frame, 0),
					self.get_matrix(self.pd_timestamps[i] - 3 * offset_frame, 0)
				]
			if (self.step == 2):
				x_c = [
					self.get_matrix(self.pd_timestamps[i] - offset_frame, 2),
					self.get_matrix(self.pd_timestamps[i] - 2 * offset_frame, 1),
					self.get_matrix(self.pd_timestamps[i] - 3 * offset_frame, 0)
				]
			if (self.step >= 3):
				x_c = [
					self.get_matrix(self.pd_timestamps[i] - offset_frame, 3),
					self.get_matrix(self.pd_timestamps[i] - 2 * offset_frame, 2),
					self.get_matrix(self.pd_timestamps[i] - 3 * offset_frame, 1)
				]
			x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame, 0) for j in depends[1]]
			x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame, 0) for j in depends[2]]
			y = self.get_matrix(self.pd_timestamps[i], 0)
			timestamps_mark[i] = 1  # 下标i对应的timestamp 时的流量将被预测
			if len_closeness > 0:
				XC.append(np.vstack(x_c))
			if len_period > 0:
				XP.append(np.vstack(x_p))
			if len_trend > 0:
				XT.append(np.vstack(x_t))
			Y.append(y)
			timestamps_Y.append(self.timestamps[i])
			i += 1
		XC = np.asarray(XC)
		XP = np.asarray(XP)
		XT = np.asarray(XT)
		Y = np.asarray(Y)
		print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
		print("step: ", self.step, "count: ", count, "i_start: ", i_start, "len of pd_timestamps: ",
			  len(self.pd_timestamps))

		return XC, XP, XT, Y, timestamps_Y, timestamps_mark


if __name__ == '__main__':
	pass
