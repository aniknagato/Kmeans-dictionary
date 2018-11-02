from collections import Counter

import random
import math
import numpy

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_data(file_name):
	# file_name = '/Users/vphan/Dropbox/datasets/iris.csv'
	with open(file_name) as f:
		header = f.readline()
		points = []
		for line in f:
			items = line.strip().split(',')
			r = [
				int(items[0]),
				int(items[1]),
				float(items[2])
			]
			points.append(r)
	random.shuffle(points)
	return points



class UserMatrix:
	def __init__(self, points):
		self.points = points

	def similarity(self, u1, u2):
		points = self.points

		u1_list = {}
		u2_list = {}

		for i in points:
			if i[0] == u1:
				u1_list[i[1]] = i[2]

			if i[0] == u2:
				u2_list[i[1]] = i[2]

		for k,v in u1_list.items():
			if k not in u2_list.keys():
				u2_list[k] = 0

		for k,v in u2_list.items():
			if k not in u1_list.keys():
				u1_list[k] = 0	

		sumprod = 0
		sum1 =0 
		sum2 = 0

		for k in u1_list.keys():
			sumprod += u1_list[k] * u2_list[k]
			sum1 += u1_list[k]**2
			sum2 += u2_list[k]**2

		cos_sim = sumprod/(math.sqrt(sum1) * math.sqrt(sum2))
		return cos_sim



class MovieMatrix:
	def __init__(self, points):
		self.points = points

	def similarity(self, m1, m2):
		points = self.points

		m1_list = {}
		m2_list = {}

		for i in points:
			if i[1] == m1:
				m1_list[i[0]] = i[2]

			if i[1] == m2:
				m2_list[i[0]] = i[2]

		for k,v in m1_list.items():
			if k not in m2_list.keys():
				m2_list[k] = 0

		for k,v in m2_list.items():
			if k not in m1_list.keys():
				m1_list[k] = 0	

		sumprod = 0
		sum1 =0 
		sum2 = 0

		for k in m1_list.keys():
			sumprod += m1_list[k] * m2_list[k]
			sum1 += m1_list[k]**2
			sum2 += m2_list[k]**2

		cos_sim = sumprod/(math.sqrt(sum1) * math.sqrt(sum2))
	
		return cos_sim


def usersim(d1,d2):
	u1_list = d1
	u2_list = d2

	for k,v in u1_list.items():
		if k not in u2_list.keys():
			u2_list[k] = 0

	for k,v in u2_list.items():
		if k not in u1_list.keys():
			u1_list[k] = 0	

	sumprod = 0
	sum1 =0 
	sum2 = 0

	for k in u1_list.keys():
		sumprod += u1_list[k] * u2_list[k]
		sum1 += u1_list[k]**2
		sum2 += u2_list[k]**2

	cos_sim = sumprod/(math.sqrt(sum1) * math.sqrt(sum2))
	return cos_sim

def dictavg(ld):

	s = {}

	for ii in ld:
		sc = Counter(s)
		ic = Counter(ii)
		ss = sc + ic
		s = dict(ss)

	ll = len(s)
	for k,v in s.items():
		s[k] = v/ll

	return s


class Kmeans:
	def __init__(self, points):
		self.points = points
		self.centroids = []
		self.predicted = {}
		users = {}
		for i in points:
			if i[0] not in users.keys():
				users[i[0]] = {}
			else:
				userpro = users[i[0]]
				userpro[i[1]] = i[2]
				users[i[0]] = userpro

		self.userprofiles = users

		# for k,v in self.userprofiles.items():
		# 	print( str(k) + " :::::   " + str(v))

	def algorithm(self):

		knum = 5 # number of clusters 

		points = self.points

		u = self.userprofiles

		ll = u.keys()

		rl = random.sample(ll,5)

		c = [u[rl[0]],u[rl[1]],u[rl[2]],u[rl[3]],u[rl[4]]]
		self.centroids = c


		t = 10

		while (t>0):

			# assign users levels

			for k,v in u.items():

				cc = self.centroids

				sim = [usersim(v,cc[0]), usersim(v,cc[1]), usersim(v,cc[2]), usersim(v,cc[3]), usersim(v,cc[4])]

				mini = min(sim)

				lev = sim.index(mini)

				# print(lev)

				self.predicted[k] = lev


			# update clusters

			g = [[],[],[],[],[]]

			predict = self.predicted

			for k,v in predict.items():
				prelist = g[v]
				prelist.append(self.userprofiles[k])
				g[v] = prelist

			newcc = []
			for ii in g:
				newcc.append(dictavg(ii))

			self.centroids = newcc

			# print(self.centroids[0])
			# print("..........................................")
			# print(newcc[0])
			t -= 1



if __name__ == '__main__':

	filename = 'C:/Users/anik/GEM/ratings-small.csv'
	points = get_data(filename)
	# for i in points:
	# 	print(i)


	# print("user similarity......")
	# um = UserMatrix(points)
	# umsim = um.similarity(40,41)
	# print (umsim)


	# print("Movie similarity .....")
	# mm = MovieMatrix(points)
	# mmsim = mm.similarity(165,435)
	# print (mmsim)


	kmeans = Kmeans(points)
	kmeans.algorithm()

	lev = kmeans.predicted

	b = numpy.zeros(shape=(1000,1000))
	i=0
	for p in points:
	    mid = p[1]
	    uid = p[0]
	    if (mid < 100 and uid <100):
	        b[uid,mid] = lev[uid]
	    if i > 100:
	        break
	    print(i)
	    i += 1


	fig, ax = plt.subplots(figsize=(100, 100))
	colormap = sns.diverging_palette(220, 10, as_cmap=True)
	sns.heatmap(b, cmap=colormap, annot=True, fmt=".2f")
