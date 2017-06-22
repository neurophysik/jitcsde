#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import sem
import warnings

def KMC(Xs, h, kmax=4, nbins=100):
	"""
	Estimates the Kramers–Moyal coefficients from a time series
	
	Parameters
	----------
	Xs : iterable of floats
		the time series
	
	h : number
		the sampling interval
	
	kmax : integer
		the number of Kramers–Moyal coefficients to be computed.
	
	nbins : integer
		the number of bins in X space
	
	Returns
	-------
	
	A list containing:
	
	bins : 1-dimensional NumPy array of length `nbins`
	the centres of the bins in X space
	
	kmc_1 : pair of 1-dimensional NumPy arrays of length `nbins`
	the 1st Kramers–Moyal coefficients and their confidence interval (as estimated wia the standard error of the mean)
	
	kmc_2 : pair of 1-dimensional NumPy arrays of length `nbins`
	the 2nd Kramers–Moyal coefficients and their confidence interval (as estimated wia the standard error of the mean)
	
	[…]
	"""
	Xs = list(Xs)
	N = len(Xs)
	
	# range of X for statistics
	Min = min(Xs) - (max(Xs)-min(Xs))/nbins*0.5
	Max = max(Xs) + (max(Xs)-min(Xs))/nbins*0.5
	# bin borders
	X_bins = np.linspace(Min,Max,nbins+1)
	# intialise results with middles of bins:
	results = [(X_bins[1:]+X_bins[:-1])/2]
	
	def Bin(value):
		return int((value-Min)/(Max-Min)*nbins)
	
	# list of nbins−1 empty lists
	data = [ [] for _ in range(nbins) ]
	
	# collect increments:
	for j in range(N-1):
		increment = Xs[j+1]-Xs[j]
		b = Bin(Xs[j])
		data[b].append(increment)
	
	# convert to arrays to make the following easier:
	for i in range(nbins):
		data[i] = np.array(data[i])
	
	# add KMCs and their standard errors to results:
	for k in range(1,kmax+1):
		with np.errstate(divide='ignore',invalid='ignore'):
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", message="Mean of empty slice.")
				warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
				means = np.array([ np.mean(data[i]**k)/h for i in range(nbins) ])
				sems = np.array([ sem(data[i]**k)/h for i in range(nbins) ])
		results.append((means,sems))
	
	return results
