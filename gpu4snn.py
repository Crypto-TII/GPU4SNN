### Accelerating Spike Propagation for GPU-based Spiking Neural Network Simulations

### The reference code is taken from the state-of-the-art (SOTA) implementataion:
### SOTA : "Dynamic parallelism for synaptic updating in GPU-accelerated spiking neural network simulations"
### Paper: https://www.sciencedirect.com/science/article/pii/S0925231218304168
### Code: https://bitbucket.org/bkasap/dynamicparallelismsnn/src/master/

### This repository includes SOTA (AP, N, S) algorithms and modified (AB, SKL) algorithms.

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, argparse, csv
import re

os.system("make clean")
os.system("make rmdata")
os.system("make all")

nvidia_gpu = sys.argv[1] #'Quadro'  #GPU name
os.system("rm -rf %s" %(nvidia_gpu))

os.system("mkdir %s"%(nvidia_gpu))
iterations = sys.argv[2] #'2000'  # number of iterations

N = 2500
Nsyn = 1000

os.system("./gpu4snn %s %s %s"%(N, Nsyn, iterations))

Regisme = 0
Algo = 0

#path = './' + nvidia_gpu + '/' + iterations + '/' +'/Results/'
path = './' + nvidia_gpu + '/' + iterations
os.system("mkdir %s"%(path))
os.system("mv ./Results %s"%(path))

path = './' + nvidia_gpu + '/' + iterations + '/Results/'
dataframes_list = []
mode0 = []
mode1 = []
mode2 = []
X = []
Y = []
exclude_algo =8 #5
positions = [311, 312, 313]
colors = ['b', 'g', 'r']

data = dict()
spikes_count = dict()
data_i = dict()

#fig, ax = plt.subplots(figsize=(12,12))
width1 = .5
height1 = 0

width2 = 18
height2 = 26

fontsize2use = 26

dpi1 = 80

class Switcher(object):
    def numbers_to_algos(self, argument):
        """Dispatch method"""
        method_name = 'algo_' + str(argument)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid algo")
        # Call the method as we return it
        return method()
 
    def algo_0(self):
        return "AP"
    
    def algo_1(self):
        return "N"
 
    def algo_2(self):
        return "S"
 
    def algo_3(self):
        return "AB"
        
    def algo_4(self):
        return "SKL"

a=Switcher()

for subdir, dirs, files in os.walk(path):
	#print(subdir, dirs, files)
	#print(subdir)
	if(len(dirs)>0):
		data = dict()
		col_list = ["N", "elapsed(ms)", "elapsedwithdata(ms)", "totalspkcount",  ]
		#print(dirs)
		dirs =  sorted(dirs)
		#fig = plt.figure(figsize=(15,10))

		print(dirs)
		
		######################## for N spikes ###############################
		for i in dirs:			
			path1 = path+str(i) +'/sim_overview.csv'
			temp = re.findall(r'\d+', i)
			res = list(map(int, temp))
			print("\n The Mode is : " + "Mode_"+str(res[2])+"Alg_"+str(res[3]))
			with open(path1, 'r', encoding ='UTF-8', newline = '') as csvfile:				
				reader = csv.DictReader(csvfile, delimiter=' ')
				#print (reader)
				#reader = csv.DictReader(csvfile, delimiter=' ', skiprows=[1])
				reader =  pd.read_csv(csvfile, delimiter=' ', usecols=col_list).to_dict(orient="records")
				#print (reader)
				for row in reader:
					print(row['N'], row['elapsed(ms)'], row['elapsedwithdata(ms)'], row['totalspkcount'])
					#if(res[2]==2):
					#X = list("_Algo_"+str(res[3]))
					algo = "Mode_"+str(res[2])
					if algo not in data:
						data[algo] = []
					if(res[3]<exclude_algo):   #a.numbers_to_algos(1)
						data[algo].append((a.numbers_to_algos(res[3]), float(row['totalspkcount'])))
						#data[algo].append(("Alg_"+str(res[3]), float(row['totalspkcount'])))
					#print("\n")							
					
		# Plot the subgraphs
		plt.subplots(figsize=(width2,height2))
		for i, l in enumerate(data.keys()):
			plt.subplot(positions[i])			
			data_i = dict(data[l])
			#data_i = [float(line[1]) for line in data_i.values()]
			#plt.figure(figsize=(12,12))
			plt.xticks(fontsize=fontsize2use)  
			plt.yticks(fontsize=fontsize2use)    
			bars = plt.bar(data_i.keys(), data_i.values(), color=colors[i], width=width1)
			j=0
			for bar in bars:
    				yval = bar.get_height()
    				#print (list(data_i.keys())[j])
    				plt.text(bar.get_x(), yval+height1, data_i[str(list(data_i.keys())[j])], fontsize=fontsize2use)
    				j=j+1
			plt.xlabel(l, fontsize=fontsize2use)
			plt.ylabel("Total Spike Count", fontsize=fontsize2use)
			plt.title("Total Spike Count by Each Approach", fontsize=fontsize2use)
		# Show the plots
		plt.tight_layout()
		plt.ticklabel_format(useOffset=False, style='plain', axis='y')
		plt.savefig(path+'totalspkcount.pdf', dpi=dpi1, bbox_inches='tight')
		plt.show()
		
		spikes_count = data	
		
		######################## simulation time ###############################
		for i in dirs:			
			path1 = path+str(i) +'/sim_overview.csv'
			temp = re.findall(r'\d+', i)
			res = list(map(int, temp))
			#print("The numbers list is : " + "Mode_"+str(res[2])+"_Algo_"+str(res[3]))
			with open(path1, 'r', encoding ='UTF-8', newline = '') as csvfile:
				#reader = csv.DictReader(csvfile, delimiter=' ')
				#reader = csv.DictReader(csvfile, delimiter=' ', skiprows=[1])
				reader =  pd.read_csv(csvfile, delimiter=' ', usecols=col_list).to_dict(orient="records")
				for row in reader:
					#print(row['N'], row['elapsed(ms)'], row['elapsedwithdata(ms)'], row['totalspkcount'])
					#if(res[2]==2):
					#X = list("_Algo_"+str(res[3]))
					algo = "Mode_"+str(res[2])
					if algo not in data:
						data[algo] = []
					if(res[3]<exclude_algo):   #a.numbers_to_algos(res[3]),
						data[algo].append((a.numbers_to_algos(res[3]), float(row['elapsed(ms)'])))	
						#data[algo].append(("Alg_"+str(res[3]), float(row['elapsed(ms)'])))			
					
		# Plot the subgraphs
		plt.subplots(figsize=(width2,height2))
		for i, l in enumerate(data.keys()):
			plt.subplot(positions[i])			
			data_i = dict(data[l])			
			#data_i = [float(line[1]) for line in data_i.values()]
			plt.xticks(fontsize=fontsize2use)  
			plt.yticks(fontsize=fontsize2use)    
			bars = plt.bar(data_i.keys(), data_i.values(), color=colors[i], width=width1)
			j=0
			for bar in bars:
    				yval = bar.get_height()
    				#print (list(data_i.keys())[j])
    				plt.text(bar.get_x(), yval+height1, dict(spikes_count[l])[str(list(data_i.keys())[j])], fontsize=fontsize2use)
    				j=j+1
			plt.xlabel(l, fontsize=fontsize2use)
			plt.ylabel("Time (ms)", fontsize=fontsize2use)
			plt.title("Elapsed Time (ms) in Kernel Invocations from CPU", fontsize=fontsize2use)
			#plt.title("Elapsed Time (ms) without Data Transfers between CPU and GPU", fontsize=fontsize2use)

		# Show the plots
		plt.tight_layout()
		plt.savefig(path+'elapsed.pdf', dpi=dpi1, bbox_inches='tight')
		plt.show()
		
		
		######################## without data transfer ###############################
		for i in dirs:			
			path1 = path+str(i) +'/sim_overview.csv'
			temp = re.findall(r'\d+', i)
			res = list(map(int, temp))
			#print("The numbers list is : " + "Mode_"+str(res[2])+"Alg_"+str(res[3]))
			with open(path1, 'r', encoding ='UTF-8', newline = '') as csvfile:
				#reader = csv.DictReader(csvfile, delimiter=' ')
				#reader = csv.DictReader(csvfile, delimiter=' ', skiprows=[1])
				reader =  pd.read_csv(csvfile, delimiter=' ', usecols=col_list).to_dict(orient="records")
				for row in reader:
					#print(row['N'], row['elapsed(ms)'], row['elapsedwithdata(ms)'], row['totalspkcount'])
					#if(res[2]==2):
					#X = list("_Algo_"+str(res[3]))
					algo = "Mode_"+str(res[2])
					if algo not in data:
						data[algo] = []
					if(res[3]<exclude_algo): #a.numbers_to_algos(res[3]),
						data[algo].append((a.numbers_to_algos(res[3]), float(row['elapsedwithdata(ms)'])))	
						#data[algo].append(("Alg_"+str(res[3]), float(row['elapsedwithdata(ms)'])))							
					
		# Plot the subgraphs
		plt.subplots(figsize=(width2,height2))
		for i, l in enumerate(data.keys()):
			plt.subplot(positions[i])			
			data_i = dict(data[l])
			#data_i = [float(line[1]) for line in data_i.values()]
			plt.xticks(fontsize=fontsize2use)  
			plt.yticks(fontsize=fontsize2use)    
			bars = plt.bar(data_i.keys(), data_i.values(), color=colors[i], width=width1)
			j=0
			for bar in bars:
    				yval = bar.get_height()
    				#print (list(data_i.keys())[j])
    				plt.text(bar.get_x(), yval+height1, dict(spikes_count[l])[str(list(data_i.keys())[j])], fontsize=fontsize2use)
    				j=j+1
			plt.xlabel(l, fontsize=fontsize2use)
			plt.ylabel("Total Time (ms)", fontsize=fontsize2use)
			plt.title("Total Elapsed Time (ms) with Data Transfers between CPU and GPU", fontsize=fontsize2use)
		# Show the plots
		plt.tight_layout()
		plt.savefig(path+'elapsedwithdata.pdf', dpi=dpi1, bbox_inches='tight')
		plt.show()
