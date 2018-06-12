import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
	
def plot_data(x_coord, y_coord, y_hat, corr):
	print(corr)
	fig = plt.figure()
	splt = fig.add_subplot(1, 1, 1)
	splt.plot(x_coord, y_hat, '--', c = 'black')
	splt.scatter(x_coord, y_coord, s = 20, c = 'black', alpha = 0.5)
	plt.title('Data')
	plt.xlabel('x\n')
	plt.ylabel('y')
	#plt.show()

def reg_data(x_array, y_array):
	x_mat = np.mat(x_array)
	y_mat = np.mat(y_array).T
	b_mat = (x_mat.T*x_mat).I*(x_mat.T*y_mat)
	y_hat = (x_mat*b_mat).flatten().A[0]
	#corr = np.corrcoef(y_hat.T, y_mat.T)
	plot_data(x_mat[:,1].flatten().A[0], y_mat.flatten().A[0], y_hat, corr)

def init_data(df):
	x_array = []
	y_array = []
	for i in df.index:
		x_array.append([df.iloc[i, 0],df.iloc[i, 1]])
		y_array.append(df.iloc[i, 2])
	#plot_data(x_coord, y_coord)
	reg_data(x_array, y_array)

if __name__ == '__main__':
	data = pd.read_csv('data.txt', header = None, sep = '\t')
	df = pd.DataFrame(data)
	init_data(df)