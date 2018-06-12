import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
	
def plot_data(x_coord, y_coord, y_hat, sort_index):
	fig = plt.figure()
	splt = fig.add_subplot(1, 1, 1)
	splt.plot(x_coord, y_hat[sort_index], c = 'black')
	splt.scatter(x_coord, y_coord[sort_index], s = 20, c = 'black', alpha = 0.5)
	plt.title('Data')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

def locally_weighted(x_mat, y_mat, k = 1.0):
	m_x_mat = np.shape(x_mat)[0]
	weights = []
	bs = []
	for i in range(m_x_mat):
		weights.append(np.mat(np.eye(m_x_mat)))
		for j in range(m_x_mat):
			diff_mat = x_mat[i, :]-x_mat[j, :]
			weights[i][j, j] = np.exp(diff_mat*diff_mat.T/(-2.0*k**2))
		bs.append((x_mat.T*weights[i]*x_mat).I*(x_mat.T*weights[i]*y_mat))
	return bs

def reg_data(test_array, x_array, y_array):
	x_mat = np.mat(x_array)
	y_mat = np.mat(y_array).T
	m_x_mat = np.shape(x_mat)[0]
	y_hat = np.zeros(m_x_mat)
	bs = locally_weighted(x_mat, y_mat, 0.05)
	for i in range(m_x_mat):
		y_hat[i] = x_mat[i, :]*bs[i]
	#corr = np.corrcoef(y_hat.T, y_mat.T)
	sort_index = x_mat[:, 1].argsort(0)
	plot_data(x_mat[:,1].flatten().A[0].sort(), y_mat.flatten().A[0], y_hat, sort_index)
	return y_hat

def init_data(df):
	x_array = []
	y_array = []
	for i in df.index:
		x_array.append([df.iloc[i, 0],df.iloc[i, 1]])
		y_array.append(df.iloc[i, 2])
	#plot_data(x_coord, y_coord)
	reg_data(x_array, x_array, y_array)
	return x_array, y_array

if __name__ == '__main__':
	data = pd.read_csv('data.txt', header = None, sep = '\t')
	df = pd.DataFrame(data)
	init_data(df)