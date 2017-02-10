import sys


def normalize():
#	filelist = ['common_neigh_tr.txt', 'common_neigh_te.txt', 'adamic_adar_tr.txt', 'adamic_adar_te.txt', 'shortest_path_tr.txt', 'shortest_path_te.txt', 'common_spot_tr.txt', 'common_spot_te.txt', 'clustering_coeff_tr.txt', 'clustering_coeff_te.txt', 'friends_measure_tr.txt', 'friends_measure_te.txt', 'katzB_tr.txt', 'katzB_te.txt', 'mean_distance_tr.txt', 'mean_distance_te.txt', 'dist_common_spot_tr.txt', 'dist_common_spot_te.txt', 'check_common_spot_tr.txt', 'check_common_spot_te.txt','jaccard_neigh_tr.txt', 'jaccard_neigh_te.txt', 'cosine_common_spot_tr.txt', 'cosine_common_spot_te.txt', 'check_cosine_common_spot_tr.txt', 'check_cosine_common_spot_te.txt', 'prob_common_spot_tr.txt', 'prob_common_spot_te.txt', 'common_time_spot_tr.txt', 'common_time_spot_te.txt', 'dist_common_spot_tr.txt', 'dist_common_spot_te.txt', 'check_common_time_spot_tr.txt', 'check_common_time_spot_te.txt', 'common_crt_time_spot_tr.txt', 'common_crt_time_spot_te.txt']
	filelist = [ 'adamic_adar_entropy_tr.txt', 'adamic_adar_entropy_te.txt', 'spot_product_tr.txt', 'spot_product_te.txt', 'weight_common_spot_tr.txt', 'weight_common_spot_te.txt', 'min_entropy_tr.txt', 'min_entropy_te.txt']
	data_list = []
	for filename in filelist:
		data_list[:] = []
		val_max = -10000000
		val_min = 10000000
		with open(filename, 'r') as f:
			for line in f:
				u1, u2, val = line.strip().split()
				u1 = int(u1)
				u2 = int(u2)
				val = float(val)
				if val > val_max:
					val_max = val
				if val < val_min:
					val_min = val
				data_list.append([u1,u2,val])
		normal = val_max - val_min
		for data in data_list:
			data[2] /= normal
		newname = filename.split('.')[0] + '_nol.txt'
		with open(newname, 'w') as f:
			for data in data_list:
				print(data[0], data[1], data[2], file=f)
		print('normalize', filename, 'completely')



def libSVM_feat():
	featlist = ['jaccard_spot', 'check_common_spot', 'cosine_common_spot', 'check_common_time_spot', 'common_crt_time_spot', 'dist_common_spot', 'weight_common_spot', 'check_cosine_common_spot', 'spot_product', 'adamic_adar_entropy', 'min_entropy', 'katzB']
	sample_data = {'tr':[], 'te':[]}
	feat_dict = {'tr':dict(), 'te':dict()}
	temp_list = []
	with open('Gowalla_new/link_prediction/gowalla.train.txt', 'r') as f:
		for line in f:
			u1, u2, y = line.strip().split()
			u1 = int(u1)
			u2 = int(u2)
			y = int(y)
			sample_data['tr'].append((u1, u2, y))
	with open('Gowalla_new/link_prediction/gowalla.test.txt', 'r') as f:
		for line in f:
			u1, u2, y = line.strip().split()
			u1 = int(u1)
			u2 = int(u2)
			y = int(y)
			sample_data['te'].append((u1, u2, y))
	for t in ['tr', 'te']:
		for feat in featlist:
			readfile = feat+'_'+t+'.txt'
			with open(readfile, 'r') as f:
				print('open', readfile)
				for line in f:
#					print(line.strip().split())
					u1, u2, v = line.strip().split()
					u1 = int(u1)
					u2 = int(u2)
					v = float(v)
					if (u1, u2) not in feat_dict[t]:
						feat_dict[t][(u1, u2)] = []
					feat_dict[t][(u1, u2)].append(v)
		writefile = 'multifeat'+'_'+t+'.txt'
		with open(writefile, 'w') as f:
			feat_len = len(featlist)
			for u1, u2, y in sample_data[t]:
				print(y, end=' ', file=f)
				for i in range(1, feat_len+1):
#					print(i)
					if feat_dict[t][(u1, u2)][i-1] == 0.0:
						continue
					print(str(i)+':'+str(feat_dict[t][(u1, u2)][i-1]), end=' ', file=f)
				print('', end='\n', file=f)
		


def Matrix_feat():
	featlist = ['jaccard_spot', 'check_common_spot', 'cosine_common_spot', 'check_common_time_spot', 'common_crt_time_spot', 'dist_common_spot', 'weight_common_spot', 'check_cosine_common_spot', 'spot_product', 'adamic_adar_entropy', 'min_entropy', 'katzB']
	sample_data = {'tr':[], 'te':[]}
	feat_dict = {'tr':dict(), 'te':dict()}
	temp_list = []
	with open('Gowalla_new/link_prediction/gowalla.train.txt', 'r') as f:
		for line in f:
			u1, u2, y = line.strip().split()
			u1 = int(u1)
			u2 = int(u2)
			y = int(y)
			sample_data['tr'].append((u1, u2, y))
	with open('Gowalla_new/link_prediction/gowalla.test.txt', 'r') as f:
		for line in f:
			u1, u2, y = line.strip().split()
			u1 = int(u1)
			u2 = int(u2)
			y = int(y)
			sample_data['te'].append((u1, u2, y))
	for t in ['tr', 'te']:
		for feat in featlist:
			readfile = feat+'_'+t+'.txt'
			with open(readfile, 'r') as f:
				print('open', readfile)
				for line in f:
#					print(line.strip().split())
					u1, u2, v = line.strip().split()
					u1 = int(u1)
					u2 = int(u2)
					v = float(v)
					if (u1, u2) not in feat_dict[t]:
						feat_dict[t][(u1, u2)] = []
					feat_dict[t][(u1, u2)].append(v)
		writefile = 'multifeat'+'_'+t+'.txt'
		with open(writefile, 'w') as f:
			feat_len = len(featlist)
			for u1, u2, y in sample_data[t]:
				for i in range(feat_len):
#					print(i)
					print(feat_dict[t][(u1, u2)][i], end=' ', file=f)
				print('', end='\n', file=f)


if __name__ == '__main__':
	Matrix_feat()
