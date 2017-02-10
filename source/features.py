import sys
import networkx as nx
import itertools as itertl
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Manager
from math import log
from math import sqrt

import exploitation as explt


## Topology feature   ##
def common_neigh(pair, G, record):
	assert len(pair) == 2
	x, y = pair
	N_x = G.neighbors(x)
	N_y = G.neighbors(y)
	N_common = set(N_x) & set(N_y)
	score = len(N_common)
	if record != None:
		record[pair] = score
	return score


def jaccard_coeff(pair, G, record):
	assert len(pair) == 2
	x, y = pair
	N_x = G.neighbors(x)
	N_y = G.neighbors(y)
	num_union = len( set(N_x) | set(N_y) )
	num_inter = len( set(N_x) & set(N_y) )
	score = None
	if num_union != 0:
		score = num_inter/num_union
	else:
		score = None
	if record != None:
		record[pair] = score
	return score


# notice that remove the exist pair to get the better result
def shortest_path(pair, G, record):
	assert len(pair) == 2
	x, y = pair
	score = None
	path_len = 0
	try:
		path_len = nx.shortest_path_length(G, x, y)
	except:
		score = None
	if path_len != 0:
		score = 1/path_len
	if record != None:
		record[pair] = score
	return score


def Adamic_Adar(pair, G, record):
	assert len(pair) == 2
	x, y = pair
	N_x = G.neighbors(x)
	N_y = G.neighbors(y)
	N_inter = set(N_x) & set(N_y)
	score = None
	if len(N_inter) != 0:
		score = 0.0
		for z in N_inter:
			N_num = len(G.neighbors(z))
			score += 1/log(N_num)
	if record != None:
		record[pair] = score
	return score


def prefer_attach(pair, G, record):
	assert len(pair) == 2
	x, y = pair
	N_x = G.neighbors(x)
	N_y = G.neighbors(y)
	score = len(N_x)*len(N_y)
	if record != None:
		record[pair] = score
	return score


def cluster_coeff(pair, G, record):
	assert len(pair) == 2
	CC = [0 for v in pair]
	score = 1.0
	for p in range(len(pair)):
		num_tri = 0
		N = G.neighbors(pair[p])
		num_N = len(N)
		if num_N > 1:
			for i in range(num_N):
				for j in range(i+1, num_N):
					if G.has_edge(N[i], N[j]):
						num_tri += 1
			CC[p] = num_tri / (num_N*(num_N-1)/2)
			score *= CC[p]
		else:
			CC[p] = None
			score = None
			break
	if record != None:
		record[pair] = score
	return score


# approximate katzB #
def friends_measure(pair, G, record):
	assert len(pair) == 2
	x, y = pair
	N_x = G.neighbors(x)
	N_y = G.neighbors(y)
	score = 0
	for u in N_x:
		for v in N_y:
			if u == v or G.has_edge(u,v):
				score += 1
	if record != None:
		record[pair] = score
	return score


# topological with location
# check_dict: { user:{locations} }
def common_spot(pair, check_dict, record):
	assert len(pair) == 2
	x, y = pair
	P_x = set([p for p in check_dict[x]])
	P_y = set([p for p in check_dict[y]])
	P_common = P_x & P_y
	score = len(P_common)
	if record != None:
		record[pair] = score
	return score


def jaccard_spot(pair, check_dict, record):
	assert len(pair) == 2
	x, y = pair
	P_x = set([p for p in check_dict[x]])
	P_y = set([p for p in check_dict[y]])
	num_union = len( P_x | P_y )
	num_inter = len( P_x & P_y )
	score = None
	if num_union != 0:
		score = num_inter/num_union
	else:
		score = None
	if record != None:
		record[pair] = score
	return score


def spot_product(pair, check_dict, record):
	assert len(pair) == 2
	x, y = pair
	score = 0.0
	P_x = sum([check_dict[x][l][0] for l in check_dict[x]])
	P_y = sum([check_dict[y][l][0] for l in check_dict[y]])
	score = P_x*P_y
	if record != None:
		record[pair] = score
	return score


# check-in MF data is too big to load in... 
def common_spotMF(pair, check_dict, record):
	assert len(pair) == 2
	x, y = pair
	score = 0.0
	P_x = {l:check_dict[x][l][0] for l in check_dict[x]}
	P_y = {l:check_dict[x][l][0] for l in check_dict[y]}
	with open('check_pair_te.txt', 'r') as f1:
		with open('check_pair_pred.txt', 'r') as f2:
			for line1 in f1:
				line2 = f2.readline()
				u, l, s = line1.strip().split()
				u = int(u)
				l = int(l)
				if x == u and l not in P_x:
					p = float(line2.strip())
					P_x[l] = p
				if y == u and l not in P_y:
					p = float(line2.strip())
					P_y[l] = p
	if set(P_x.keys()) != set(P_y.keys()):
		print(x, 'and', y, 'predicted check-in data unequal')
	P_union = set(P_x.keys()) | set(P_y.keys())
	for l in P_union:
		if l in P_x and l in P_y:
			score += P_x[l]*P_y[l]
	if record != None:
		record[pair] = score
	return score



def geo_distance(pair, usergeo_dict, record):
	assert len(pair) == 2
	x, y = pair
	if x in usergeo_dict:
		x0, x1 = usergeo_dict[x][0], usergeo_dict[x][1]
	else:
		x0, x1 = usergeo_dict['mean'][0], usergeo_dict['mean'][1]
	if y in usergeo_dict:
		y0, y1 = usergeo_dict[y][0], usergeo_dict[y][1]
	else:
		y0, y1 = usergeo_dict['mean'][0], usergeo_dict['mean'][1]
	score = sqrt((x0-y0)**2 + (x1-y1)**2)
	if record != None:
		record[pair] = score
	return score


def geo_cluster_coeff():
	pass




def common_time_spot(pair, check_dict, record):
	assert len(pair) == 2
	delta_t = 1
	x, y = pair
	spot_inter = set(check_dict[x]) & set(check_dict[y])
	score = 0.0
	for l in spot_inter:
		for time_x in check_dict[x][l][1]:
			for time_y in check_dict[y][l][1]:
				date_x, t_x = time_x.split('T')
				date_y, t_y = time_y.split('T')
				t_x = int(t_x)
				t_y = int(t_y)				
				if date_x == date_y:
					if t_x - t_y < delta_t or t_y - t_x < delta_t:
						score += 1
					elif t_x - t_y == delta_t or t_y - t_x == delta_t:
						score += 0.5
	if record != None:
		record[pair] = score
	return score



# define the critical time which has the higher weight of relationship
def common_crt_time_spot(pair, check_dict, record):
	assert len(pair) == 2
	delta_t = 1
	w_list = [25, 5, 1]
	x, y = pair
	spot_inter = set(check_dict[x]) & set(check_dict[y])
	score = 0.0
	for l in spot_inter:
		for time_x in check_dict[x][l][1]:
			for time_y in check_dict[y][l][1]:
				date_x, t_x = time_x.split('T')
				date_y, t_y = time_y.split('T')
				t_x = int(t_x)
				t_y = int(t_y)
				if date_x == date_y:
					if t_x - t_y < delta_t or t_y - t_x < delta_t:
						if t_x <= 4 or t_x >= 21:
							score += w_list[0]
						elif t_x >= 17 and t_x < 21:
							score += w_list[1]
						elif t_x < 17 and t_x > 4:
							score += w_list[2]
					elif t_x - t_y == delta_t or t_y - t_x == delta_t:
						if t_x <= 4 or t_x >= 21:
							score += w_list[0]*0.5
						elif t_x >= 17 and t_x < 21:
							score += w_list[1]*0.5
						elif t_x < 17 and t_x > 4:
							score += w_list[2]*0.5
	if record != None:
		record[pair] = score
	return score




def prob_common_spot(pair, check_dict, record):
	assert len(pair) == 2
	x, y = pair
	spot_inter = set(check_dict[x]) & set(check_dict[y])
	x_view = sum([check_dict[x][l][0] for l in check_dict[x]])
	y_view = sum([check_dict[y][l][0] for l in check_dict[y]])
	score = 0.0
	for l in spot_inter:
		x_count, date_list = check_dict[x][l]
		y_count, date_list = check_dict[y][l]
		score += (x_count/x_view)*(y_count/y_view)
	if record != None:
		record[pair] = score
	return score


# the rate of common spot on the same time over time interval
def col_location_rate(pair, check_dict, record):
	pass


def weight_common_spot(pair, check_dict, record):
	assert len(pair) == 2
	x, y = pair
	spot_inter = set(check_dict[x]) & set(check_dict[y])
	score = 0.0
	for l in spot_inter:
		score += check_dict[x][l][0] * check_dict[y][l][0]
	if record != None:
		record[pair] = score
	return score


# the cosine similiar of the vectors of viewed spots of two users
def cosine_common_spot(pair, check_dict, record):
	assert len(pair) == 2
	x, y = pair
	spot_union = set(check_dict[x]) | set(check_dict[y])
	x_veclen = 0.0
	y_veclen = 0.0
	score = 0.0
	for l in spot_union:
		if l in check_dict[x]:
			x_count, date_list = check_dict[x][l]
			x_veclen += (x_count**2)
			if l in check_dict[y]:
				y_count, date_list = check_dict[y][l]
				y_veclen += (y_count**2)
				score += (x_count*y_count)
		else:
			y_count, date_list = check_dict[y][l]
			y_veclen += (y_count**2)
	x_veclen = sqrt(x_veclen)
	y_veclen = sqrt(y_veclen)
	score /= (x_veclen*y_veclen)
	if record != None:
		record[pair] = score
	return score


def min_entropy(pair, check_dict, ent_dict, record):
	assert len(pair) == 2
	x, y = pair
	spot_inter = set(check_dict[x]) & set(check_dict[y])
	ent = 0.0
	score = 0.0
	for l in spot_inter:
		if l in ent_dict:
			ent = ent_dict[l]
			if ent < score:
				score = ent
	if record != None:
		record[pair] = score
	return score


def adamic_adar_entropy(pair, check_dict, ent_dict, record):
	assert len(pair) == 2
	x, y = pair
	spot_inter = set(check_dict[x]) & set(check_dict[y])
	score = 0.0
	for l in spot_inter:
		if l in ent_dict:
			score += 1/ent_dict[l]
		else:
			score += 1/ent_dict['mean']
	if record != None:
		record[pair] = score
	return score



# The smaller count of check-ins, the more important of the common spot
def check_common_spot(pair, check_dict, spot_dict, record):
	assert len(pair) == 2
	x, y = pair
	spot_inter = set(check_dict[x]) & set(check_dict[y])
	score = 0.0
	for l in spot_inter:
		if l in spot_dict:
			count = spot_dict[l][0]
			if count == 0 or count == 1:
				score += 1
			else:
				score += log(1/count)
		else:
			score += log(1/spot_dict['mean'][0])
	if record != None:
		record[pair] = score
	return score



def check_cosine_common_spot(pair, check_dict, spot_dict, record):
	assert len(pair) == 2
	x, y = pair
	spot_union = set(check_dict[x]) | set(check_dict[y])
	x_veclen = 0.0
	y_veclen = 0.0
	score = 0.0
	for l in spot_union:
		if l in check_dict[x]:
			x_count, date_list = check_dict[x][l]
			x_veclen += (x_count**2)
			if l in check_dict[y]:
				y_count, date_list = check_dict[y][l]
				y_veclen += (y_count**2)
				if l in spot_dict:
					s_count = spot_dict[l][0]
					if s_count == 0 or s_count == 1:
						score += (x_count*y_count)
					else:
						score += (1/log(s_count)*(x_count*y_count))
				else:
					score += (1/log(spot_dict['mean'][0])*(x_count*y_count))
		else:
			y_count, date_list = check_dict[y][l]
			y_veclen += (y_count**2)
	x_veclen = sqrt(x_veclen)
	y_veclen = sqrt(y_veclen)
	score /= (x_veclen*y_veclen)
	if record != None:
		record[pair] = score
	return score


def check_common_time_spot(pair, check_dict, spot_dict, record):
	assert len(pair) == 2
	x, y = pair
	delta_t = 1
	spot_inter = set(check_dict[x]) & set(check_dict[y])
	score = 0.0
	for l in spot_inter:
		for time_x in check_dict[x][l][1]:
			for time_y in check_dict[y][l][1]:
				date_x, t_x = time_x.split('T')
				date_y, t_y = time_x.split('T')
				t_x = int(t_x)
				t_y = int(t_y)
				if date_x == date_y:
					if t_x - t_y < delta_t or t_y - t_x < delta_t:
						t_weight = 1.0
					elif t_x - t_y == delta_t or t_y - t_x == delta_t:
						t_weight = 0.5
					else:
						t_weight = 0.0
						continue
					if l in spot_dict:
						s_count = spot_dict[l][0]
						if s_count == 0 or s_count == 1:
							score += 1*t_weight
						else:
							score += 1/log(s_count)*t_weight
					else:
						score += 1/log(spot_dict['mean'][0])*t_weight
	if record != None:
		record[pair] = score
	return score




# some spots in check-in data are not in spot info !!!
# The farther place two people go, the more likely they are friends
def dist_common_spot(pair, usergeo_dict, spot_dict, check_dict, record):
	assert len(pair) == 2
	x, y = pair
	if x in usergeo_dict:
		x0, x1 = usergeo_dict[x]
	else:
		x0, x1 = usergeo_dict['mean']
	if y in usergeo_dict:
		y0, y1 = usergeo_dict[y]
	else:
		y0, y1 = usergeo_dict['mean']
	spot_inter = set(check_dict[x]) & set(check_dict[y])
	score = 0.0
	for l in spot_inter:
		if l in spot_dict:
			c, s0, s1 = spot_dict[l]
		else:
			c, s0, s1 = spot_dict['mean']
		x_dist = sqrt((x0-s0)**2 + (x1-s1)**2)
		y_dist = sqrt((y0-s0)**2 + (y1-s1)**2)
		value = x_dist*y_dist
#		if value <= 0:
#			print('dist error: x_dist:', x_dist, 'y_dist', y_dist)
#			sys.exit()
		if value <= 1.0:
			score += 1.0
		else:
			score += log(x_dist*y_dist)
	if record != None:
		record[pair] = score
	return score





# The farther place two people go in the same time. the more likely they are friends
def dist_common_time_spot():
	pass


#  the frequency of viewing spots as the importance of this spots (done: check_)
#  the entrpy of spots (prob. log prob.)
#  the critical meeting time (e.g. midnight or evening...) may be important
#  the common spots on the same time over a time interval

def write_result(filename, record_dict, pair_list):
	with open(filename, 'w') as f:
		for pair in pair_list:
			if record_dict[pair] == None:
				print(pair[0], pair[1], 0.0, file=f)
			else:
				print(pair[0], pair[1], record_dict[pair], file=f)
	print('writing', filename, 'completely')



def features():
	# initialization #
	pool = ThreadPool(3)
	manager = Manager()
	result_dict = manager.dict()
	sample_edges = {'tr':list(), 'te':list()}
	sample_num = {'tr':0, 'te':0}
	G = nx.Graph()
	U = dict()                # user mean lat. lng.
	S = dict()                # spot info.
	C = dict()                # Checkin
	E = dict()                # spot entropy
	'''
	feat_dict = {'common_neigh':comm_neigh,			'jaccard_coeff':jaccard_coeff, 
			'prefer_attach':prefer_attach, 		'shortest_path':shortest_path, 
			'Adamic_Adar':Adamic_Adar, 		'cluster_coeff':cluster_coeff, 
			'friends_measure':friends_measure, 	'common_spot':comm_spot, 
			'jaccard_spot':jaccard_spot,		 'common_time_spot':common_time_spot, 
			'common_crt_time_spot':common_crt_time_spot, 'common_spotMF':common_spotMF, 
			'geo_distance':geo_distance, 		'check_common_spot':check_common_spot, 
			'dist_common_spot':dist_common_spot, 	'prob_common_spot':prob_common_spot, 
			'cosine_common_spot':cosine_common_spot, 'check_cosine_common_spot':check_cosine_common_spot, 
			'geo_culster_coeff':geo_cluster_coeff, 'dist_common_time_spot':dist_common_time_spot,
			'adamic_adar_entropy':adamic_adar_entropy, 'min_entropy':min_entropy}

	argv_dict = {'common_neigh':('T', G, result_dict), 	'jaccard_coeff':('T', G, result_dict), 
			'prefer_attach':('T', G, result_dict), 	'shortest_path':('T', G, result_dict), 
			'Adamic_Adar':('T', G, result_dict), 	'cluster_coeff':('T', G, result_dict), 
			'friends_measure':('T', G, result_dict), 'common_spot':('T', C, result_dict), 
			'jaccard_spot':('T', C, result_dict), 	'common_time_spot':('T', C, result_dict), 
			'common_crt_time_spot':('T', C, result_dict), 'common_spotMF':('T', C, result_dict), 
			'geo_distance':('T', U, result_dict), 	'geo_cluster_coeff':(),  
			'prob_common_spot':('T', C, result_dict), 'cosine_common_spot':('T', C, result_dict), 
			'check_common_spot':('T', C, S, result_dict), 'check_cosine_common_spot':('T', C, S, result_dict), 
			'dist_common_spot':('T', U, S, C, result_dict), 'dist_common_time_spot':(), 'check_common_time_spot':('T', C, S, result_dict)
			'spot_product':('T', C, result_dict),		'adamic_adar_entropy':('T', C, E, result_dict),
			'min_entropy':('T, C, E, result_dict'), 	'weight_common_spot':('T', C, result_dict)}
	'''
#	feat_dict = {'spot_product':spot_product, 'min_entropy':min_entropy, 'weight_common_spot':weight_common_spot}
#	argv_dict = {'spot_product':('T', C, result_dict), 'min_entropy':('T', C, E, result_dict), 'weight_common_spot':('T', C, result_dict)}
	feat_dict = {'common_spotMF':common_spotMF}
	argv_dict = {'common_spotMF':('T', C, result_dict)}
	# load exploited data #
	explt.mkUserGraph(G)
	print("make graph completely")
	explt.mkUserGeo(U)
	print("make user geo completely")
	explt.getSpotInfo(S, 'CLL')
	print("make spot info completely")
	explt.getCheckin(C, 'CLT')
	print("make check info completely")
	explt.mkSpotEnt(E)
	print("make spot entropy completely")
	with open('Gowalla_new/link_prediction/gowalla.train.txt') as f:
		for line in f:
			u1, u2, y = line.strip().split()
			u1 = int(u1)
			u2 = int(u2)
			sample_edges['tr'].append((u1,u2))
	sample_num['tr'] = len(sample_edges['tr'])
	with open('Gowalla_new/link_prediction/gowalla.test.txt') as f:
		for line in f:
			u1, u2, y = line.strip().split()
			u1 = int(u1)
			u2 = int(u2)
			sample_edges['te'].append((u1,u2))
	sample_num['te'] = len(sample_edges['te'])
	print("load sample data completely")
	# generate features #
	for f in feat_dict:
		for t in ['tr', 'te']:
			result_dict.clear()
			argvs = None
			argv = argv_dict[f]
			argv_len = len(argv_dict[f])
			if f == 'shortest_path':                     # Because of removing and overflowing memory..., cannot parallize
				print('generate', f, '...')
				for pair in sample_edges[t]:
					if G.has_edge(pair[0], pair[1]):
						G.remove_edge(pair[0], pair[1])
						shortest_path(pair, G, result_dict)
						G.add_edge(pair[0], pair[1])
					else:
						shortest_path(pair, G, result_dict)
				write_result(f+'_'+t+'.txt', result_dict, sample_edges[t])
				continue
			elif f == 'common_spotMF':
				print('gererate', f, '...')
				for pair in sample_edges[t]:
					common_spotMF(pair, C, result_dict)
				write_result(f+'_'+t+'.txt', result_dict, sample_edges[t])
				continue
			if argv_len == 3:
				argvs = itertl.zip_longest(sample_edges[t], itertl.repeat(argv[1], sample_num[t]), itertl.repeat(result_dict, sample_num[t]))
			elif argv_len == 4:
				argvs = itertl.zip_longest(sample_edges[t], itertl.repeat(argv[1], sample_num[t]), itertl.repeat(argv[2], sample_num[t]), itertl.repeat(result_dict, sample_num[t]))
			elif argv_len == 5:
				argvs = itertl.zip_longest(sample_edges[t], itertl.repeat(argv[1], sample_num[t]), itertl.repeat(argv[2], sample_num[t]), itertl.repeat(argv[3], sample_num[t]), itertl.repeat(result_dict, sample_num[t]))
			else:
				print('unknown argv length')
				sys.exit()
			print('generate', f, '...')
			pool.starmap(feat_dict[f], argvs)
			write_result(f+'_'+t+'.txt', result_dict, sample_edges[t])
	
	
if __name__ == '__main__':
	features()
