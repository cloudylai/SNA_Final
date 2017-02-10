import sys

def acc(pred_file, rlt_file):
	pred_list = []
	rlt_list = []
	with open(pred_file, 'r') as f:
		for line in f:
			u1, u2, y = line.strip().split()
			u1 = int(u1)
			u2 = int(u2)
			y = int(y)
			pred_list.append((u1,u2,y))
	with open(rlt_file, 'r') as f:
		for line in f:
			u1, u2, y = line.strip().split()
			u1 = int(u1)
			u2 = int(u2)
			y = int(y)
			rlt_list.append((u1,u2,y))
	pred_num = len(pred_list)
	rlt_num = len(rlt_list)
	acc_val = 0.0
	if pred_num != rlt_num:
		print("Error: numbers of data are different")
		sys.exit()
	for i in range(pred_num):
		if pred_list[i][0] != rlt_list[i][0] or pred_list[i][1] != rlt_list[i][1]:
			print('Error: users are different')
			sys.exit()
		else:
			if pred_list[i][2] == rlt_list[i][2]:
				acc_val += 1
	acc_val /= pred_num
	print("Accuracy:", acc_val)
	return acc


if __name__ == '__main__':
	acc(sys.argv[1], sys.argv[2])
