% Files %
trainfile = 'Gowalla/link_prediction/train_and_test/gowalla.train.txt';
testfile = 'Gowalla/link_prediction/train_and_test/gowalla.test.txt';

delimiter = '\t';
headerlines = 0;

% import data %
trE = importdata(trainfile, delimiter, headerlines);
teE = importdata(testfile, delimiter, headerlines);
usersize = 2919325+1;
fprintf('load edges completely\n');

temp1 = trE(:,1);
temp2 = teE(:,2);
row = [temp1+1; temp2+1];
col = [temp2+1; temp1+1];

clear temp1;
clear temp2;

beta = 0.005;
A = sparse(row, col, 1, usersize, usersize);
clear row;
clear col;
I = speye(usersize);
Score = (I-beta*A)\I - I;
fprintf('calculate katzB completely');

% output file %
fid = -1;
fid = fopen('katzB_tr.txt', 'w');
if fid == -1,
	fprintf('cannot open file katzB_tr.txt\n');
	quit cancel;
end
for k =1:size(trE,1),
	u1 = trE.data(k,1);
	u2 = trE.data(k,2);
	fprintf(fid, '%d %d ', u1, u2);
	value = nonzeros(Score(u1+1, u2+1));
	fprintf(fid, '%g\n', value);
end
fclose(fid);
fid = -1;
fid = fopen('katzB_te.txt', 'w');
if fid == -1,
	fprintf('cannot open file katzB_te.txt', 'w');
	quit cancel;
end
for k = 1:size(teE,1),
	u1 = teE.data(k,1);
	u2 = teE.data(k,2);
	fprintf(fid, '%d %d ', u1, u2);
	if Score(u1+1, u2+1) == 0.0,
		fprintf(fid, '0.0\n');
	else
		s = nonzeros(Score(u1+1, u2+1));
		fprintf(fid, '%g\n', s);
	end
end
fclose(fid);
