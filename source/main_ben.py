import sys
import networkx as nx
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier as etc
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.ensemble import BaggingClassifier as bc
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
#ALL NODE AND LABELS ARE "STRINGS"

#read traning pairs
def read_train_file(filename):
    L_train = []#training pairs, a list of tuples
    #user_u1, user_v1
    #user_u2, user_v2
    #....
    Y_train = []
    with open(filename, "r") as f:
        for line in f:
            (u, v, label) = line.split()
            L_train.append((u, v))
            Y_train.append(label)
            if label == "1":#has egde
                G.add_edge(u, v)
            else:
                G.add_node(u)
                G.add_node(v)
    return (L_train, Y_train)

def read_test_file(filename):
    L_test = []#testing pairs, a list of tuple
    #user_u1, user_v1
    #user_u2, user_v2
    #....
    Y_test = []
    with open(filename, "r") as f:
        for line in f:
            (u, v, label) = line.split()
            L_test.append((u, v))
            Y_test.append(label)
            #may have new nodes not in training
            G.add_node(u)
            G.add_node(v)
    return ( L_test, Y_test)


def read_users_info(filename):
    with open(filename) as f:
        for line in f:
            u = line.split()[0]
            U[u] = {} 
            U[u]['follow_cnt'] = int(line.split()[-1])#follower numer


def read_spots_info(filename):
    with open(filename) as f:
        for line in f:
            line = line.split()
            spot = "s_"+line[0]
            S[spot] = {} #spot ID
            S[spot]['cate'] = line[2]#category
            S[spot]['check_cnt'] = int(line[4])#checkin_cnt
            S[spot]['LL'] = (float(line[6]), float(line[8]))#lat and lng

def read_checkins_info(filename):
    with open(filename) as f:
        for line in f:
            line = line.split()
            u = line[0]
            C[u] = []
            for info in line[1:]:
                t = info.split("T")[0]
                spot = "s_"+info.split(":")[-1]
                C[u].append((t, spot))
                B.add_edge(u,spot)
                HB.add_edge(u,spot)
                H.add_edge(u,spot)
def add_H_edges():
    D = {s: S[s]['LL'] for s in S}
    Index = sorted(D, key = D.get)
    cnt = 0
    M = len(Index)
    for i in range( M - 1):
        j = i + 1
        s1 = Index[i]
        s2 = Index[j]
        dist = 0
        while dist < 0.1 and j < M: 
            dist = np.sqrt( (S[s1]['LL'][0]-S[s2]['LL'][0])**2 + (S[s1]['LL'][1]-S[s2]['LL'][1])**2)
            H.add_edge(s1, s2)
            j+= 1
            if j < M:
                s2 = Index[j]

def calculate_mean_LL():
    nnz_lat = 0
    nnz_lng = 0
    nnz_cnt = 0
    for u in U:
        mean_lat = 0
        mean_lng = 0
        cnt = 0
        if u in C:
            for c in C[u]:
                spot = c[1]
                if spot in S:
                    mean_lat += S[spot]['LL'][0]
                    mean_lng += S[spot]['LL'][1]
                    cnt += 1
            if cnt > 0:
                mean_lat /= cnt
                mean_lng /= cnt
                nnz_lat += mean_lat
                nnz_lng += mean_lng
                nnz_cnt += 1
                U[u]['mean_LL'] = (mean_lat, mean_lng)
    nnz_lat /= nnz_cnt
    nnz_lng /= nnz_cnt

    for u in U:
        if 'mean_LL' not in U[u]:
            U[u]['mean_LL'] = (nnz_lat, nnz_lng)
           
def calculate_spot_entropy():
    for spot in S:
        S[spot]['entropy'] = 0
    for u in C:
        s_dict = {}
        for c in C[u]:
            spot = c[1]
            if spot not in s_dict:
                s_dict[spot] = 1
            else:
                s_dict[spot] += 1
        for spot in s_dict:
            if spot in S:
                if S[spot]['check_cnt'] > 0:
                    q = s_dict[spot]/ S[spot]['check_cnt'] 
                    S[spot]['entropy'] -= q*np.log(q)
def calculate_CCC(G, u, v):
    p = 0
    for k in C[u]:
        if k in C[v]:
            p+= 1
    return p
def calculate_cate_cnt():
    for u in U:
        U[u]['cate'] = {}
        if u in C:
            for c in C[u]:
                spot = c[1]
                if spot in S:
                    cat = S[spot]['cate']
                    if cat not in U[u]['cate']:
                        U[u]['cate'][cat] = 1
                    else:
                        U[u]['cate'][cat] += 1
                
def get_features(L, flag):
    X = [[] for i in range(len(L))]

    #=====================Social features(user-to-user graph)======================

    #g0.adamic adar score
    if flag['g0'] is True:
        print("get feature g0") 
        preds = nx.adamic_adar_index(G, L)
        cnt = 0
        for (u, v, p) in preds:
            X[cnt].append(p)
            cnt += 1

    #g1.jaccard coefficient
    if flag['g1'] is True:
        print("get feature g1") 
        preds = nx.jaccard_coefficient(G, L)
        cnt = 0
        for (u, v, p) in preds:
            X[cnt].append(p)
            cnt += 1
    #g2.resource_allocation
    if flag['g2'] is True:
        print("get feature g2") 
        preds = nx.resource_allocation_index(G, L)
        cnt = 0
        for (u, v, p) in preds:
            X[cnt].append(p)
            cnt += 1

    #g3.preferentail_attachment
    if flag['g3'] is True:
        print("get feature g3") 
        preds = nx.preferential_attachment(G, L)
        cnt = 0
        for (u, v, p) in preds:
            X[cnt].append(p)
            cnt += 1
    

    #g4.shortest path length
    if flag['g4'] is True:
        print("get feature g4")
        cnt = 0
        for (u, v) in L:
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                if nx.has_path(G, u, v):
                    X[cnt].append(nx.shortest_path_length(G, source = u, target = v)/50000)
                else:
                    X[cnt].append(1)
                G.add_edge(u, v)
            else:
                if nx.has_path(G, u, v):
                    X[cnt].append(nx.shortest_path_length(G, source = u, target = v)/50000)
                else:
                    X[cnt].append(1)
            cnt += 1

    #g5.common neighbors
    if flag['g5'] is True:
        print("get feature g5")
        cnt = 0
        for (u,v) in L:
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                T = [w for w in nx.common_neighbors(G, u, v)]
                G.add_edge(u, v)
            else:
                T = [w for w in nx.common_neighbors(G, u, v)]
            X[cnt].append(len(T))
            cnt += 1

    #g6.Approximate katz for social graph
    if flag['g6'] is True:
        print("get feature g6")
        cnt = 0
        for (u, v) in L:
            p = 0
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                for x in G.neighbors(u):    
                    for y in G.neighbors(v):
                        if x == y or G.has_edge(x, y):
                            p += 1
                G.add_edge(u, v)
            else:
                for x in G.neighbors(u):    
                    for y in G.neighbors(v):
                        if x == y or G.has_edge(x, y):
                            p += 1
            X[cnt].append(p)
            cnt += 1

    #=========================checkin features=========================================
    #c0.follower number
    if flag['c0'] is True:
        print("get feature c0")
        cnt = 0
        for (u,v) in L:
            X[cnt].append(U[u]['follow_cnt']*U[v]['follow_cnt'])# fu*fv
            cnt += 1

    #c1.same time same location
    if flag['c1'] is True:
        print("get feature c1")
        cnt = 0
        for (u,v) in L:
            p = calculate_CCC(G, u, v)
            X[cnt].append(p)
            cnt += 1

    #c2.same time same distinct spot
    if flag['c2'] is True:
        print("get deature c2")
        cnt = 0
        for (u, v) in L:
            p = 0
            dis_same_spot = []
            for k in C[u]:
                if k[1] not in dis_same_spot and k in C[v]:
                    dis_same_spot.append(k[1])
                    p += 1
            X[cnt].append(p)
            cnt += 1

    #c3.same distinct spot (not necessarily same time)
    if flag['c3'] is True:
        cnt = 0
        print("get feature c3")
        for (u, v) in L:
            p = 0
            dis_same_spot = []
            for k in C[u]:
                if k[1] not in dis_same_spot :
                    for m in C[v]:
                        if k[1] == m[1]:
                            dis_same_spot.append(k[1])
                            p += 1
                            break
            X[cnt].append(p)
            cnt += 1

    #c4.min Entropy
    if flag['c4'] is True:
        print("get feature c4")
        cnt = 0
        for (u,v) in L:
            p = 0
            E_list = []
            for k in C[u]:
                if k in C[v]:
                    spot = k[1]
                    if spot in S  and S[spot]['entropy'] > 0:
                        E_list.append(S[spot]['entropy'])
            if len(E_list) > 0:
                p = min(E_list)
            X[cnt].append(p)
            cnt += 1
        

    #c5. distance of mean_LL
    if flag['c5'] is True:
        cnt = 0
        print("get feature c5")
        for (u, v) in L:
            dist = np.sqrt( (U[u]['mean_LL'][0]-U[v]['mean_LL'][0])**2  +  (U[u]['mean_LL'][1]-U[v]['mean_LL'][1])**2 )
            X[cnt].append(dist)
            cnt += 1

    #c6.weighted same location
    if flag['c6'] is True:
        print("get feature c6")
        cnt = 0
        for (u,v) in L:
            p = 0
            for k in C[u]:
                if k in C[v]:
                    spot = k[1]
                    #if spot in S and S[spot]['entropy'] > 0:
                        #p += 1/S[spot]['entropy']
                    if spot in S :
                        dist = np.sqrt( (S[spot]['LL'][0]-U[u]['mean_LL'][0])**2  +  (S[spot]['LL'][1]-U[u]['mean_LL'][1])**2 )
                        p += dist
                        dist = np.sqrt( (S[spot]['LL'][0]-U[v]['mean_LL'][0])**2  +  (S[spot]['LL'][1]-U[v]['mean_LL'][1])**2 )
                        p += dist
            X[cnt].append(p)
            cnt += 1

    #c7.PP
    if flag['c7'] is True:
        print("get feature c7")
        cnt = 0
        for (u,v) in L:
            p = len(C[u])*len(C[v])
            X[cnt].append(p)
            cnt += 1

    #c8.Total Common Friend Closeness (TCFC)
    if flag['c8'] is True:
        print("get feature c8")
        cnt = 0
        for (u, v) in L:
            p = 0
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                for w in nx.common_neighbors(G, u, v):
                    T1 = [x for x in nx.common_neighbors(G, u, w)]
                    T2 = [x for x in nx.common_neighbors(G, v, w)]
                    p += len(T1)*len(T2)
                G.add_edge(u, v)
            else:
                for w in nx.common_neighbors(G, u, v):
                    T1 = [x for x in nx.common_neighbors(G, u, w)]
                    T2 = [x for x in nx.common_neighbors(G, v, w)]
                    p += len(T1)*len(T2)
            X[cnt].append(p)
            cnt += 1

    #c9.Total Common friend Checkin Count (TCFCC)
    if flag['c9'] is True:
        print("get feature c9")
        cnt = 0
        for (u, v) in L:
            p = 0
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                for w in nx.common_neighbors(G, u, v):
                    p += calculate_CCC(G, u, w)*calculate_CCC(G, v, w)
                G.add_edge(u, v)
            else:
                for w in nx.common_neighbors(G, u, v):
                    p += calculate_CCC(G, u, w)*calculate_CCC(G, v, w)
            X[cnt].append(p)
            cnt += 1

    #c10. Common Category Checkin Counts Product (CCCP)
    if flag['c10'] is True:
        print("get feature c10")
        cnt = 0
        for (u, v) in L:
            p = 0
            for cat in U[u]['cate']:
                if cat in U[v]['cate']:
                    p += U[u]['cate'][cat]*U[v]['cate'][cat]
            X[cnt].append(p)
            cnt += 1

    #c11. Common Category Checkin Counts Product Ratio(CCCPR)
    if flag['c11'] is True:
        print("get feature c11")
        cnt = 0
        for (u, v) in L:
            p = 0
            u_cate_total = sum(U[u]['cate'][cat]**2 for cat in U[u]['cate'])
            v_cate_total = sum(U[v]['cate'][cat]**2 for cat in U[v]['cate'])
            for cat in U[u]['cate']:
                if cat in U[v]['cate']:
                    p += (U[u]['cate'][cat]*U[v]['cate'][cat]/ np.sqrt(u_cate_total*v_cate_total))
            X[cnt].append(p)
            cnt += 1
	#c12.trip route length all
    if flag['c12'] is True:
        print("get feature c12")
        cnt = 0
        for (u,v) in L:
            tripDayLen1 = list()
            tripDayLen2 = list()
            tripDay = "starting"
            tripLen = 0.0
            lastSpot = [0.0,0.0]
            for k in C[u]:
                if not (lastSpot[0] == 0.0 and lastSpot[1] == 0.0):
                    if k[1] in S:
                        tripLen += np.sqrt((lastSpot[0]-S[k[1]]['LL'][0])**2 + (lastSpot[1]-S[k[1]]['LL'][1])**2)
                        lastSpot[0] = S[k[1]]['LL'][0]
                        lastSpot[1] = S[k[1]]['LL'][1]
                else:
                    if k[1] in S:
                        lastSpot[0] = S[k[1]]['LL'][0]
                        lastSpot[1] = S[k[1]]['LL'][1]
            tripDay = "starting"
            tripLen2 = 0.0
            lastSpot = [0.0,0.0]
            for k in C[v]:
                if not (lastSpot[0] == 0.0 and lastSpot[1] == 0.0):
                    if k[1] in S:
                        tripLen2 += np.sqrt((lastSpot[0]-S[k[1]]['LL'][0])**2 + (lastSpot[1]-S[k[1]]['LL'][1])**2)
                        lastSpot[0] = S[k[1]]['LL'][0]
                        lastSpot[1] = S[k[1]]['LL'][1]
                else:
                    if k[1] in S:
                        lastSpot[0] = S[k[1]]['LL'][0]
                        lastSpot[1] = S[k[1]]['LL'][1]
            X[cnt].append(tripLen + tripLen2)
            cnt += 1

    #=========================Heter Graph features=====================================

    #h0.Approximate katz for bipartite graph
    if flag['h0'] is True:
        print("get feature h0")
        cnt = 0
        for (u, v) in L:
            p = 0
            for x in B.neighbors(u):    
                for y in B.neighbors(v):
                    if x == y or B.has_edge(x, y):
                        p += 1
            X[cnt].append(p)
            cnt += 1
    
    #h1.Approximate katz on HB
    if flag['h1'] is True:
        print("get feature h1")
        cnt = 0
        for (u, v) in L:
            p = 0
            if HB.has_edge(u, v):
                HB.remove_edge(u, v)
                for x in HB.neighbors(u):    
                    for y in HB.neighbors(v):
                        if x == y or HB.has_edge(x, y):
                            p += 1
                HB.add_edge(u, v)
            else:
                for x in HB.neighbors(u):    
                    for y in HB.neighbors(v):
                        if x == y or HB.has_edge(x, y):
                            p += 1
            X[cnt].append(p)
            cnt += 1
    
    #h2.Approximate katz on H
    if flag['h2'] is True:
        print("get feature h2")
        cnt = 0
        for (u, v) in L:
            p = 0
            if H.has_edge(u, v):
                H.remove_edge(u, v)
                for x in H.neighbors(u):    
                    for y in H.neighbors(v):
                        if x == y or H.has_edge(x, y):
                            p += 1
                H.add_edge(u, v)
            else:
                for x in H.neighbors(u):    
                    for y in H.neighbors(v):
                        if x == y or H.has_edge(x, y):
                            p += 1
            X[cnt].append(p)
            cnt += 1

    #h3.shortest path length on B
    if flag['h3'] is True:
        print("get feature h3")
        cnt = 0
        for (u, v) in L:
            if nx.has_path(B, u, v):
                X[cnt].append(nx.shortest_path_length(B, source = u, target = v)/50000)
            else:
                X[cnt].append(1)
            cnt += 1

    #h4.clustering coefiicient on H
    if flag['h4'] is True:
        print("get feature h4")
        cnt = 0
        for (u, v) in L:
            if H.has_edge(u, v):
                H.remove_edge(u, v)
                p = nx.clustering(H, u)*nx.clustering(H, v)
                H.add_edge(u, v)
            else:
                p = nx.clustering(H, u)*nx.clustering(H, v)
            X[cnt].append(p)
            cnt += 1
    
    #h5. number of (user's loc friends)'s loc friends
    if flag['h5'] is True:
        print("get feature h5")
        cnt = 0
        for (u,v) in L:
            counter1 = 0
            for neighbor in H.neighbors(u):
                if not neighbor.isnumeric():
                    for neighbor2 in H.neighbors(neighbor):
                        if not neighbor.isnumeric():
                            counter1 += 1
            counter2 = 0
            for neighbor in H.neighbors(v):
                if not neighbor.isnumeric():
                    for neighbor2 in H.neighbors(neighbor):
                        if not neighbor.isnumeric():
                            counter2 += 1

            #print(str(counter1)+" "+str(counter2)+"\n")
            X[cnt].append(counter1 * counter2)
            cnt += 1
    return X

if __name__ == "__main__":
    clf = rf(n_estimators = 1000, n_jobs = -1, max_depth = 20) #classifier
    global G
    G = nx.Graph()
    (L_train, Y_train) = read_train_file("gowalla.train.txt")
    (L_test, Y_test) = read_test_file("gowalla.test.txt" )

    global B
    B = nx.Graph()#bipartite Graph of checkin data
    global HB
    HB = G.copy()
    global H
    H = G.copy()
    

    global U
    U = {}# a dict of user info
    #user1: follower number1
    #user2: follower number2
    #...
    read_users_info("users_info_new.dat")

    global S
    S = {}# a dict of spot info
    read_spots_info("spots_info.dat")

    global C
    C = {}# a dict of tuples 
    #user1:(time, checkin_location)
    #user2:(time, checkin_location)
    #...
    read_checkins_info("checkins_info.dat")
    add_H_edges()
    #calculate mean lag and mean lng for each user
    calculate_mean_LL()

    calculate_spot_entropy()

    calculate_cate_cnt()
    # the flag of whether to user the ith feature
    flag = {}
    for i in range(20):
        flag['g'+str(i)] = False
        flag['c'+str(i)] = False
        flag['h'+str(i)] = False

    flag['g0'] = True 
    #flag['g1'] = True 
    flag['g2'] = True 
    #flag['g3'] = True 
    flag['g4'] = True 
    flag['g5'] = True 
    flag['g6'] = True 

    flag['c0'] = True 
    flag['c1'] = True 
    #flag['c2'] = True 
    flag['c3'] = True 
    #flag['c4'] = True 
    flag['c5'] = True 
    flag['c6'] = True 
    flag['c7'] = True 
    flag['c8'] = True 
    flag['c9'] = True 
    flag['c10'] = True 
    flag['c11'] = True 
    flag['c12'] = True 

    #flag['h0'] = True 
    #flag['h1'] = True 
    #flag['h2'] = True 
    #flag['h3'] = True 
    #flag['h4'] = True 
    flag['h5'] = True 


    #for i in range(20):
        #flag['g'+str(i)] = False
        #flag['c'+str(i)] = False
        #flag['h'+str(i)] = False

    #**********************Training ************************

    print("Training")
    #read training file

    #get features 
    X_train = get_features( L_train, flag)

    #fit
    print("fit")
    clf.fit(X_train, Y_train)
    print(clf.feature_importances_)
    scores = clf.score(X_train, Y_train)
    print("Accuracy in smaple =" , scores)


    #**********************Testing*************************  
    print("Testing")
    #read testing file
    #get features
    X_test = get_features(L_test, flag)

    #predict and get scores
    scores = clf.score(X_test, Y_test)
    print("Accuracy out smaple =" , scores)
