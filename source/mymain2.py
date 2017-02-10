import sys
import networkx as nx
import numpy as np
from math import log
from math import sqrt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier as etc
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.ensemble import BaggingClassifier as bc
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB as nb

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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

        mean_LL = [0.0, 0.0]
        mean_cnt = 0.0

        for line in f:
            line = line.split()
            spot = "s_"+line[0]
            S[spot] = {} #spot ID
            S[spot]['cate'] = line[2]#category
            S[spot]['check_cnt'] = int(line[4])#checkin_cnt
            S[spot]['LL'] = (float(line[6]), float(line[8]))#lat and lng

            mean_cnt += int(line[4])
            mean_LL[0] += float(line[6])
            mean_LL[1] += float(line[8])

        mean_cnt /= len(S)
        mean_LL[0] /= len(S)
        mean_LL[1] /= len(S)
        S['mean'] = {}
        S['mean']['check_cnt'] = mean_cnt
        S['mean']['LL'] = (mean_LL[0], mean_LL[1])



def read_checkins_info(filename):
    with open(filename) as f:
        for line in f:
            line = line.split()
            u = line[0]
            C[u] = []

            C2[u] = []

            for info in line[1:]:
                t = info.split("T")[0]

                h = info.split("T")[1].split(":")[0]
                h = int(h)

                spot = "s_"+info.split(":")[-1]
                C[u].append((t, spot))
                
                C2[u].append((t, h, spot))

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
        print(i)
        while dist < 0.1 and j < M: 
            dist = np.sqrt( (S[s1]['LL'][0]-S[s2]['LL'][0])**2 + (S[s1]['LL'][1]-S[s2]['LL'][1])**2)
            H.add_edge(s1, s2)
            j+= 1
            if j < M:
                s2 = Index[j]
            cnt += 1

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

    #0.same time same location
    if flag[0] is True:
        print("get feature0")
        cnt = 0
        for (u,v) in L:
            p = calculate_CCC(G, u, v)
            X[cnt].append(p)
            cnt += 1

    #1.resource_allocation
    if flag[1] is True:
        print("get feature1") 
        preds = nx.resource_allocation_index(G, L)
        cnt = 0
        for (u, v, p) in preds:
            X[cnt].append(p)
            cnt += 1

    #2.jaccard coefficient
    if flag[2] is True:
        print("get feature2") 
        preds = nx.jaccard_coefficient(G, L)
        cnt = 0
        for (u, v, p) in preds:
            X[cnt].append(p)
            cnt += 1

    #3.adamic adar score
    if flag[3] is True:
        print("get feature3") 
        preds = nx.adamic_adar_index(G, L)
        cnt = 0
        for (u, v, p) in preds:
            X[cnt].append(p)
            cnt += 1

    
    #4.preferentail_attachment
    if flag[4] is True:
        print("get feature4") 
        preds = nx.preferential_attachment(G, L)
        cnt = 0
        for (u, v, p) in preds:
            X[cnt].append(p)
            cnt += 1
    
    #5.shortest path length
    if flag[5] is True:
        print("get feature5")
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

    #6.follower number
    if flag[6] is True:
        print("get feature6")
        cnt = 0
        for (u,v) in L:
            X[cnt].append(U[u]['follow_cnt']*U[v]['follow_cnt'])# fu*fv
            cnt += 1

    #7.common neighbors
    if flag[7] is True:
        print("get feature7")
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

    #8. distance of mean_LL
    if flag[8] is True:
        cnt = 0
        print("get feature8")
        for (u, v) in L:
            dist = np.sqrt( (U[u]['mean_LL'][0]-U[v]['mean_LL'][0])**2  +  (U[u]['mean_LL'][1]-U[v]['mean_LL'][1])**2 )
            X[cnt].append(dist)
            cnt += 1
    
    #9.same time same distinct spot
    if flag[9] is True:
        print("get deature9")
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
        
    #10.same distinct spot (not necessarily same time)
    if flag[10] is True:
        cnt = 0
        print("get feature10")
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
        
    #11.weighted same location
    if flag[11] is True:
        print("get feature11")
        cnt = 0
        for (u,v) in L:
            p = 0
            for k in C[u]:
                if k in C[v]:
                    spot = k[1]
                    if spot in S and S[spot]['entropy'] > 0:
                        p += 1/S[spot]['entropy']
                        #dist = np.sqrt( (S[spot]['LL'][0]-U[u]['mean_LL'][0])**2  +  (S[spot]['LL'][1]-U[u]['mean_LL'][1])**2 )
                        #p += dist
                        #dist = np.sqrt( (S[spot]['LL'][0]-U[v]['mean_LL'][0])**2  +  (S[spot]['LL'][1]-U[v]['mean_LL'][1])**2 )
                        #p += dist
            X[cnt].append(p)
            cnt += 1

    #12.weighted same location
    if flag[12] is True:
        print("get feature12")
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
                        #dist = np.sqrt( (S[spot]['LL'][0]-U[u]['mean_LL'][0])**2  +  (S[spot]['LL'][1]-U[u]['mean_LL'][1])**2 )
                        #p += dist
                        #dist = np.sqrt( (S[spot]['LL'][0]-U[v]['mean_LL'][0])**2  +  (S[spot]['LL'][1]-U[v]['mean_LL'][1])**2 )
                        #p += dist
            X[cnt].append(p)
            cnt += 1
    #13.pp
    if flag[13] is True:
        print("get feature13")
        cnt = 0
        for (u,v) in L:
            p = len(C[u])*len(C[v])
            X[cnt].append(p)
            cnt += 1

    #Total Common Friend Closeness (TCFC)
    if flag[14] is True:
        print("get feature14")
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

    #15.Total Common friend Checkin Count (TCFCC)
    if flag[15] is True:
        print("get feature15")
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
    #16. Common Category Checking Counts Rroduct (CCCP)
    if flag[16] is True:
        print("get feature16")
        cnt = 0
        for (u, v) in L:
            p = 0
            for cat in U[u]['cate']:
                if cat in U[v]['cate']:
                    p += U[u]['cate'][cat]*U[v]['cate'][cat]
            X[cnt].append(p)
            cnt += 1

    #17. Common Category Checking Counts Product Ratio(CCCPR)
    if flag[17] is True:
        print("get feature17")
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

    #18.Approximate katz for social graph
    if flag[18] is True:
        print("get feature18")
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

    #19.Approximate katz for bipartite graph
    if flag[19] is True:
        print("get feature19")
        cnt = 0
        for (u, v) in L:
            p = 0
            for x in B.neighbors(u):    
                for y in B.neighbors(v):
                    if x == y or B.has_edge(x, y):
                        p += 1
            X[cnt].append(p)
            cnt += 1

    # other features #
    #20.Similar locations for heter graph#
    if flag[25] is True:
        print("get feature25")
        cnt = 0
        Nu = set()
        Nv = set()
        sim_Nu = set()
        sim_Nv = set()
        for (u, v) in L:
            p = 0
            Nu.clear()
            Nv.clear()
            sim_Nu.clear()
            sim_Nv.clear()
            for l in H.neighbors(u):
                if 's_' in l:
                    Nu.add(l)
                    for m in H.neighbors(l):
                        if 's_' in m:
                            sim_Nu.add(m)
            for l in H.neighbors(v):
                if 's_' in l:
                    Nv.add(l)
                    for m in H.neighbors(l):
                        if 's_' in m:
                            sim_Nv.add(m)
            N_union = Nu | Nv
            for s in N_union:
                if s in Nv and s in Nu:
                    p += 1
                elif s in Nv and s in sim_Nu:
                    p += 0.5
                elif s in sim_Nv and s in Nu:
                    p += 0.5
            X[cnt].append(p)
            cnt += 1



    #same time same location (hour)
    if flag[26] is True:
        print("get feature26")
        cnt = 0
        for (u, v) in L:
            p = 0.0
            for k in C2[u]:
                for m in C2[v]:
                    if k == m:
                        p += 1
                    elif k[0] == m[0] and k[2] == m[2] and (k[1] == m[1]-1 or k[1] == m[1]+1):
                        p += 0.5
            X[cnt].append(p)
            cnt += 1

    #dist. checkin cout weight common location
    if flag[27] is True:
        print("get feature27")
        cnt = 0
        for (u, v) in L:
            p = 0
            for k in C[u]:
                for m in C[v]:
                    if k == m:
                        spot = k[1]
                        dist = 0.0
                        if spot in S:
                            dist += sqrt( (S[spot]['LL'][0]-U[u]['mean_LL'][0])**2 + (S[spot]['LL'][1]-U[u]['mean_LL'][1])**2 )
                            dist += sqrt( (S[spot]['LL'][0]-U[v]['mean_LL'][0])**2 + (S[spot]['LL'][1]-U[v]['mean_LL'][1])**2 )
                            if S[spot]['check_cnt'] > 1:
                                if dist > 1.0:
                                    p += 1/(S[spot]['check_cnt'])*dist
                                else:
                                    p += 1/(S[spot]['check_cnt'])*dist
                            else:
                                if dist > 1.0:
                                    p += dist
                                else:
                                    p += dist
                        else:
                            dist += sqrt( (S['mean']['LL'][0]-U[u]['mean_LL'][0])**2 + (S['mean']['LL'][1]-U[u]['mean_LL'][1])**2 )
                            dist += sqrt( (S['mean']['LL'][0]-U[v]['mean_LL'][0])**2 + (S['mean']['LL'][1]-U[v]['mean_LL'][1])**2 )
                            if dist > 1.0:
                                p += 1/(S['mean']['check_cnt'])*dist
                            else:
                                p += 1/(S['mean']['check_cnt'])*dist
            X[cnt].append(p)
            cnt += 1

    #cosine simillarity
    if flag[28] is True:
        print("get feature28")
        cnt = 0
        u_dict = dict()
        v_dict = dict()
        for (u, v) in L:
            u_dict.clear()
            v_dict.clear()
            p = 0.0
            for k in C[u]:
                if k not in u_dict:
                    u_dict[k] = 0
                u_dict[k] += 1
            for m in C[v]:
                if m not in v_dict:
                    v_dict[m] = 0
                v_dict[m] += 1
            for k in u_dict:
                if k in v_dict:
                    p += u_dict[k]*v_dict[k]
            p /= ( sqrt(sum([k**2 for k in u_dict.values()])*sum([m**2 for m in v_dict.values()])))
            X[cnt].append(p)
            cnt += 1

    #personal normalized cosine similarity
    if flag[29] is True:
        print('get feature29')
        cnt = 0
        u_dict = dict()
        v_dict = dict()
        n_dict = dict()
        '''
        m_p = 0.0
        for u in U:
            for v in U:
                if u != v and G.has_edge(u, v):
                    u_dict.clear()
                    v_dict.clear()
                    for k in C[u]:
                        if k not in u_dict:
                            u_dict[k] = 0
                        u_dict[k] += 1
                    for l in C[v]:
                        if l not in v_dict:
                            v_dict[l] = 0
                        v_dict[l] += 1
                    for k in u_dict:
                        if k in v_dict:
                            m_p += u_dict[k]*v_dict[k]
        m_p /= len(G.edges())
        '''
        for (u, v) in L:
            u_dict.clear()
            v_dict.clear()
            u_p = 0.0
            v_p = 0.0
            p = 0.0
            for k in C[u]:
                if k not in u_dict:
                    u_dict[k] = 0
                u_dict[k] += 1
            for l in C[v]:
                if l not in v_dict:
                    v_dict[l] = 0
                v_dict[l] += 1
            for n in G.neighbors(u):
                n_dict.clear()
                for m in C[n]:
                    if m not in n_dict:
                        n_dict[m] = 0
                    n_dict[m] += 1
                for k in u_dict:
                    if k in n_dict:
                        u_p += u_dict[k]*n_dict[k]
            u_len = len(G.neighbors(u))
            if u_len != 0:
                u_p /= len(G.neighbors(u))
            for n in G.neighbors(v):
                n_dict.clear()
                for m in C[n]:
                    if m not in n_dict:
                        n_dict[m] = 0
                    n_dict[m] += 1
                for l in v_dict:
                    if l in n_dict:
                        v_p += v_dict[l]*n_dict[l]
            v_len = len(G.neighbors(v))
            if v_len != 0:
                v_p /= len(G.neighbors(v))
            for k in u_dict:
                if k in v_dict:
                    p += u_dict[k]*v_dict[k]
            '''
            if u_p != 0.0 and v_p != 0.0:
                p = (p/u_p + p/v_p)/2
            elif u_p == 0.0 and v_p != 0.0:
                p = (p/m_p + p/v_p)/2
            elif u_p != 0.0 and v_p == 0.0:
                p = (p/u_p + p/m_p)/2
            else:
                p = p/m_p
            '''
            p = p + ((p-u_p)+(p-v_p))/2
            X[cnt].append(p)
            cnt += 1


    
    #Similar total common friend count check (similar TCFCC)
    if flag[30] is True:
        print("get feature30")
        cnt = 0
        for (u, v) in L:
            p = 0
            removed = False
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                removed = True
            for w in nx.common_neighbors(G, u, v):
                q = 0
                r = 0
                for n in C[w]:
                    for k in C[u]:
                        if n[0] == k[0]:
                            if n[1] == k[1]:
                                q += 1
                            elif H.has_edge('s_'+str(n[1]), 's_'+str(k[1])):
                                q += 1
                    for m in C[v]:
                        if n[0] == m[0]:
                            if n[1] == m[1]:
                                r += 1
                            elif H.has_edge('s_'+str(n[1]), 's_'+str(m[1])):
                                r += 1
            if removed:
                G.add_edge(u, v)
            p += q*r
            X[cnt].append(p)
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


    global C2
    C2 = {}# dict of tuples
    #user1:(date, hour, checkin_location)
    #user2:(date, hour, checkin_location)
    #...


    read_checkins_info("checkins_info.dat")
    add_H_edges()
    #calculate mean lag and mean lng for each user
    calculate_mean_LL()

    calculate_spot_entropy()

    calculate_cate_cnt()
    # the flag of whether to user the ith feature
    flag = {i:False for i in range(35)}
    flag[0] = True 
    flag[1] = True 
    #flag[2] = True 
    flag[3] = True 
    #flag[4] = True 
    flag[5] = True 
    flag[6] = True 
    flag[7] = True 
    flag[8] = True 
    #flag[9] = True 
    flag[10]= True 
    #flag[11] = True 
    #flag[12] = True 
    flag[13] = True 
    flag[14] = True 
    #flag[15] = True 
    #flag[16] = True 
    flag[17] = True 
    flag[18] = True 
    #flag[19] = True 
    #flag[20] = True 
    #flag[21] = True 
    #flag[22] = True

    #flag[25] = True
    #flag[26] = True

    #flag[27] = True
    #flag[28] = True
    #flag[29] = True
    #flag[30] = True
    #for i in range(19):
    #    flag[i] = False


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
    
    P_test = clf.predict(X_test)

    scores = clf.score(X_test, Y_test)
    print("Accuracy out smaple =" , scores)

    prec = precision_score(Y_test, P_test, average=None)
    print("each precision scores =", prec)
    prec = precision_score(Y_test, P_test, average='weighted')
    print("weighted precision scores =", prec)
    rec = recall_score(Y_test, P_test, average=None)
    print("each recall scores =", rec)
    rec = recall_score(Y_test, P_test, average='weighted')
    print("weighted recall scores =", rec)
