# -*- coding: utf-8 -*-


import numpy as np
import datetime
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import sparse
import pydot
import pylab as py
import sys
import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn import preprocessing
from sklearn.decomposition import PCA
from operator import itemgetter
import nx_altair as nxa
import random
from celluloid import Camera
### Functions to get network data
## 3d networks where the third dimension are the days

def make_reaction_network(df, users): ###users are the total list of users, df contains ts, user from and user to
  
    time_min = min(df['ts']).date()
    time_max = max(df['ts']).date()
    
    d = +datetime.timedelta(days = 1)

    t = time_min

    timescale = []
    while t <= time_max:
        t = t + d
        timescale.append(t)
        
    duration = len(timescale)

    network = np.zeros((len(users), len(users), duration), dtype=int)

    for k in range(len(df)):
        user = df.iloc[k]['user']
        original_user = df.iloc[k]['original_user']
        time = df.iloc[k]['ts'].date()

        i = np.where(users == user)[0][0]
        j = np.where(users == original_user)[0][0]
        t = (time-time_min).days

        network[i, j, t] = network[i, j, t] + 1

    return {'nw': network, 'ts': timescale}

def make_thread_network(df, users):
      
    time_min = min(df['ts']).date()
    time_max = max(df['ts']).date()
    
    d = +datetime.timedelta(days = 1)

    t = time_min

    timescale = []
    while t <= time_max:
        t = t + d
        timescale.append(t)
        
    duration = len(timescale)

    network = np.zeros((len(users), len(users), duration), dtype=int)
    
    threads = np.unique(threads_data['thread_ts'])
    threads = [pd.to_datetime(i) for i in threads]
    
    for k in threads:
        #df.iloc['thread_ts' == k]
        ou = df.loc[df['thread_ts'] == k]['original_user']
        u = df.loc[df['thread_ts'] == k]['user']
        thread_users = list(np.unique(np.concatenate((ou, u))))
        thread_users_indx = [np.nonzero(users==i)[0][0] for i in thread_users]
        time = k.date()
        t = (time-time_min).days
        
        for i in thread_users_indx:
            for j in thread_users_indx:
                if i != j:
                    network[i, j, t] = network[i, j, t] + 1
                                 

    return {'nw': network, 'ts': timescale}

def make_channel_network(df, users, channels):
    
    time_min = min(df['ts']).date()
    time_max = max(df['ts']).date()
    
    d = +datetime.timedelta(days = 1)

    t = time_min

    timescale = []
    while t <= time_max:
        t = t + d
        timescale.append(t)
        
    duration = len(timescale)

    #channels = np.unique(df['channel'])
    
    network = np.zeros((len(users), len(channels), duration), dtype = int)
    
    for k in range(len(df)):
        user = df.iloc[k]['user']
        channel = df.iloc[k]['channel']
        time = df.iloc[k]['ts'].date()
        
        i = np.where(users == user)[0][0]
        j = np.where(channels == channel)[0][0]
        t = (time-time_min).days

        network[i, j, t] = network[i, j, t] + 1
        

    return {'nw': network, 'ts': timescale}

def make_reaction_from_channel(network):
    #### make reaction network out of channel network
    n_users = np.shape(network)[0]
    timescale = np.shape(network)[2]
    nwcn = np.zeros((n_users, n_users), dtype = int)
    
    for w in range(timescale):
        connections = np.nonzero(network[:, :, w])

        for i in np.unique(connections[1]):
            indx = np.nonzero(connections[1] == i)
            same_channel = connections[0][indx]
            for j in same_channel:
                nwcn[j, same_channel] += 1

    for j in range(n_users):
        nwcn[j][j] = 0
            
    return nwcn

def same_timescale(*args): ## input argument: networks + timescales as a list of tuples!
    #### bring networks to the same timescale
    timescales = []
    new_nw_list = []
    for a in args:
        timescales.append(a['ts'])
    
    timescales = np.concatenate(timescales)
    absolute_ts = np.sort(np.unique(timescales))
    
    for a in args:
        nw = a['nw']
        ts = a['ts']
        size = (np.shape(nw))
        new_size = size[0:2] + (len(absolute_ts), )
        new_nw = np.zeros(new_size)
        new_nw[:, :, [i in ts for i in absolute_ts]] = nw
        new_nw_list.append(new_nw)
        
    return absolute_ts, new_nw_list

def make_time_network(network, ts, timescale): 
    ### make weekly timescale network out of daily timescale network
    
    if timescale == 'week': 
        time_ts = [i.isocalendar()[1] for i in ts]
    elif timescale == 'month':
        time_ts = [i.month for i in ts]
    else:
        raise ValueError('The timescale has to be week or month')
        

    u, indices = np.unique(time_ts, return_inverse=True)
    ### sum up the network based on the number in weekly_ts!
    time_network = np.zeros((np.shape(network)[0], np.shape(network)[1], len(u)))

  
    for i in list(range(len(u))):
        indx = np.nonzero(i == indices)[0]
        new_network = np.sum(network[:, :, indx], axis = 2)
        time_network[:, :, i] = new_network
        #weekly_network = np.stack(weekly_network, new_network)

    return {'nw': time_network, 'ts': u}

def make_networks_with_timeframe(times, total_weekly_nw, tnw_weekly, rnw_weekly):
### make networks at a specific timeframe!

    #times = [(1, 27), (10, 13)]
    networks_matrix = dict.fromkeys(range(len(times)))
    networks = dict.fromkeys(range(len(times)))
    
    nw = total_weekly_nw['nw']
    weeks = total_weekly_nw['ts']
    
    for k in range(len(times)):
        startweek = min(times[k])
        endweek = max(times[k])
        start = np.nonzero(weeks == startweek)[0][0]
        end = np.nonzero(weeks == endweek)[0][0]
    
        channel_user_matrix = np.sum(nw[:, :, start:end], axis = 2)
        channel_user_matrix_sparse = sparse.csr_matrix(channel_user_matrix)
        channel_user_nw = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(channel_user_matrix_sparse)
    
        nwcn = np.zeros((len(users), len(users)), dtype = int)
    
        for w in range(start, end):
            connections = np.nonzero(nw[:, :, w])
    
            for i in np.unique(connections[1]):
                indx = np.nonzero(connections[1] == i)
                same_channel = connections[0][indx]
                for j in same_channel:
                    nwcn[j, same_channel] += 1
    
            for j in range(len(users)):
                nwcn[j][j] = 0
                
        channel_user_user_nw = nx.from_numpy_matrix(nwcn)
    
        
        thread_nw_matrix = np.sum(tnw_weekly['nw'][:, :, start:end], axis = 2)
        thread_nw = nx.from_numpy_matrix(thread_nw_matrix)
    
        
        reaction_nw_matrix = np.sum(rnw_weekly['nw'][:, :, start:end], axis = 2)
        reaction_nw = nx.DiGraph(reaction_nw_matrix)
        
        networks[k] = {'channel_user': channel_user_nw, 'user_user': channel_user_user_nw, 'thread': thread_nw,
                      'reaction': reaction_nw}
        networks_matrix[k] = {'channel_user': channel_user_matrix, 'user_user': nwcn, 'thread': thread_nw_matrix,
                      'reaction': reaction_nw_matrix}
        
    return networks_matrix, networks

def make_subgraphs(networks):
    subgraphs = dict.fromkeys(networks.keys(),{})
    for j, i in networks.items():
        for k, v in i.items():
            #G = v 
            if nx.is_directed(v):
                S = [v.subgraph(c).copy() for c in nx.strongly_connected_components(v)]
            else:
                S = [v.subgraph(c).copy() for c in nx.connected_components(v)]
    
            component_sizes = [c.size() for c in S]
           # subgraphs_indx = np.nonzero(component_sizes)[0]
    
            maxcc_indx = np.argmax(component_sizes)
            maxcc = S[maxcc_indx]
    
            subgraphs[j][k] = maxcc
    return subgraphs

data = pd.read_csv('/Users/klaudiamur/Dropbox/CommNet/data SU/data.csv')

data = data[~data['user'].isnull()]
data['ts'] = pd.to_datetime(data['ts'], unit = 's')
data['thread_ts'] = pd.to_datetime(data['thread_ts'], unit = 's')

active_users = np.unique(data['user'])
#teams = pd.unique(data['team'])
channels = np.unique(data['channel'])

### Get reaction data

reaction_data = data[~data['reactions'].isnull()]
posts_with_reactions = reaction_data

len_df = 0
for i in range(len(reaction_data)):
    a = eval(reaction_data['reactions'].iloc[i])
    n_reactions= len(a)
    for j in range(n_reactions):
        count = a[j]['count']
        len_df = len_df + count


columns = ['original_user', 'ts', 'channel', 'user', 'name', 'thread_ts']
index = list(range(len_df))
df = pd.DataFrame(index = index, columns = columns)
c = 0
for i in range(len(reaction_data)):
    a = eval(reaction_data['reactions'].iloc[i])
    
    n_reactions= len(a)
    for j in range(n_reactions):
        n_users = len(a[j]['users'])
        
        for k in range(n_users):
            
            new_row = {'name': a[j]['name'], 'user' : a[j]['users'][k], 'original_user' : reaction_data['user'].iloc[i], 'ts' : reaction_data['ts'].iloc[i], 'channel' : reaction_data['channel'].iloc[i], 'thread_ts' : reaction_data['thread_ts'].iloc[i]}
            df.iloc[c, :] = new_row
            c = c+1

reaction_data = df

### get thread data
threads_data = data[~data['parent_user_id'].isnull()]
posts_with_comments_id = np.unique(threads_data['thread_ts'])
posts_with_comments = data.loc[data['ts'].isin(posts_with_comments_id)]
threads_data = threads_data.sort_values('thread_ts')
threads_data = threads_data.rename({'parent_user_id': 'original_user'}, axis = 'columns')

## get list of all users
active_users = np.unique(data['user'])
users_reacting = np.unique(reaction_data['user'])
users_reacted_to = np.unique(reaction_data['original_user'])
users = np.unique(np.concatenate((active_users, users_reacting, users_reacted_to), axis = 0))

### make networks
a = make_channel_network(data, users, channels)
b = make_channel_network(reaction_data, users, channels)
tnw = make_thread_network(threads_data, users)
rnw = make_reaction_network(reaction_data, users)
abs_ts, channel_nws= same_timescale(a, b, tnw, rnw)
total_nw = channel_nws[0] + channel_nws[1]

total_weekly_nw = make_time_network(total_nw, abs_ts, 'week')
tnw_weekly = make_time_network(channel_nws[2], abs_ts, 'week')
rnw_weekly = make_time_network(channel_nws[3], abs_ts, 'week')

#times = [(1, 27), (1, 12), (13,27), (11, 14)]
#times =  [(1, 27)] + [(i, i+1) for i in range(1, 26)]
times =  [(i, i+1) for i in range(1, 26)]
nw_matrix, nw = make_networks_with_timeframe(times, total_weekly_nw, tnw_weekly, rnw_weekly)

subgraphs = make_subgraphs(nw)

#cliques = {a: [[i for i in list(nx.find_cliques(nw[j][a])) if len(i) > 2] for j in range(len(nw))] for a in ['thread', 'user_user']}

#### find stats (plot them as nodecolors)
users_data = pd.DataFrame(users)

for i in nw_matrix.keys():
    for k, v in {
            'tot_posts_ratio': np.sum(nw_matrix[i]['channel_user'], axis = 1)/np.sum(nw_matrix[i]['channel_user']),
            'active_channels': np.count_nonzero(nw_matrix[i]['channel_user'], axis = 1)/np.shape(nw_matrix[i]['channel_user'])[1],
            'thread_starter': np.sum(nw_matrix[i]['thread'], axis = 1)/np.sum(nw_matrix[i]['thread']),
            'thread_answerer': np.sum(nw_matrix[i]['thread'], axis = 0)/np.sum(nw_matrix[i]['thread']),
            'reaction_reciever': np.sum(nw_matrix[i]['reaction'], axis = 1)/np.sum(nw_matrix[i]['reaction']),
            'reaction_giver': np.sum(nw_matrix[i]['reaction'], axis = 0)/np.sum(nw_matrix[i]['reaction'])
        }.items():

        users_data[str(i)+'_'+k] = pd.Series(v)


#users_data = pd.DataFrame(users)

#### add network measurements to user data frame
for k, v in nw.items():
    for name, ntwork in v.items():
        if name == 'channel_user':
            list_of_methods = ['degree_centrality',  'betweenness_centrality', 'closeness_centrality']
            #top_nodes, bottom_nodes= nx.algorithms.bipartite.basic.sets(nw)
            #users_data_before_c = pd.DataFrame(columns = list_of_methods_nodes)
            top_nodes = list(range(len(users)))

            for j in list_of_methods:
                method_to_call = getattr(nx.algorithms.bipartite.centrality, j)
                result = method_to_call(ntwork, top_nodes)
                users_data[str(k)+'_'+j+'_'+name] = pd.Series(result).loc[:len(users)]
            
            
        if name == 'user_user' or name == 'thread':
            list_of_methods = ['degree_centrality', 'betweenness_centrality', 'clustering']
            
            for i in list_of_methods:
                method_to_call = getattr(nx, i)
                result = method_to_call(ntwork)
                users_data[str(k)+'_'+i+'_'+name] = pd.Series(result)
            
        if name == 'reaction':
            list_of_methods = ['in_degree_centrality', 'out_degree_centrality']
            
            for i in list_of_methods:
                method_to_call = getattr(nx, i)
                result = method_to_call(ntwork)
                users_data[str(k)+'_'+ i+'_'+name] = pd.Series(result)

   
users_data = users_data.fillna(0)

# X = preprocessing.StandardScaler().fit_transform(users_data.loc[:, users_data.columns != 0])

# reduced_data = PCA(n_components=2).fit_transform(X)


# top_degree_centrality = np.array(users_data.iloc[:, 28].argsort()[::-1])
# top_betweenness_centrality = np.array(users_data.iloc[:, 30].argsort()[::-1])
# top_eigenvector_centrality = np.array(users_data.iloc[:, 29].argsort()[::-1])
# top_tot_posts = np.array(users_data.iloc[:, 1].argsort()[::-1])



########
## Do some sort of diversity/complexity measurement by seeing the tillhörighet to the channels as a vector!!
##
########


channels_user_frequency = preprocessing.normalize(np.sum(total_weekly_nw['nw'], axis = 2), norm='l1')

def calc_nw_complexity(network_matrix, feature_matrix, degree):
    #### how far away are the points from each other in the vectorspace?
    
    adj_matrix = np.sign(network_matrix)
    for i in range(degree-1):
        adj_matrix = np.sign(np.dot(adj_matrix, network_matrix))
        
    #data_combined = np.dot(adj_matrix, feature_matrix) ## wait is it really only the ones it is combined to??
    data1 = np.zeros(len(adj_matrix))
    
    for i in range(len(adj_matrix)):
        d = 0
        if not np.sum(adj_matrix[i]) == 0:
            for j in range(len(adj_matrix)):
                if adj_matrix[i][j] > 0:
                    d += np.linalg.norm(feature_matrix[i] - feature_matrix[j])
            d = d/np.sum(adj_matrix[i])
        data1[i] = d
    return data1

### as a function of how different all are of each other??

        
complexity_data = {}

for k, v in nw_matrix.items():
    complexity_data[k] = {}
    for name_nw, nwtmp in v.items():
        if name_nw != 'channel_user':
            d = calc_nw_complexity(nwtmp, channels_user_frequency, 1)
            #d_tot = np.sum(d)
            complexity_data[k][name_nw] = d
            #complexity_tot[name_nw][k] = d_tot


complexity_tot = dict.fromkeys(nw_matrix[0].keys(),np.zeros(len(times)))
complexity_tot.pop('channel_user')
for k, v in complexity_tot.items():
    d_list =[]
    for i in range(len(times)):
        d_tot = np.mean(complexity_data[i][k])
        d_list.append(d_tot)
        #print(v, d_tot)
    complexity_tot[k] = d_list
 ### ok des funkt offensichtlich net
 
title = 'thread'       
plt.figure(figsize=(10,7))
#cmap = 'Blues'
for i in range(len(times)):
    if i < 7:
        c = 'b'
    else:
        c='r'
    plt.plot(np.sort(complexity_data[i][title]), c=c,alpha=(2*i /len(times)%1))
plt.legend(labels = times)
plt.title('Diversity of interactions, '+title+'network')
plt.show()

plt.figure(figsize=(10,7))
#cmap = 'Blues'
for k in complexity_tot.keys():
    plt.plot(range(1, 26), complexity_tot[k])
plt.legend(labels = complexity_tot.keys())
plt.title('Diversity of interaction over time')
plt.xlabel('Week')
plt.ylabel('Diversity of interaction')
plt.show()

    
    
    
        
reduced_data = PCA(n_components=2).fit_transform(channels_user_frequency)       
        
#color = users_data['0_tot_posts_ratio']
color = users_data['0_degree_centrality_user_user']

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=color, cmap = 'Purples')



# ##### PLOT ######
# x = 'reaction'
# indx_nw = 0
# G = nw[indx_nw][x]

# Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
# G0 = G.subgraph(Gcc[0])
# pos = nx.drawing.nx_agraph.graphviz_layout(G0, prog = 'neato')

# #nodecolors =list((nx.betweenness_centrality(nw[0]['thread'])).values())
# #nodecolors =list(nx.clustering(G0).values())
# #nodecolors =list(nx.eigenvector_centrality(G0).values())
# #nodecolors = users_data.loc[list(G0.nodes), '0_active_channels']
# #G = subgraphs[0]['thread']
# #nodecolors = np.array(users_data.iloc[list(G.nodes),  28])
# # options = {
# #     'cmap': 'Purples',
# #     #"edge_cmap": 'grey',
# #     "node_color": nodecolors,
# #     'alpha': 0.8,
# #     'node_size':150,
# #     #"edge_color": edgecolors,
# #     #'edge_color': 'grey',
# #    # "width": 0.2,
# #     #"edge_cmap": 'RdBu_r',
# #     #"with_labels": False, 
# #     'vmin': 0,
# #     #'vmax': 1

# # }
# plt.figure(dpi=800, figsize = (12,12))
# # plt.title('Degree centrality')

# #nodes = nx.draw_networkx_nodes(G0, pos, **options)
# nx.draw_networkx_nodes(G0, pos, node_color='Purple', node_size = 150)
# nx.draw_networkx_edges(G0, pos, width = 0.2)
# #plt.colorbar(nodes)
# plt.show()


# cliques = list(nx.find_cliques(G0))
# size_cliques = [len(i) for i in cliques]
# max_clique_indx = np.argmax(size_cliques)
# max_clique_nodes = cliques[max_clique_indx]
# #max_clique_nodes = cliques[20]


# plt.figure(dpi=800, figsize = (10,10))
# #plt.title('Degree centrality')

# nodecolors = ['purple' if i in max_clique_nodes else 'thistle' for i in list(G0.nodes)]

#nodecolors = np.array(users_data.iloc[list(G0.nodes),  2])
#nodecolors = [1 if i in max_clique_nodes else 0.2 + random.random()*0.4 for i in list(G0.nodes)]

#nx.draw_networkx_nodes(G0, pos, node_color =nodecolors, cmap = 'Purples', node_size = 150, vmin = 0)

#nx.draw_networkx_nodes(max_clique, pos, node_color = 'purple')
#nx.draw_networkx_edges(G0, pos, width = 0.2, alpha = 0.6)
#plt.colorbar(nodes)
#plt.show()


#### make correlation plots
# plt.imshow(users_data.corr())

# data_corr = users_data[['0_tot_posts_ratio', '0_active_channels', '2_degree_centrality_thread', '2_eigenvector_centrality_thread', '2_betweenness_centrality_thread']]

# plt.matshow(data_corr.corr(), vmin = 0)
# plt.colorbar()
# plt.show()




# x = 'channel_user'
# indx_nw = 0
# G = nw[indx_nw][x]

# Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
# G0 = G.subgraph(Gcc[0])
# #pos = nx.drawing.nx_agraph.graphviz_layout(G0, prog = 'dot')
# pos = nx.bipartite_layout(G0, list(range(161, len(list(G.nodes())))), align='horizontal')
# pos1 = {key:([value[0], 0.8] if value[1] == 1 else value) for (key,value) in pos.items()}
# pos2 = {key:([value[0]+0.015, 0.83] if value[1] == 1 else value) for (key,value) in pos.items()}

# fig = plt.figure(dpi=200, figsize = (15,10))
# #plt.ylim((-0.3,1.2))
# # plt.xlim((0,600))
# plt.axis('off')
# nl_users = [j for j in range(161) if j in list(G0.nodes())]
# nl_channels = [j for j in range(161, 203) if j in G0.nodes()]

# channel_labels = dict([(i, channels[i-161]) for i in nl_channels])
# # for i in range(24):
# #     nc = users_data[(str(i+1)+'_degree_centrality_thread')][list(G0.nodes())]
# #     nx.draw_networkx_nodes(G0, pos, node_color=nc, cmap = 'Reds', node_size = 150, vmin=-0.02, vmax=0.1)
# #     nx.draw_networkx_edges(nw[i+1][x], pos, width = 0.2)
# #     title = 'fig' + str(i) + '.png'
# #     plt.savefig('/Users/klaudiamur/Google Drive/Commwork/Clients/KTH:SU/thread_nw/'+title)
    
# # ###change node color with betweenness centrality!!!

# camera = Camera(fig)


# for i in [12]:
    
    
#     d_c = nx.degree_centrality(nw[i+1][x])
    
#     # nc_users = [d_c.get(j) for j in nl_users]
#     # nc_channels = [d_c.get(j) for j in nl_channels]

#     nc_users = (np.sum(nw_matrix[i+1]['channel_user'], axis = 1)/200)[nl_users]
#     nc_channels = np.sum(nw_matrix[i+1]['channel_user'], axis = 0)/50
    
    
#     nx.draw_networkx_nodes(G0, pos1, nodelist = nl_users, node_color = nc_users, cmap = 'Purples', node_size = 50, vmin = -0.1, vmax = 0.5)
#     nx.draw_networkx_nodes(G0, pos1, nodelist = nl_channels, node_color = nc_channels, cmap = 'Reds', node_size = 200, vmin = -0.05, vmax = 1)
#     #nx.draw_networkx_labels(G0, pos,  labels = channel_labels, verticalalignment='top')
    
#     #text = nx.draw_networkx_labels(G0,pos2, labels = channel_labels, verticalalignment='bottom')

#     #for _,t in text.items():
#     #    t.set_rotation(80)
        
    
#     plt.text(0.8, -0.2, 'Week ' + str(i+1))
#     plt.text(0, 0.85, 'Channels', fontsize=15)
#     plt.text(0, -0.3, 'Users', fontsize=15 )
#     nx.draw_networkx_edges(nw[i+1][x], pos1, width = 0.2, alpha = 0.9, edge_color = 'grey')
#     title = 'fig' + str(i) + '.png'
    
#     camera.snap()
    
# #animation = camera.animate()
 
# #animation.save('/Users/klaudiamur/Google Drive/Commwork/Clients/KTH:SU/gifs/celluloid_minimal.gif', writer = 'imagemagick')
# plt.savefig('/Users/klaudiamur/Google Drive/Commwork/Clients/KTH:SU/channel_user_1/'+title)
# plt.show()


    


##### Do analysis of posts!!! Filter all the posts that get reactions and answers
### ok so thread data does NOT contain the original posts!
### get list of POSTS that GET REACTIONS or ANSWERS

# n_of_answers = threads_data['thread_ts'].value_counts()
# n_of_reactions = reaction_data['ts'].value_counts()

# #nodes = np.unique( answers_to_post['user']) np.unique(reactions_to_post['user'])  np.unique(reactions_to_answers['user']))

# n_nodes = np.zeros(len(n_of_answers))
# n_answers = np.zeros(len(n_of_answers))
# n_reactions = np.zeros(len(n_of_answers))
# n_reaction_to_answers = np.zeros(len(n_of_answers))
# for i in range(len(n_of_answers)):
#     indx = i
#     thread_start_ts = n_of_reactions.index[indx]
#     answers_to_post = threads_data[threads_data['thread_ts']==thread_start_ts].sort_values('ts')
#     reactions_to_post = reaction_data[reaction_data['ts'] == thread_start_ts].sort_values('ts')
#     reactions_to_answers = reaction_data[reaction_data['thread_ts']==thread_start_ts].sort_values('ts')
    
#     nodes = np.unique(answers_to_post['user'].append((data[data['ts'] == thread_start_ts]['user'], reactions_to_post['user'], reactions_to_answers['user'])))
    
#     n_nodes[i] = len(nodes)
#     n_answers[i] = len(answers_to_post)
#     n_reactions[i] = len(reactions_to_post)
#     n_reaction_to_answers[i] = len(reactions_to_answers)
    
# ### ok 26 it is!!!
    
# indx = 26

# #thread_start_ts = n_of_answers.index[indx]
# thread_start_ts = n_of_reactions.index[indx]
# answers_to_post = threads_data[threads_data['thread_ts']==thread_start_ts].sort_values('ts')
# reactions_to_post = reaction_data[reaction_data['ts'] == thread_start_ts].sort_values('ts')
# reactions_to_answers = reaction_data[reaction_data['thread_ts']==thread_start_ts].sort_values('ts')
# nodes = np.unique(answers_to_post['user'].append((data[data['ts'] == thread_start_ts]['user'], reactions_to_post['user'], reactions_to_answers['user'])))

# node_indx = [np.where(users == i)[0][0] for i in nodes]
    





