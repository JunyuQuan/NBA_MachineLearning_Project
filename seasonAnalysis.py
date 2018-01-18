# -*- coding: utf-8 -*-
#Analysis of NBA players using season long stats

import numpy as np
import sklearn
from sklearn.decomposition import PCA
import csv
# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import random
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV





# region Data Legend
# 0 Player
# 1 Pos
# 2 Age
# 3 Tm
# 4 G
# 5 GS
# 6 MP
# 7 PER
# 8 TS %
# 9 3PAr
# 10 FTr
# 11 ORB %
# 12 DRB %
# 13 TRB %
# 14 AST %
# 15 STL %
# 16 BLK %
# 17 TOV %
# 18 USG %
# 19 OWS
# 20 DWS
# 21 WS
# 22 WS per  48
# 23 OBPM
# 24 DBPM
# 25 BPM
# 26 VORP
# 27 FG
# 28 FGA
# 29 FG %
# 30 3P
# 31 3PA
# 32 3P%
# 33 2P
# 34 2PA
# 35 2P %
# 36 eFG %
# 37 FT
# 38 FTA
# 39 FT %
# 40 ORB
# 41 DRB
# 42 TRB
# 43 AST
# 44 STL
# 45 BLK
# 46 TOV
# 47 PF
# 48 PTS

#

# region Read 2013-2014 stats csv 
print('Reading NBA season data ...')
seasonData = []
with open('2013_2014_cumulative.csv', newline='') as f:
    read_f = csv.reader(f)
    for row in read_f:
        if seasonData == []:
            seasonData.append(row)
        elif int(row[4]) > 20 and row[0] != seasonData[-1][0] and row[7]>'0':
            seasonData.append(row)


seasonArray = np.array(seasonData)
# print(np.shape(seasonArray))
seasonArray_numericOnly = np.c_[seasonArray[:,4], seasonArray[:,6], seasonArray[0:, 27:]] # Only using basic counting stats
# print(seasonArray_numericOnly)

seasonArray_float = seasonArray_numericOnly.astype(float)

# print(seasonArray_float)

# endregion

# region Run PCA
print('Running PCA ...')
pca = PCA(n_components=2)
seasonArray_float = sklearn.preprocessing.normalize(seasonArray_float, axis=0)

seasonArray_PCA = pca.fit_transform(seasonArray_float[:, :])

# print(seasonArray[:, 0])
# endregion

# region Plot PCA
print('Plotting Results of PCA ...')
fig, ax = plt.subplots(1,1, figsize=(6,6))
# ax = fig.gca(projection='3d')

X = seasonArray_PCA[:,0]
Y = seasonArray_PCA[:,1]
# Z = seasonArray_PCA[:,2]
PER = seasonArray[:, 7].astype(float)
PER = 100*PER/np.max(PER)
label = seasonArray[:, 0]
# print(PER)

#  Plot data
for x,y, lab in zip(X, Y, label):
        ax.scatter(x,y,label=lab)

# Make colormap and apply to data after plotting
colormap = cm.gist_ncar #nipy_spectral, Set1,Paired
colorst = [colormap(i) for i in np.linspace(0, 0.9,len(ax.collections))]
for t,j1 in enumerate(ax.collections):
    j1.set_color(colorst[t])

for i, txt in enumerate(label):
    ax.annotate(txt, (X[i],Y[i]), xytext = (X[i]+.005,Y[i]+.0010))

plt.title('Reduced Dimensionality Data of 2013-2014 Season, Labeled with Player Names', fontsize=18)
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
# ax.legend(fontsize='small')
# plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=

# endregion

# region KMeans
est = KMeans(n_clusters = 15, n_init=50)

# Calculate KMeans
est.fit(seasonArray_PCA)
labels = est.labels_

fig1, ax1 = plt.subplots(1,1)
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

ax1.scatter( X,Y,
               c=labels.astype(np.float), cmap = colormap, edgecolor='k')
plt.title('KMeans, 15 Clusters, 50 Initializations', fontsize = 18)
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
# endregion

#region Visualize KMeans decision boundary


# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = seasonArray_PCA[:, 0].min(), seasonArray_PCA[:, 0].max()
y_min, y_max = seasonArray_PCA[:, 1].min(), seasonArray_PCA[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))

# Obtain labels for each point in mesh. Use last trained model.
Z = est.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
# plt.figure(3)
fig3, ax3 = plt.subplots(1,1, figsize=(6,6))
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=colormap,
           aspect='auto', origin='lower')
X = seasonArray_PCA[:, 0]
Y = seasonArray_PCA[:, 1]
plt.plot(X, Y, 'k.', markersize=2)
# Plot the centroids as a white X
centroids = est.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
# for i, txt in enumerate(label):
#     ax3.annotate(txt, (X[i],Y[i]), xytext = (X[i]+.005,Y[i]+.0010))

plt.title('K-means clustering on the PCA-reduced Statistics from the 2013-2014 season \n'
          'Centroids are marked with white cross', fontsize = 18)
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.xlim(1.1*x_min, 1.1*x_max)
plt.ylim(1.1*y_min, 1.1*y_max)


# endregionY

# region Make 2014-2015 player: Team dict
print('Assigning PCA Stats to 2014-2015 Teams ...')
seasonData2 = []
with open('2014_2015_cumulative.csv', newline='') as f:
    read_f = csv.reader(f)
    for row in read_f:
        if seasonData2 == []:
            seasonData2.append([row[0], row[3]])
        elif row[4] >= '20' and row[0] != seasonData2[-1][0]:
            seasonData2.append([row[0], row[3]])


seasonArray2 = np.array(seasonData2)
# print(np.shape(seasonArray2))
# print(seasonArray2[0,1])

Team_stat = []
# np.shape(seasonArray2)[0]
for TeamName in seasonArray2[:,1]:
    if TeamName in Team_stat:
        pass
        # print('repeat!')
    else:
        Team_stat.append(TeamName)

Team_stat = {el:[0,0] for el in Team_stat} #dict(zip(Team_stat, [[0, 0]]*31))

pca_team = list(zip( seasonArray2[:, 1], seasonArray_PCA[:, 0], seasonArray_PCA[:, 1]))
player_pca_dict = dict(zip(seasonArray[:, 0], pca_team))
# print(player_pca_dict)
for k in player_pca_dict:
    teamStr = player_pca_dict[k][0]
    Team_stat[teamStr][0] += player_pca_dict[k][1]
    Team_stat[teamStr][1] +=  player_pca_dict[k][2]

# cnt = 1
# for k in Team_stat:
#     Team_stat[k].append(cnt)
#     cnt+=1
# endregion

# region Import 2014-2015 Schedule and Organize
print('Organizing 2014-2015 Teams ...')
WL = []
with open('2014_2015.csv', newline='') as f:
    read_f = csv.reader(f)
    for row in read_f:
        WL.append(row)


WL = [[Team_stat.get(item, item) for item in it] for it in WL]

for index, row in enumerate(WL):
    if row[1] > row[3]:
        row.append([1])
    else:
        row.append([0])
    del row[1]
    del row[2]
    # row = [sum(row[0]), sum(row[1]), row[-1][0]]
    # WL[index] = row
    WL[index] = sum(row,[])

# endregion

# region SVM

print('Running SVM ...')
L = len(WL)
testL = int(.9*L)

#seed 15 => 53 vs 47%
# random.seed(45)

teamDataWL_O = [row[:-1] for row in WL]
homeTeamW_O = [row[-1] for row in WL]



# for running multiple iterations with various random seeds in order to find true and false pos rate
TNlist =[]; TPlist =[]; FNlist =[]; FPlist =[]
# for i in range(0,100):
# initialize with original values
teamDataWL = teamDataWL_O
homeTeamW = homeTeamW_O

randNums = random.sample(range(L), testL)
teamDataWL_test = [teamDataWL[x] for x in randNums]
homeTeamW_test = [homeTeamW[x] for x in randNums]
# for index in sorted(randNums, reverse=True):
#     del teamDataWL[index]
#     del homeTeamW[index]
teamDataWL_cv = [teamDataWL[int(ind)] for ind, val in enumerate(teamDataWL) if ind not in randNums]
homeTeamW_cv = [homeTeamW[int(ind)] for ind, val in enumerate(teamDataWL) if ind not in randNums]

CV = []

X_test = np.array(teamDataWL_test)
Y_test = np.array(homeTeamW_test)
X_cv = np.array(teamDataWL_cv)
Y_cv = np.array(homeTeamW_cv)

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_test, Y_test)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

svmModel = svm.SVC(kernel='rbf', C = grid.best_params_[0], gamma  = grid.best_params_[1]) # use auto props
svmModel.fit(X_test, Y_test)
for item in X_cv:
    CV.append(svmModel.predict([item]))

TP =0; FP =0; TN =0; FN = 0
for est, act in zip(CV, Y_cv):
    if np.round(est) == 1:
        TP += 1*(np.round(est) == act)
        FP += 1*(np.round(est) != act)
    elif np.round(est) == 0:
        TN += 1*(np.round(est) == act)
        FN += 1*(np.round(est) != act)

TNlist.append(TN)
TPlist.append(TP)
FNlist.append(FN)
FPlist.append(FP)

FPlist = [x.tolist() for x in FPlist if type(x) == np.ndarray]
TPlist = [x.tolist() for x in TPlist if type(x) == np.ndarray]
FNlist = [x.tolist() for x in FNlist if type(x) == np.ndarray]
TNlist = [x.tolist() for x in TNlist if type(x) == np.ndarray]

confusionMat = np.array([[np.mean(TPlist), np.mean(FPlist)], [np.mean(FNlist), np.mean(TNlist)]])
print(confusionMat)
# for iteratively testing different values of C and gamma
# svmTuneFlag = input('Shall I perform svm tuning? Y/N ')

# if svmTuneFlag == 'Y':
#     maxHit =0
#     bestVals =[]
#     itcnt =0
#
#
#     for i in range(-6, 4):
#         C = 10**i
#         for j in range(-6,4):
#             hits = 0
#             hitRate =0
#             CV = []
#             g = 10**j
#
#             svmModel = svm.SVC(kernel='rbf', C=C, gamma=g)
#             svmModel.fit(X_test, Y_test)
#
#             for item in X_cv:
#                 CV.append(svmModel.predict([item]))
#
#             hits = [1 * (np.round_(x) == y) for x, y in zip(CV, Y_cv)]
#             hitRate = sum(hits) / len(CV)
#
#             if hitRate > maxHit:
#                 maxHit = hitRate
#                 bestVals=[C, g]
#                 bestModel = svmModel
#
#             itcnt += 1
#             print('[%-100s] %d%% ' % ('='*(itcnt//1), 100/100 * itcnt))
#
#     print(bestVals)
#     print(' %.2f %% success' % (hitRate[0]*100) )
#     svmModel = bestModel
# else:
#     C = 10000#10 #10000
#     g = 10 #1000 #100
#     svmModel = svm.SVC(kernel='rbf', C = C, gamma = g)
#     svmModel.fit(X_test, Y_test)
#     for item in X_cv:
#         CV.append(svmModel.predict([item]))
#
#
#     hits = [1*(np.round_(x) == y) for x, y in zip(CV, Y_cv)]
#     hitRate = sum(hits)/len(CV)
#     print(' %.2f %% success' % (hitRate[0]*100) )
#




# hits = [1*(np.round_(x) == y) for x, y in zip(CV, Y_cv)]
# hitRate = sum(hits)/len(CV)
# print(' %.2f %% success' % (hitRate[0]*100) )
#
# dummyhits = [1*(np.round_(x) == y) for x, y in zip([1]*(L-testL), Y_cv)]
# dummyhitRate = sum(dummyhits)/len(CV)
# print(' %.2f %% dummy rate' % (dummyhitRate*100))
# endregion

# region Plot SVM results
print('Plotting SVM results ...')


# xx, yy = np.meshgrid(np.linspace(min(X_test[:,0]),max(X_test[:,0]), 500),
#                      np.linspace(min(X_test[:,1]),max(X_test[:,1]), 500))
xx, yy = np.meshgrid(np.linspace(min(X_test[:,0]),max(X_test[:,0]), 500),
                     np.linspace(min(X_test[:,2]),max(X_test[:,2]), 500))

plt.figure(4)
Z = svmModel.decision_function(np.c_[xx.ravel(), xx.ravel(), yy.ravel(), yy.ravel()])
# Z = svmModel.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.coolwarm)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linetypes='--')
# plt.scatter(X_test[:, 0], X_test[:, 1], s=30, c=Y_test, cmap=plt.cm.coolwarm)
# plt.scatter(X_cv[:, 0], X_cv[:, 1], s=30, c=Y_cv, cmap=plt.cm.coolwarm, edgecolors='w')

plt.scatter(X_test[:, 0], X_test[:, 2], s=30, c=Y_test, cmap=plt.cm.coolwarm)
plt.scatter(X_cv[:, 0], X_cv[:, 2], s=30, c=Y_cv, cmap=plt.cm.coolwarm, edgecolors='w')
plt.title('Cross Section of 4-D Decision Boundary for Game Prediction SVM', fontsize = 18)
plt.xlabel('Team 1 PCA Dimension 1')
plt.ylabel('Team 2 PCA Dimension 1')

plt.show()
# endregion

# region Kmeans with wins
print('Running Kmeans accounting for wins ... ')
teamWin2013 ={}
with open('Records2013-2014.csv', newline='') as f:
    read_f = csv.reader(f)
    for row in read_f:
        teamWin2013.update({row[0]:row[1]})


WinsMins = [int(teamWin2013[team])*int(mins) for mins,team in zip(seasonArray[:,6], seasonArray[:,3])]
WinsMins = np.array(WinsMins)
# WinsMins = sklearn.preprocessing.normalize(WinsMins.reshape(-1,1))
seasonArray_Wins = np.c_[seasonArray_PCA, WinsMins]

st = KMeans(n_clusters = 5, n_init=20)

# Calculate KMeans
st.fit(seasonArray_Wins)
labels2 = st.labels_

fig5 = plt.figure(5)
ax = Axes3D(fig5, rect=[0, 0, .95, 1], elev=48, azim=30)

X = seasonArray_Wins[:,0]
Y = seasonArray_Wins[:,1]
Z = seasonArray_Wins[:,2]


names = seasonArray[:, 0]
ax.scatter(X, Y, Z, c=labels2.astype(np.float), cmap = colormap, edgecolor='k')
ax.set_title('KMeans with Wins', fontsize = 18)
ax.set_xlabel('PCA Dimension 1')
ax.set_ylabel('PCA Dimension 2')
ax.set_zlabel('Win-Minutes')


fig7 = plt.figure(7)
ax = Axes3D(fig7, rect=[0, 0, .95, 1], elev=48, azim=30)
ax.scatter(X, Y, Z, c=labels2.astype(np.float), cmap = colormap, edgecolor='k')
for i in range(len(X)): #plot each point + it's index as text above
    # ax.scatter(X[i], Y[i], Z[i], c=clab[i], cmap=colormap, edgecolor='k')
    ax.text(X[i], Y[i], Z[i],  names[i], zorder=1)
ax.set_title('KMeans with Wins', fontsize = 18)
ax.set_xlabel('PCA Dimension 1')
ax.set_ylabel('PCA Dimension 2')
ax.set_zlabel('Win-Minutes')

# endregion

# region PER vs PCA
print('PCA vs PER ...')
# add minutes to seasonArrary_numericonly then convert to seasonarray_float and normalize
seasonArray_numericOnly = np.c_[seasonArray[:,6], seasonArray_numericOnly]
seasonArray_float = seasonArray_numericOnly.astype(float)
seasonArray_float = sklearn.preprocessing.normalize(seasonArray_float, axis=0)

pca = PCA(n_components=1)
seasonArray_PCA2 = pca.fit_transform(seasonArray_float[:, :])
PER_float = seasonArray[:,7].astype(float)


plt.figure(6)
plt.clf()
z = np.polyfit(PER_float,  seasonArray_PCA2, 1)
# plt.plot(PER_float,np.polyval(z,PER_float),"r--")
plt.plot(PER_float, seasonArray_PCA2, 'bo')
plt.title('PER vs PCA, Inidiviual Player Statistics 2013-2014 NBA Season', fontsize = 18)
plt.xlabel("PER")
plt.ylabel('PCA Dimension 1')

P = PER_float.tolist()
S = [x for y in seasonArray_PCA2.tolist() for x in y]
R = np.corrcoef(P, S)[0,1]
print(R)

#
# seasonArray_PCA2 = np.c_[seasonArray_PCA2, seasonArray[:,0]]
#
#
# def takefirst(elem):
#     return float(elem[0])
#
#
# seasonArray_PCA2 = np.array(sorted(seasonArray_PCA2, key = takefirst))
#
# plt.figure(7)
# plt.clf()
# plt.plot(seasonArray_PCA2[:,0], 'b.')
# plt.xticks(range(0,len(seasonArray_PCA2)), seasonArray_PCA2[:,1], rotation='vertical', fontsize=4)

# endregion

plt.show()