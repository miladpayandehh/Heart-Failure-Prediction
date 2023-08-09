import numpy as np
import pandas as pd
import seaborn as sb
import sklearn.metrics as met
import sklearn.neighbors as ne
import matplotlib.pyplot as plt
import sklearn.linear_model as li
import sklearn.preprocessing as pp
import sklearn.neural_network as nn
import sklearn.model_selection as ms


# Section 1: Dataset Statistical details

# np.random.seed(0)
# plt.style.use('ggplot')
#
# Nu = [[0, 'age'],
#       [2, 'creatinine_phosphokinase'],
#       [4, 'ejection_fraction'],
#       [6, 'platelets'],
#       [7, 'serum_creatinine'],
#       [8, 'serum_sodium']]
#
# Ca = [[1, 'anaemia'],
#       [3, 'diabetes'],
#       [5, 'high_blood_pressure'],
#       [9, 'sex'],
#       [10, 'smoking']]
#
# NuNames = [i[1] for i in Ca]
#
# DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
# DF.drop(['time'], axis=1, inplace=True)
#
# with pd.option_context('display.max_rows', None,'display.max_columns', None):
#     print(DF.describe())


# Section 2: Numerical Features Correlation

# np.random.seed(0)
# plt.style.use('ggplot')
#
# Nu = [[0, 'age'],
#       [2, 'creatinine_phosphokinase'],
#       [4, 'ejection_fraction'],
#       [6, 'platelets'],
#       [7, 'serum_creatinine'],
#       [8, 'serum_sodium']]
#
# Ca = [[1, 'anaemia'],
#       [3, 'diabetes'],
#       [5, 'high_blood_pressure'],
#       [9, 'sex'],
#       [10, 'smoking']]
#
# NuNames = [i[1] for i in Nu]
#
# DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
# DF.drop(['time'], axis=1, inplace=True)
#
# D = DF.to_numpy()
#
# Co = DF[NuNames].corr()
# sb.heatmap(Co, annot=True, fmt='.4f',
#            vmin=-1, vmax=+1,
#            cmap='RdYlGn',
#            xticklabels=NuNames, yticklabels=NuNames)
# plt.show()


# Section 3: Categorical Features Correlation

# np.random.seed(0)
# plt.style.use('ggplot')
#
# Nu = [[0, 'age'],
#       [2, 'creatinine_phosphokinase'],
#       [4, 'ejection_fraction'],
#       [6, 'platelets'],
#       [7, 'serum_creatinine'],
#       [8, 'serum_sodium']]
#
# Ca = [[1, 'anaemia'],
#       [3, 'diabetes'],
#       [5, 'high_blood_pressure'],
#       [9, 'sex'],
#       [10, 'smoking']]
#
# NuNames = [i[1] for i in Ca]
#
# DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
# DF.drop(['time'], axis=1, inplace=True)
#
# D = DF.to_numpy()
#
# Co = DF[NuNames].corr()
# sb.heatmap(Co, annot=True, fmt='.4f',
#            vmin=-1, vmax=+1,
#            cmap='RdYlGn',
#            xticklabels=NuNames, yticklabels=NuNames)
# plt.show()


# Section 4: Correlation between Numerical Features and label

# np.random.seed(0)
# plt.style.use('ggplot')
#
# Nu = [[0, 'age'],
#       [2, 'creatinine_phosphokinase'],
#       [4, 'ejection_fraction'],
#       [6, 'platelets'],
#       [7, 'serum_creatinine'],
#       [8, 'serum_sodium']]
#
# Ca = [[1, 'anaemia'],
#       [3, 'diabetes'],
#       [5, 'high_blood_pressure'],
#       [9, 'sex'],
#       [10, 'smoking']]
#
# NuNames = [i[1] for i in Nu]
#
# DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
# DF.drop(['time'], axis=1, inplace=True)
#
# D = DF.to_numpy()
#
# for i in Nu:
#     X = D[:, i[0]]
#     Y = D[:, -1]
#     A = X[Y==0]
#     B = X[Y==1]
#     Bins = np.linspace(np.min(X), np.max(X), num=25)
#     plt.hist([A, B], bins=Bins, color=['b', 'r'], label=['0', '1'])
#     plt.xlabel(i[1])
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.show()


# Section 5: Correlation between Categorical Features and label

# np.random.seed(0)
# plt.style.use('ggplot')
#
# Nu = [[0, 'age'],
#       [2, 'creatinine_phosphokinase'],
#       [4, 'ejection_fraction'],
#       [6, 'platelets'],
#       [7, 'serum_creatinine'],
#       [8, 'serum_sodium']]
#
# Ca = [[1, 'anaemia'],
#       [3, 'diabetes'],
#       [5, 'high_blood_pressure'],
#       [9, 'sex'],
#       [10, 'smoking']]
#
# NuNames = [i[1] for i in Nu]
# CaNames = [i[1] for i in Ca]
#
# DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
# DF.drop(['time'], axis=1, inplace=True)
#
# D = DF.to_numpy()
#
# for i in Ca:
#     X = D[:, i[0]]
#     Y = D[:, -1]
#     M = np.zeros((2, 2))
#     for x, y in zip(X, Y):
#         M[int(x), int(y)] += 1
#     sb.heatmap(M, annot=True, fmt='.0f',
#                cmap='RdYlGn',
#                xticklabels=[0, 1], yticklabels=[0, 1])
#     plt.xlabel('DEATH_EVENT')
#     plt.ylabel(i[1])
#     plt.show()


# Section 6: Logistic Regression Model - Accuracy

# np.random.seed(0)
# plt.style.use('ggplot')
#
# DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
# DF.drop(['time'], axis=1, inplace=True)
#
# D = DF.to_numpy()
#
# X = D[:, :-1]
# Y = D[:, -1].reshape((-1, 1))
#
# trX0, teX0, trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=0)
#
# Scaler = pp.MinMaxScaler(feature_range=(-1, +1))
# trX = Scaler.fit_transform(trX0)
# teX = Scaler.transform(teX0)
#
# LR = li.LogisticRegression(random_state=0)
# LR.fit(trX, trY)
#
# trAc = LR.score(trX, trY)
# teAc = LR.score(teX, teY)
#
# print(f'{trAc = }')
# print(f'{teAc = }')


# Section 7: Logistic Regression Model - Accuracy (after balancing)

# np.random.seed(0)
# plt.style.use('ggplot')
#
# DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
# DF.drop(['time'], axis=1, inplace=True)
#
# D = DF.to_numpy()
#
# X = D[:, :-1]
# Y = D[:, -1].reshape((-1, 1))
#
# trX0, teX0, trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=5)
#
# Scaler = pp.MinMaxScaler(feature_range=(-1, +1))
# trX = Scaler.fit_transform(trX0)
# teX = Scaler.transform(teX0)
#
# n0 = trY[trY == 0].size
# n1 = trY[trY == 1].size
#
# W = {0: n1/(n0+n1), 1: n0/(n0+n1)}
#
# LR = li.LogisticRegression(random_state=0, class_weight = W)
# LR.fit(trX, trY)
#
# trAc = LR.score(trX, trY)
# teAc = LR.score(teX, teY)
#
# print(f'{trAc = }')
# print(f'{teAc = }')


# Section 8: LR Classification Report

np.random.seed(0)
plt.style.use('ggplot')

DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
DF.drop(['time'], axis=1, inplace=True)

D = DF.to_numpy()

X = D[:, :-1]
Y = D[:, -1].reshape((-1, 1))

trX0, teX0, trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=5)

Scaler = pp.MinMaxScaler(feature_range=(-1, +1))
trX = Scaler.fit_transform(trX0)
teX = Scaler.transform(teX0)

n0 = trY[trY == 0].size
n1 = trY[trY == 1].size

W = {0: n1/(n0+n1), 1: n0/(n0+n1)}

LR = li.LogisticRegression(random_state=0, class_weight = W)
LR.fit(trX, trY)

trAc = LR.score(trX, trY)
teAc = LR.score(teX, teY)

print(f'{trAc = }')
print(f'{teAc = }')

trPr = LR.predict(trX)
tePr = LR.predict(teX)

trCR = met.classification_report(trY, trPr)
teCR = met.classification_report(teY, tePr)

print('_' * 50)
print(f'Train CR:\n{trCR}')
print('_' * 50)
print(f'Test CR:\n{teCR}')
print('_' * 50)


# Section 9: KNN Model

# def PrintReport(Model, trX, teX, trY, teY):
#     trAc = Model.score(trX, trY)
#     teAc = Model.score(teX, teY)
#     trPr = Model.predict(trX)
#     tePr = Model.predict(teX)
#     trCR = met.classification_report(trY, trPr)
#     teCR = met.classification_report(teY, tePr)
#     print(f'{trAc = }')
#     print(f'{teAc = }')
#     print('_' * 50)
#     print(f'Train CR:\n{trCR}')
#     print('_' * 50)
#     print(f'Test CR:\n{teCR}')
#     print('_' * 50)
#
# np.random.seed(0)
# plt.style.use('ggplot')
#
# DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
# DF.drop(['time'], axis=1, inplace=True)
#
# D = DF.to_numpy()
#
# X = D[:, :-1]
# Y = D[:, -1].reshape((-1, 1))
#
# trX0, teX0, trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=5)
#
# Scaler = pp.MinMaxScaler(feature_range=(-1, +1))
# trX = Scaler.fit_transform(trX0)
# teX = Scaler.transform(teX0)
#
# n0 = trY[trY == 0].size
# n1 = trY[trY == 1].size
#
# W = {0: n1/(n0+n1), 1: n0/(n0+n1)}
# KNN = ne.KNeighborsClassifier(n_neighbors=5, weights='distance')
# KNN.fit(trX, trY)
# PrintReport(KNN, trX, teX, trY, teY)


# Section 10: ANN Model

# def PrintReport(Model, trX, teX, trY, teY):
#     trAc = Model.score(trX, trY)
#     teAc = Model.score(teX, teY)
#     trPr = Model.predict(trX)
#     tePr = Model.predict(teX)
#     trCR = met.classification_report(trY, trPr)
#     teCR = met.classification_report(teY, tePr)
#     print(f'{trAc = }')
#     print(f'{teAc = }')
#     print('_' * 50)
#     print(f'Train CR:\n{trCR}')
#     print('_' * 50)
#     print(f'Test CR:\n{teCR}')
#     print('_' * 50)
#
# np.random.seed(0)
# plt.style.use('ggplot')
#
# DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
# DF.drop(['time'], axis=1, inplace=True)
#
# D = DF.to_numpy()
#
# X = D[:, :-1]
# Y = D[:, -1].reshape((-1, 1))
#
# trX0, teX0, trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=5)
#
# Scaler = pp.MinMaxScaler(feature_range=(-1, +1))
# trX = Scaler.fit_transform(trX0)
# teX = Scaler.transform(teX0)
#
# n0 = trY[trY == 0].size
# n1 = trY[trY == 1].size
#
# W = {0: n1/(n0+n1), 1: n0/(n0+n1)}
# MLP = nn.MLPClassifier(hidden_layer_sizes=(30), activation='relu', max_iter=10, random_state=0)
# MLP.fit(trX, trY)
# PrintReport(MLP, trX, teX, trY, teY)