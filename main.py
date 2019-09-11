import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.ensemble import GradientBoostingClassifier as GBC, \
    RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


def sigmoid(predicted, rows, estimators):
    pred_arr = np.zeros(shape=(rows, estimators), dtype=np.float64)
    for i, y_pred in enumerate(predicted):
        tmp = 1/(1 + np.exp(-y_pred))
        pred_arr[:, i] = tmp[:, 0]
    return pred_arr


def plotResults(log_tr, log_test):
    plt.figure()
    plt.plot(log_test, 'r', linewidth=2)
    plt.plot(log_tr, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()
    pass

# функция считает потери на каждой итерации
def get_log_lossForEveryIter(y_true, y_pred):
    iterations = y_pred.shape[1]
    log_loss_all = np.empty(shape=(iterations,), dtype=np.float64)
    for i in range(iterations):
        log_loss_all[i] = log_loss(y_true, y_pred[:, i])
    return log_loss_all


data = pd.read_csv('./data/gbm-data.csv')
y = data.to_numpy()[:, 0]
X = data.to_numpy()[:, 1:]
estim_n = 250
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=241
)
# обучаем градиентным бустингом
learning_rates = [1, 0.5, 0.3, 0.2, 0.1]
min_test_loss_iter = 0
for learning_r in learning_rates:
    clf = GBC(n_estimators=estim_n, verbose=False, random_state=241, learning_rate=learning_r)
    clf.fit(X_train, y_train)
    # получаем предсказания для каждого объекта на каждой итерации классификатора
    train_pred = sigmoid(clf.staged_decision_function(X_train), X_train.shape[0], estim_n)
    test_pred = sigmoid(clf.staged_decision_function(X_test), X_test.shape[0], estim_n)
    train_loss = get_log_lossForEveryIter(y_train, train_pred)
    test_loss = get_log_lossForEveryIter(y_test, test_pred)
    min_test_loss_iter = np.argmin(test_loss) + 1
    min_loss_test = test_loss[np.argmin(test_loss)]
    print(f'GBC, learning rate: {learning_r} minTestLoss: {min_loss_test:.2f} on iter: {min_test_loss_iter}')
    plotResults(train_loss, test_loss)

# обучаем random forest
rf = RFC(n_estimators=min_test_loss_iter, random_state=241)
rf.fit(X_train, y_train)
y_rf_pred = rf.predict_proba(X_test)
rf_log_loss = log_loss(y_test, y_rf_pred)
print(f'min log_loss for random forest is: {rf_log_loss:.2f}')