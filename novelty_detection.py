from sklearn import svm
from env import make12

X_train, X_regular, X_novelty = make12.get_datas()



def get_model(x):
    model = svm.OneClassSVM(kernel='rbf', nu=0.01, gamma=0.1)
    model.fit(x)
    return model

model = get_model(X_train)
y_train = model.predict(X_train)
y_regular = model.predict(X_regular)
y_novelty = model.predict(X_novelty)

def get_show_error(predict, true_num, name):
    print(f'Error {name}: {predict[predict == true_num].size}')

get_show_error(y_train, -1, 'Train')
get_show_error(y_regular, -1, 'Regular')
get_show_error(y_novelty, 1, 'novelty')

