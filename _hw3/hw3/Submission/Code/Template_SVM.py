import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import svm


class SVM(object):
    """
    SVM Support Vector Machine
    with sub-gradient descent training.
    """
    def __init__(self, C=1):
        """
        Params
        ------
        n_features  : number of features (D)
        C           : regularization parameter (default: 1)
        """
        self.C = C

    def fit(self, X, y, lr=0.002, iterations=10000):
        """
        Fit the model using the training data.

        Params
        ------
        X           :   (ndarray, shape = (n_samples, n_features)):
                        Training input matrix where each row is a feature vector.
        y           :   (ndarray, shape = (n_samples,)):
                        Training target. Each entry is either -1 or 1.
        lr          :   learning rate
        iterations  :   number of iterations
        """
        n_samples, n_features = X.shape

        # Initialize the parameters wb
        wb = std * np.random.randn(n_features + 1) # +1 for bias term

        # initialize any container needed for save results during training
        w, b = SVM.unpack_wb(wb, n_features)
        self.set_params(w, b)
        best_obj_f = None
        best_wb = None
        obj_list = []

        for i in range(iterations):
            pass
            # calculate learning rate with iteration number i
            lr_t = lr / ((i + 1) ** 0.5)

            # calculate the subgradients
            subgradients = self.subgradient(wb, X, y)

            # update the parameter wb with the gradients
            wb = wb - subgradients * lr_t

            # calculate the new objective function value
            obj_f = self.objective(wb, X, y)

            # compare the current objective function value with the saved best value
            # update the best value if the current one is better
            obj_list.append(obj_f)
            if best_obj_f is None:
                best_obj_f = obj_f
                best_wb = wb
            else:
                if obj_f < best_obj_f:
                    best_obj_f = obj_f
                    best_wb = wb
            # Logging
            #if (i+1)%1000 == 0:
            #    print(f"Training step {i+1:6d}: LearningRate[{lr_t:.7f}], Objective[{f:.7f}]")

        # Save the best parameter found during training 
        self.wb = best_wb
        self.w, self.b = self.unpack_wb(self.wb, n_features)

        # optinally return recorded objective function values (for debugging purpose) to see
        # if the model is converging
        return obj_list

    @staticmethod
    def unpack_wb(wb, n_features):
        """
        Unpack wb into w and b
        """
        w = wb[:n_features]
        b = wb[-1]

        return (w,b)

    def g(self, X, wb):
        """
        Helper function for g(x) = WX+b
        """
        n_samples, n_features = X.shape

        w,b = self.unpack_wb(wb, n_features)
        gx = np.dot(w, X.T) + b

        return gx

    def hinge_loss(self, X, y, wb):
        """
        Hinge loss for max(0, 1 - y(Wx+b))

        Params
        ------

        Return
        ------
        hinge_loss
        hinge_loss_mask
        """
        hinge = 1 - y*(self.g(X, wb))
        hinge_mask = (hinge > 0).astype(np.int)
        hinge = hinge * hinge_mask

        return hinge, hinge_mask


    def objective(self, wb, X, y):
        """
        Compute the objective function for the SVM.

        Params
        ------
        X   :   (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
        y   :   (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.
        Return
        ------
        obj (float): value of the objective function evaluated on X and y.
        """
        n_samples, n_features = X.shape

        # Calculate the objective function value 
        # Be careful with the hinge loss function

        return obj

    def subgradient(self, wb, X, y):
        """
        Compute the subgradient of the objective function.

        Params
        ------
        X   :   (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
        y   :   (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.
        Return
        ------
        subgrad (ndarray, shape = (n_features+1,)):
                subgradient of the objective function with respect to
                the coefficients wb=[w,b] of the linear model 
        """
        n_samples, n_features = X.shape
        w, b = self.unpack_wb(wb, n_features)

        # Retrieve the hinge mask
        hinge, hinge_mask = self.hinge_loss(X, y, wb)

        ## Cast hinge_mask on y to make y to be 0 where hinge loss is 0
        cast_y = - hinge_mask * y

        # Cast the X with an addtional feature with 1s for b gradients: -y
        cast_X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)

        # Calculate the gradients for w and b in hinge loss term
        grad = self.C * np.dot(cast_y, cast_X)

        # Calculate the gradients for regularization term
        grad_add = np.append(2*w,0)
        
        # Add the two terms together
        subgrad = grad+grad_add

        return subgrad

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Params
        ------
        X   :    (ndarray, shape = (n_samples, n_features)): test data

        Return
        ------
        y   :   (ndarray, shape = (n_samples,):
                Predictions with values of -1 or 1.
        """
        # retrieve the parameters wb
        w, b = self.get_params()
        # calculate the predictions
        # y = np.sign(np.dot(np.array(features),self.w)+self.b)

        # return the predictions
        return y

    def get_params(self):
        """
        Get the model parameters.

        Params
        ------
        None

        Return
        ------
        w       (ndarray, shape = (n_features,)):
                coefficient of the linear model.
        b       (float): bias term.
        """
        return (self.w, self.b)

    def set_params(self, w, b):
        """
        Set the model parameters.

        Params
        ------
        w       (ndarray, shape = (n_features,)):
                coefficient of the linear model.
        b       (float): bias term.
        """
        self.w = w
        self.b = b

    


def load_data():
    """
    Helper function for loading in the data

    ------
    # of training samples: 419
    # of testing samples: 150
    ------
    """
    df = pd.read_csv("../../Data/breast_cancer_data/data.csv")
    cols = df.columns
    X = df[cols[2:-1]].to_numpy()
    y = df[cols[1]].to_numpy()
    y = (y=='M').astype(np.int) * 2 - 1

    train_X = X[:-150]
    train_y = y[:-150]

    test_X = X[-150:]
    test_y = y[-150:]

    return train_X, train_y, test_X, test_y

def plot_decision_boundary(clf, X, y, title='SVM'):
    """
    Helper function for plotting the decision boundary

    Params
    ------
    clf     :   The trained SVM classifier
    X       :   (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
    y       :   (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.
    title   :   The title of the plot

    Return
    ------
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    plt.subplot(1, 1, 1)
    
    meshed_data = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(meshed_data)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, linewidth=1, edgecolor='black')
    plt.xlabel('First dimension')
    plt.ylabel('Second dimension')
    plt.xlim(xx.min(), xx.max())
    plt.title(title)
    plt.show()

def ensembles_part_4(kernel, train_X, train_y, test_X, test_y):


def main():
    # Set the seed for numpy random number generator
    # so we'll have consistent results at each run
    np.random.seed(0)

    # Load in the training data and testing data
    train_X, train_y, test_X, test_y = load_data()

    # Ensembles 2.4 
    ensembles_part_4('linear', train_X, train_y, test_X, test_y)
    ensembles_part_4('poly', train_X, train_y, test_X, test_y)
    ensembles_part_4('rbf', train_X, train_y, test_X, test_y)

    # For using the first two dimensions of the data
    #train_X = train_X[:,:2]
    #test_X = test_X[:,:2]

    breakpoint()
    clf = SVM(C=C)
    objs = clf.fit(train_X, train_y, lr=0.002, iterations=10000)

    train_preds  = clf.predict(train_X)
    test_preds  = clf.predict(test_X)

    print(f"f1_score: [{metrics.f1_score(test_y, test_preds)}, {metrics.f1_score(train_y, train_preds)}]")
    #plot_decision_boundary(clf, train_X, train_y)



if __name__ == '__main__':
    main()
