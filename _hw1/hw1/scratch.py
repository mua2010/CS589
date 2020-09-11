def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Calculate the Gini index for a split dataset


def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            breakpoint()
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Select the best split point for a dataset


def get_split(dataset):
    breakpoint()
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


dataset = [[2.771244718, 1.784783929, 0],
           [1.728571309, 1.169761413, 0],
    [3.678319846, 2.81281357, 0],
    [3.961043357, 2.61995032, 0],
    [2.999208922, 2.209014212, 0],
    [7.497545867, 3.162953546, 1],
    [9.00220326, 3.339047188, 1],
    [7.444542326, 0.476683375, 1],
    [10.12493903, 3.234550982, 1],
    [6.642287351, 3.319983761, 1]]
split = get_split(dataset)
print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))




def main():
    X_t = np.genfromtxt('../../Data/x_train.csv', delimiter=',')  # shape = (200000, 29)
    y_t = np.genfromtxt('../../Data/y_train.csv', delimiter=',')  # shape = (200000,)
    def k_fold_validation(max_depth=5, train_x=None, train_y=None):
        print("#######################################################")
        print("Decision Tree FOR Max_Depth="+str(max_depth))
        print("#######################################################")
        X = train_x
        y = train_y
        dt = DecisionTree(max_depth=5)
        kf = KFold(n_splits=5)
        kf.get_n_splits(train_x)
        score = 0

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            startt = time.time()
            print("FOLD=" + str(i + 1))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            score += f1_score(y_test, y_pred)
            ent = time.time()
            print("TIME for this fold is: "+ str(ent-startt) +" Seconds")

        print("Final F1 for all the folds is " + str(score / 5))
    Max_Depth_List = [3, 6, 9, 12, 15]
    for max_depth in Max_Depth_List:
        start = time.time()
        k_fold_validation(max_depth, X_t, y_t)
        end = time.time()
        print("Total Time for all 5 folds with d= "+str(max_depth)+" is "+str(end-start)+" seconds")