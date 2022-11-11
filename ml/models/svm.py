from sklearn.svm import SVC


def svm(grid_search=False, param_grid=None):
    model = SVC(probability=True, kernel='poly')

    if grid_search:
        if not param_grid:
            param_grid={
                'kernel':['rbf','poly'],
            }
        model = SVC(probability=True)
        model = GridSearchCV(model, param_grid)

    return model
