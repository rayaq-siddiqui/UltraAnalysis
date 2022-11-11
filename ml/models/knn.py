from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def knn(grid_search=False, param_grid=None):
    model = KNeighborsClassifier(n_neighbors=5)

    if grid_search:
        if not param_grid:
            param_grid={
                'n_neighbors':[5, 10, 15],
            }
        model = KNeighborsClassifier()
        model = GridSearchCV(model, param_grid)

    return model
