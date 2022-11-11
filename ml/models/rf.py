from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def rf(grid_search=False, param_grid=None):
    model = RandomForestClassifier(n_estimators=200)

    if grid_search:
        if not param_grid:
            param_grid={
                'n_estimators':[50, 100, 150, 200],
            }
        model = RandomForestClassifier()
        model = GridSearchCV(model, param_grid)

    return model
