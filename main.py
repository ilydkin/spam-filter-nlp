import pandas as pd
from loguru import logger
import joblib
import json

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, HashingVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

transformers = (
    TfidfVectorizer(),
    HashingVectorizer(),
    CountVectorizer(),
)

models = (
    LogisticRegression(),
    SGDClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier(),
    AdaBoostClassifier(),
    SVC(),
    GaussianNB(),
    MLPClassifier(),
    KNeighborsClassifier(),
    XGBClassifier()
)

logger.add("data/logs/info_{time}.log",
           colorize=True,
           format="<green>{time}</green> <level>{message}</level>",
           level="INFO")


@logger.catch()
def main():
    with open('data/grid_search_params.json', 'r') as file:
        grid_search_params = json.load(file)

    df = pd.read_csv('data/processed.csv')

    X = df['tokens']
    y = df['CATEGORY']

    best_score = .0
    best_transformer = None
    best_model = LogisticRegression()
    best_params = 'default'
    df_grid_search_logs = pd.DataFrame()

    for transformer in transformers:
        try:
            clf = Pipeline([
                ('transformer', transformer),
                ('logreg', LogisticRegression())]
            )
            kfold = KFold(n_splits=5, shuffle=True)
            cv_results = cross_validate(clf, X, y, cv=kfold, scoring=['accuracy'])
            logger.info(f'{transformer} mean accuracy: {cv_results["test_accuracy"].mean()}')
        except:
            logger.debug(f'error while trying {transformer}')
            continue
        else:
            if cv_results["test_accuracy"].mean() > best_score:
                best_score = cv_results["test_accuracy"].mean()
                best_transformer = transformer

    logger.info(f'best mean accuracy {best_score} '
                f'on default LogReg was achieved by {best_transformer}')

    Xt = best_transformer.fit_transform(X)

    for model in models:
        GS = GridSearchCV(
            estimator=model,
            param_grid=grid_search_params[type(model).__name__],
            scoring='accuracy',
            cv=5,
            verbose=4
        )

        try:
            GS.fit(Xt, y)
            df_logs = pd.DataFrame(GS.cv_results_)
            df_grid_search_logs = pd.concat([df_grid_search_logs, df_logs])
            logger.info(f'model: {model} best score: {GS.best_score_}')

        except:
            logger.debug(f'error while trying {model}')
            continue

        else:
            if GS.best_score_ > best_score:
                best_score = GS.best_score_
                best_model = GS.best_estimator_
                best_params = GS.best_params_

    logger.info(f'best model: {best_model}, '
                f'accuracy: {best_score:.4f}'
                f'best parameters: {best_params}')

    df_grid_search_logs.to_csv('data/grid_search_logs.csv')
    best_model.fit(Xt, y)
    joblib.dump(best_model, 'data/best_model_trained.pkl')


if __name__ == '__main__':
    main()
