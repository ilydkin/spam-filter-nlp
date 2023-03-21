import pandas as pd
from loguru import logger
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, HashingVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

extractors = (
        TfidfVectorizer(),
        TfidfTransformer(),
        HashingVectorizer(),
        CountVectorizer()
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
    MLPClassifier()
    KNeighborsClassifier()
    XGBClassifier()
)

def main ():
        with open('data/grid_search_params.json', 'r') as file:
                grid_search_params = json.load(file)

        df = pd.read_csv('data/processed.csv')

        X = df['tokens']
        y = df['CATEGORY']

        best_score = .0
        best_extractor = None
        best_model = None
        best_params = None
        df_grid_search_logs = pd.DataFrame()

    for extractor in extractors:
        logreg = LogisticRegression()
        clf = Pipeline([
                ('extractor', extractor),
                ('logreg', LogisticRegression())]
        )
        kfold = KFold(n_splits=5, shuffle=True)
        cv_results = cross_validate (clf, X, y, cv=kfold, scoring=['accuracy'])

        if cv_results.test_accuracy.mean() > best_score:
                best_score = cv_results.test_accuracy.mean()
                best_extractor = extractor

        logger.info(f'best accuracy {cv_results.test_accuracy.mean()} on default LogisiticRegression was achieved by {best_extractor}')

    for model in models:
        GS = GridSearchCV(
            estimator=model,
            param_grid=grid_search_params[type(model).__name__],
            scoring='accuracy',
            cv=5,
            verbose=4
        )

        tokenized_features = best_extractor.fit_transform(X)
        train, test, target, target_test = train_test_split (tokenized_features, y, test_size=.2, random_state=34)
        GS.fit(train, target)
        df_logs = pd.DataFrame(GS.cv_results_)
        df_grid_search_logs = pd.concat([df_grid_search_logs, df_logs])

        if GS.best_score_ > best_score:
            best_score = GS.best_score_
            best_model = GS.best_estimator_
            best_params = GS.best_params_

        logger.info(f'best model: {type(best_model).__name__}, '
                     f'accuracy: {best_score:.4f}'
                     f'best parameters: {best_params}')

        df_grid_search_logs.to_csv('data/grid_search_logs.csv')
        best_model.fit(tokenized_features, y)
        joblib.dump(best_model, 'data/best_model_trained.pkl')