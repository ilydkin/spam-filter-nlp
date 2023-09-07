# 2023.03_NLP message analyzer, spam filter

The project is another small fantasy about ML automation. 
Here you have to ran 3 scripts to get an educated e-mail spam filter model with 98% accuracy, ready for deployment. 

1. You start with `data/raw.csv` file, bearing e-mails with 'spam' denotation where applicable.
2. `preprocessor.py` applies `stopwords`, `regex`, and `lemmatizer` to parse and process the e-mails and make `data/processed.csv`. 
3. `grid_search.ipynb` is a notebook to make `data/grid_search_params.json` bearing configuration parameters for the models you want to try.
4. `main.py` does the main thing.

# main.py
1. It tries different feature transformers on LogisticRegression with default parameters and picks the best.
2. After choosing the best transformer it applies it, while checking for mean cross-validation scores of 12 models, taken in totalling 540 configurations.
3. When the best accuracy score is found, it takes the respective model and its configuration, fits it with the entire data set and serializes it as `data/best_model_trained.pkl`
4. The pickle is ready for deployment.
5. The logs are stored at `data/logs`, while `grid_search_logs.csv` contains the entire GridSearch history, in case you fancy to look for a configuration with best timing or else.

# Self-critic and further steps

Although the script returns a serialized NLP model with some 98% accuracy, it consumes several hours to produce.
The default sklearn LogisticRegression already gives the second-best result, which renders the further 12 models / 540 configurations grid search a redundant endeavour.
The right thing to do would be to create a smart pre-selection of models, depending on the given data.
To be continued...