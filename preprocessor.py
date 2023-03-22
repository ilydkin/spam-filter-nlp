import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
import pymorphy2

from loguru import logger
from datetime import datetime

morph = pymorphy2.MorphAnalyzer()
stops = stopwords.words('english')
logger.add ("data/logs/info_{time}.log",
            colorize=True,
            format="<green>{time}</green> <level>{message}</level>",
            level="INFO")


@logger.catch
def preprocessor(text):
    words = nltk.word_tokenize(text)
    words_filtered = [word.lower()
                      for word in words if
                      len(word) > 2
                      and word not in stops
                      and not re.search(r'\W+', word)
                      and not re.search(r'\d+', word)
                      and not re.search('_', word)
                      ]
    words_lemma = [morph.parse(w)[0].normal_form for w in words_filtered]
    return words_lemma


@logger.catch
def main():
    start = datetime.now()
    df = pd.read_csv('data/raw.csv',
                     index_col=False,
                     skip_blank_lines=True,
                     usecols=['CATEGORY', 'MESSAGE'])
    df['tokens'] = df['MESSAGE'].apply(preprocessor)
    df1= df.dropna()
    df1[['tokens','CATEGORY']].to_csv('data/processed.csv', index=False)

    end = datetime.now()
    logger.info(f'preprocessor.py ran in {(end-start).total_seconds()} seconds')


if __name__ == '__main__':
    main()














if __name__ == '__main__':
    main()