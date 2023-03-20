from nltk.corpus import stopwords
from nltk import word_tokenize
from string import punctuation
from pymorphy2 import MorphAnalyzer
import pandas as pd


def preprocessing(text):
    words = nltk.word_tokenize(text)
    words_filtered = [word.lower()
                    for word in words if
                    len(word) > 2
                    and word not in stops
                    and not re.search(r'\W+', word)
                    and not re.search(r'\d+', word)
                    and not re.search('_', word)
    ]

    words_lemmatized=[morph.parse(w)[0].normal_form for w in words_filtered]
    return (words_lemmatized)


def main():
    print ('raw data pre-processing')

    df = pd.read_csv('data/raw.csv',
                     index_col=False,
                     skip_blank_lines=True,
                     usecols=['CATEGORY', 'MESSAGE'])


    morph = MorphAnalyzer()
    stopwords = stopwords.words('english')
    tokens = word_tokenize(text)











if __name__ == '__main__':
    main()