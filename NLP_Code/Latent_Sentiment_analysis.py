import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD


wornet_lemmatizer = WordNetLemmatizer()
titles = [line.rstrip() for line in open('All_books_title.txt')]
stopwords = set(w.rstrip() for w in open('stopwords.txt'))
stopwords = stopwords.union({'introduction', 'volume', 'edition','series', 'application', 'approach', 'card', 'access', 'package', 'plus', 'etext', 'breif', 'val', 'fundamental', 'guide', 'essential', 'printed', 'third', 'second', 'fourth'})


def my_tokenizer(s):
    S = s.lower()
    tokens = nltk.tokenize.word_tokenize(S)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wornet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]

    return tokens


word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []
count = 0
for title in titles:
    try:
        title =title.encode('ascii', 'ignore').decode('utf-8')
        all_titles.append(title)
        tokens = my_tokenizer(title)
        all_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except Exception as e:
        print(e)
        print(title)
        count += 1
print(all_tokens)
print(len(all_tokens))
print(word_index_map)
print(len(word_index_map))
print(len([[1,2,3],[1,2],[3]]))


def tokens_to_vector(tokens):
    X = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        X[i] = 1
        print(t)
        print(i)
        print(X)
    return X


N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N))
i = 0
for tokens in all_tokens:
    X[:, i] = tokens_to_vector(tokens)
    i += 1
    print('hi')
    print(X)

def main():
    svd = TruncatedSVD()
    Z = svd.fit_transform(X)
    plt.scatter(Z[:,0], Z[:,1])
    for i in range(D):
        plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i,1]))
    plt.show()


if __name__ == '__main__':
    main()



