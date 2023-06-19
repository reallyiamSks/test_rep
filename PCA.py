import pandas as pd
from warnings import filterwarnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as pp

pd.set_option('display.width', None)

filterwarnings('ignore')

a = pd.read_csv('/home/iamsks/Desktop/LUMINAR/breast-cancer.csv')
print(a)

le = LabelEncoder()
a['diagnosis'] = le.fit_transform(a['diagnosis'])

X = a.iloc[:, 2:]
y = a.iloc[:, 1:2]

ss = StandardScaler()
X_sc = ss.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_sc, y, random_state=1, test_size=0.3)

pca = PCA(n_components=3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(pca.explained_variance_ratio_)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
pp.show()
