import numpy as np
from scipy.sparse import coo_matrix
import pyspark
from pyspark.ml.recommendation import ALS as spark_ALS
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
class ALS():
    """Basic class for Alternating Least Squares Model. Error ~ 10x greater than Spark ALS"""
    def __init__(self, n_features=10, lam=0.1, n_jobs=1, max_iter=10, n_blocks=1, tol=0.1):
        self.n_features=n_features
        self.lam=lam
        self.n_jobs=n_jobs
        self.n_blocks=n_blocks
        self.max_iter=max_iter
        self.tol=tol
        self.users=None
        self.items=None
        
    def fit(self, X, y=None):
        R = self._convert_to_sparse(X, y)
        U,V = [np.random.rand(n, self.n_features) for n in R.shape]
        L = np.diag([0.01 for _ in range(10)])
        for _ in range(self.max_iter):
            U = (R @ V) @ np.linalg.inv(V.T @ V + L)
            V = (R.T @ U) @ np.linalg.inv(U.T @ U + L) 
            if self._error(R, U, V.T) < self.tol:
                break
        return U, V
        
    def _error(self, x, u=None, v=None, lam=None):
        if u is None: u = self.users
        if v is None: v = self.items
        if lam is None: lam = self.lam
        if isinstance(x, (tuple,list)):
            x = self._convert_to_sparse(*x)
        if not isinstance(x, np.ndarray): x = x.toarray()
        return np.where(x, (x - u @ v)**2, 0).sum() \
        + (lam) * (sum(a@a.T for a in u) 
                 + sum(a@a.T for a in v))
        
    def _convert_to_sparse(self, X, y):
        cols, rows = [X[:, i].astype(int) for i in range(2)]
        return coo_matrix((y, (cols, rows)))

class SparkALS(ALS):
    """Simple Wrapper class for spark als model. Mimics behaivior of base ALS class.
    Intended for comparison purposes"""
    def __init__(self, **kwargs):
        self.spark = pyspark.sql.SparkSession.builder.getOrCreate()
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        R = np.append(X, y[:, None], 1).tolist()
        S = self.spark.createDataFrame(R, ['user','item','rating'])
        model = spark_ALS(rank=self.n_features, regParam=self.lam, 
          itemCol='item', 
          userCol='user',
          ratingCol='rating')
        model = model.fit(S)
        U = model.userFactors.toPandas()
        V = model.itemFactors.toPandas()
        U = np.array([row for row in U['features']])
        V = np.array([row for row in V['features']])
        return U,V

def random_ratings(n_users, n_items, response_rate=0.1):
    """
    Creates an X,y pair of random ratings.
    """
    R = np.array([])
    for usr in range(n_users):
        for itm in range(n_items):
            if np.random.rand() <= response_rate:
                rtg = np.random.randint(1, 6)
                R = np.append(R, [[usr, itm, rtg]])
    R=R.reshape(-1, 3)
    X,y = R[:, :2], R[:, 2]
    return X,y