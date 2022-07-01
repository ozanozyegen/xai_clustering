from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
import pickle, os
from nltk.cluster.kmeans import KMeansClusterer
from sklearn_extra.cluster import KMedoids

def encode(lst):
    ret = ''
    obs = lst
    if obs <= 0.28:
        ret += "A"
    elif obs <= .46:
        ret += "B"
    elif obs <= .64:
        ret += "C"
    elif obs <= .82:
        ret += "D"
    else:
        ret += "E"
    return ret

def directional_distance(p1,p2):
    p1 = list(map(encode, p1))
    p2 = list(map(encode, p2))
    weight = 2
    total = 0
    for i in range(len(p1)-1):
        # ord returns unicode of string, e.g. 1 for A
        dif1=ord(p1[i+1]) - ord(p1[i])
        dif2=ord(p2[i+1])-ord(p2[i])
        if dif1 - dif2 == 0: 
            total+=0

        elif (dif1<0 and dif2>0)or(dif1>0 and dif2<0):
            total+= abs(dif1 - dif2)*weight

        else:
            total+=abs(dif1 - dif2)
    return total

class Model(ABC):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def generate_clusters(self,):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, SAVE_DIR):
        pass

    @abstractmethod
    def load(self, SAVE_DIR):
        pass


class SkModel(Model):
    def __init__(self, config: dict):
        super().__init__(config)

    def generate_clusters(self, X):
        self.model.fit(X)
        cluster_labels = self.model.labels_
        # cluster_labels = self.model.fit_predict(X)
        return cluster_labels

    def predict(self, X):
        return self.model.predict(X)

    def save(self, SAVE_DIR):
        pickle.dump(self.config, open(os.path.join(SAVE_DIR, 'config.pickle'), 'wb'))
        pickle.dump(self.model, open(os.path.join(SAVE_DIR, 'model.h5'), 'wb'))

    def load(self, SAVE_DIR):
        self.config = pickle.load(open(os.path.join(SAVE_DIR, 'config.pickle'), 'rb'))
        self.model = pickle.load(open(os.path.join(SAVE_DIR, 'model.h5'), 'rb'))


class KMeansWrapper(SkModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = KMeans(n_clusters=config['N_CLUSTERS'],
                            random_state=config['RANDOM_SEED'])

    def get_centroids(self, ):
        return self.model.cluster_centers_
    
    def get_inertia(self, ):
        return self.model.inertia_

class CustomKMeansWrapper(SkModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = KMeansClusterer(config['N_CLUSTERS'], distance= directional_distance, repeats=2,
        avoid_empty_clusters=True)
    
    def generate_clusters(self, data):
        clusters = self.model.cluster(data, assign_clusters=True)
        return clusters

    def get_centroids(self):
        return self.model.means()

class PatternKMedoidWrapper(SkModel):
    def __init__(self, config: dict):
        super().__init__(config)
        # kmedoids++
        self.model = KMedoids(config['N_CLUSTERS'], metric="precomputed", init=config["CLUSTER_INIT"], random_state=5)
    
    def generate_clusters(self, dist_matrix):
        self.model = self.model.fit(dist_matrix)
        clusters = self.model.labels_
        return clusters

    def get_centroids(self):
        # will be None because use precomputed
        return self.model.cluster_centers_
    
    def get_inertia(self):
        # will be None because use precomputed
        return self.model.inertia_