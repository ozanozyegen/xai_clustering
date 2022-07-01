from interpretable_ts_clustering.models.clustering import KMeansWrapper, CustomKMeansWrapper, PatternKMedoidWrapper
from interpretable_ts_clustering.models.classification import SVMWrapper, DecisionTreeWrapper, XGBWrapper,\
   VggWrapper, FCNWrapper, LSTMWrapper, KNNWrapper
   
clustering_model_loader = dict(
    kmeans=KMeansWrapper,
    directional_kmeans = CustomKMeansWrapper,
    pattern_kmedoid=PatternKMedoidWrapper,
)

classification_model_loader = dict(
    vgg=VggWrapper,
    svm=SVMWrapper,
    decision_tree=DecisionTreeWrapper,
    xgb=XGBWrapper,
    fcn= FCNWrapper,
    lstm = LSTMWrapper,
    nndtw = KNNWrapper
)