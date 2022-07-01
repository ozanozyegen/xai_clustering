import shap
import numpy as np


def _select_samples(x, nsamples, random_state):
    if len(x) < nsamples:
        return x
    else:
        return shap.sample(x, nsamples, random_state=random_state)


def get_kernel_shap_feature_importances(model, x, nsamples=100, random_state=0):
    samples = _select_samples(x, nsamples, random_state)
    # Remove logit to evade division by zero error when the pred is 0
    # https://github.com/slundberg/shap/issues/183
    # e = shap.KernelExplainer(model, samples)#, link='logit')
    samples = np.squeeze(np.squeeze(samples))
    e = shap.KernelExplainer(model, samples, link='logit')
    print(samples.shape)
    shap_val = np.array(e.shap_values(samples))
    return shap_val, samples



def get_deepshap_feature_importances(model, x, nsamples=100, random_state=0):
    # https://github.com/slundberg/shap/issues/1110
    import tensorflow.compat.v1.keras.backend as K
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    import shap
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    
    samples = _select_samples(x, nsamples, random_state)
    e = shap.DeepExplainer(model, samples)
    shap_val = np.array(e.shap_values(samples))
    return shap_val, samples


def get_gradientshap_feature_importances(model, x, nsamples=100, random_state=0):
    # https://github.com/slundberg/shap/issues/1110
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    
    samples = _select_samples(x, nsamples, random_state)
    samples = np.squeeze(np.squeeze(samples))
    e = shap.GradientExplainer(model, samples)
    shap_val = np.array(e.shap_values(samples))
    return shap_val, samples

def get_treeshap_feature_importances(model, x, nsamples=100, random_state=0):
    samples = _select_samples(x, nsamples, random_state)
    e = shap.TreeExplainer(model, samples)
    shap_val = np.array(e.shap_values(samples))
    return shap_val, samples


shap_method_dict = dict(
    kernelshap=get_kernel_shap_feature_importances,
    deepshap=get_deepshap_feature_importances,
    gradshap= get_gradientshap_feature_importances,
    treeshap=get_treeshap_feature_importances
)