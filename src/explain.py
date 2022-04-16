import torch
import shap

class ShapPredictor:
    """This class to encapsulate the prediction function that is needed for Kernel Explainer
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, features):
        """Converts numpy array into torch tensors for the the prediction and perform the conversion back to numpy
        Args:
            features (np.ndarray): input feature

        Returns:
            prediction probability (sigmoid) (np.ndarray)
        """
        tensor = torch.tensor(features, dtype=torch.float32)
        y_logits = self.model(tensor).detach().cpu()
        y_preds = torch.sigmoid(y_logits).numpy()
        return y_preds

def get_prediction(shap_predictor, sample_features, feature, nsamples=200):
    """Generate the contribution from each features on a plot

    Args:
        shap_predictor (ShapPredictor): model
        sample_features (np.ndarray): give some sample for the shap algorithm to have a "sense" of the input range
        feature (np.ndarray): feature to perform the explaination on
        nsamples (int, optional): number of iterations to perform. Defaults to 200.

    Returns:
        _type_: plot of the shapy values 
    """
    explainer = shap.KernelExplainer(shap_predictor, sample_features)
    shap_values = explainer.shap_values(feature, nsamples=nsamples, normalize=False)
    return shap.force_plot(explainer.expected_value, shap_values[0], feature)

