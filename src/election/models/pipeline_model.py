"""Common probability adapter for ordinary sklearn-style pipelines."""
import election.models.custom_model as custom_model_module


class PipelineModel(custom_model_module.custom_model):
    feature_columns = ()
    defaults = {}

    def __init__(self, name):
        super().__init__(name)
        self.pipeline = None

    def train(self, data, configuration=None, fit_context=None):
        context = fit_context or custom_model_module.fit_context()
        custom_model_module.validate_fit(data, context)
        parameters = {**self.defaults, **(configuration or {})}
        self.pipeline = self.fit_pipeline(data, parameters)
        return self._record(data, parameters, context)

    def fit_pipeline(self, data, parameters):
        raise NotImplementedError

    def predict_proba(self, data):
        if self.pipeline is None:
            raise RuntimeError('Call train() before predicting.')
        import numpy as np
        if data.empty:
            return np.empty((0, len(self.classes_)))
        probabilities = self.pipeline.predict_proba(data[list(self.feature_columns) + ['election']])
        return self._align_probabilities(probabilities, self.pipeline.classes_)
