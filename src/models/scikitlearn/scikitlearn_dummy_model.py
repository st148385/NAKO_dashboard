from models.base_model import BaseModelScitkitLearn


class DummyModelScikitLearn(BaseModelScitkitLearn):
	def __init__(self, ds_info, **kwargs):
		super().__init__(ds_info, **kwargs)

	def fit(self, X, y, **kwargs):
		pass

	def predict(self, X):
		pass

	def explain(self):
		pass
