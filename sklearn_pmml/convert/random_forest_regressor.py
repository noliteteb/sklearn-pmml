from sklearn_pmml.convert import Feature

from sklearn.ensemble import RandomForestRegressor
from sklearn_pmml.convert.model import Schema, ModelMode, RegressionConverter
from sklearn_pmml.convert.tree import DecisionTreeConverter
from sklearn_pmml.convert.utils import estimator_to_converter


import sklearn_pmml.pmml as pmml


class RandomForestRegressionConverter(RegressionConverter):
    def __init__(self, estimator, context):
        super(RandomForestRegressionConverter, self).__init__(estimator, context)
        assert isinstance(estimator, RandomForestRegression),             'This converter can only process RandomForestRegression instances'
        assert len(context.schemas[Schema.OUTPUT]) == 1, 'Only one-label classification is supported'

    def model(self, verification_data=None):
        mining_model = pmml.MiningModel(functionName=ModelMode.REGRESSION.value)
        mining_model.append(self.mining_schema())
        mining_model.append(self.output())
        mining_model.append(self.segmentation())
        if verification_data is not None:
            mining_model.append(self.model_verification(verification_data))
        return mining_model

    def segmentation(self):
        """
        Build a segmentation (sequence of estimators)
        :return: Segmentation element
        """
        segmentation = pmml.Segmentation(multipleModelMethod="weightedAverage")

        for index, est in enumerate(self.estimator.estimators_):
            s = pmml.Segment(id=index)
            s.append(pmml.True_())
            s.append(DecisionTreeConverter(est, self.context, ModelMode.REGRESSION)._model())
            segmentation.append(s)

        return segmentation

estimator_to_converter[RandomForestRegressor] = RandomForestRegressionConverter
