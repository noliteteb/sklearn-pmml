from sklearn_pmml.convert import RealNumericFeature
from sklearn_pmml.convert.test.jpmml_test import JPMMLRegressionTest, JPMMLTest, TARGET_NAME
from unittest import TestCase
from sklearn.ensemble import RandomForestRegressor


from RF_Regression_Converter import RandomForestRegressionConverter


class TestRandomForestRegressionParity(TestCase, JPMMLRegressionTest):

    @classmethod
    def setUpClass(cls):
        if JPMMLTest.can_run():
            JPMMLTest.init_jpmml()

    def setUp(self):
        self.model = RandomForestRegressor(
            n_estimators=3,
            max_depth=3
        )
        self.init_data()
        self.converter = RandomForestRegressionConverter(
            estimator=self.model,
            context=self.ctx
        )

    @property
    def output(self):
        return RealNumericFeature(name=TARGET_NAME)
