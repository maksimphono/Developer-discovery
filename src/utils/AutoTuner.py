#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/home/trukhinmaksim/src')
import numpy as np
from src.utils.Plot3D import Plot3D
from skopt.space import Real, Integer
from skopt import gp_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
import logging
from time import time


class AutoTuner:
    # wrapper around the model, that will perform autotunnig of the model, based on the ranges and types of parameters, that were specified
    # underlying model must contain method "evaluate" for easy evaluation
    EXP_MISSING_METHOD_EVALUATE = Exception("Please define method 'evaluate() -> float' in the target model class")

    @classmethod
    def configLogger(cls, path):
        logger = logging.getLogger(__name__ + '.AutoTuner')  # Unique name
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)


    def __init__(self, modelConstructor, parameters : dict = dict()):
        self.modelConstructor = modelConstructor
        self.parameters = [*parameters]
        self.valuesSpace = [parameter.type(parameter.range[0], parameter.range[1], prior = parameter.prior, name = parameter.name) for parameter in self.parameters]
        self.model = None # will be created later during tunining process
        self.logger = logging.getLogger(__name__ + '.AutoTuner')


        # check if the constructed model object has method 'evaluate'
        model = self.modelConstructor(**{p.name : p.initial for p in self.parameters})
        if not hasattr(model, "evaluate"): raise AutoTuner.EXP_MISSING_METHOD_EVALUATE



    def objective(self, params):
        parameters = dict(zip((p.name for p in self.parameters), params))
        self.model = self.modelConstructor(**parameters)

        self.logger.info(f"Model created with parameters {parameters}")
        # Use cross-validation for a more robust evaluation
        return -self.model.evaluate()


    def tune(self, callsNum = 30, initalPointsNum = 0):
        # will tune model parameters, if 'initalPointsNum' specified, will define initial parameters, otherwise, will use pre-defined initial parameters

        self.logger.info("Tuning process starts")
        start = time()

        if initalPointsNum == 0:
            # use pre-defined initial values for parameters
            x0 = [p.initial for p in self.parameters]
            y0 = [self.objective(x0)]
            result = gp_minimize(self.objective,
                        self.valuesSpace,
                        n_calls = callsNum,
                        n_initial_points = 1,
                        x0 = x0,
                        y0 = y0,
                        random_state = 42
                     )
        else:
            # automatically generate intial values
            result = gp_minimize(self.objective,
                        self.valuesSpace,
                        n_calls = callsNum,
                        n_initial_points = initalPointsNum,
                        random_state = 42
                     )

        self.logger.info(f"Tuning process completed in {(time() - start) / 60} min with results ( {result.fun} : {list(result.x)} )")
        return result

class Param:
    EXP_WRONG_PARAMETER_TYPE = Exception("Specify argument '_type' as class Real or Integer")

    def __init__(self, _name, _type = Real, _range = (0, 10), _initial = 5, _prior = "uniform"):
        if _type not in [Real, Integer]: raise Param.EXP_WRONG_PARAMETER_TYPE

        self.name = _name
        self.type = _type
        self.range = _range
        self.initial = _initial
        self.prior = _prior

"""
class MySVC(SVC):
    @staticmethod
    def create(**kwargs):
        kwargs["random_state"] = 42
        return MySVC(**kwargs)

    def evaluate(self):        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self, X_train, y_train, cv=cv, scoring='accuracy')

        return -np.mean(scores)

t = AutoTuner(MySVC.create, [
    Param(_name = "C", _type = Real, _range = (1e-6, 100.0), _initial = 1),
    Param(_name = "gamma", _type = Real, _range = (1e-6, 10.0), _initial = 1)
])
"""

#plot_convergence(result)
#plt.show()