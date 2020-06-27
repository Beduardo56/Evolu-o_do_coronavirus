import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skopt import gp_minimize
from models.models import sir_model, func_logistica

def run_sir_optimizer(ydata: np.array, space:  tuple) -> tuple:
    def func_to_minimize(params):
        length = len(ydata)
        ycalc = sir_model(length, params[0], params[1],params[2], params[3], params[4])
        MAE = mean_absolute_error(ydata, ycalc)
        print(f'MSE: {mean_squared_error(ydata, ycalc)}')
        return MAE ** (1/2)
    
    y = gp_minimize(func_to_minimize, space,  n_calls=100, random_state=1234, n_random_starts=20)
    return (y.x, y.fun ** 2)

def run_logistic_function_optimizer(y_data: np.array, x_data: np.array, bounds: tuple):
    popt, _ = curve_fit(func_logistica, x_data, ydata, bounds=(0,[1, 200]))

    return popt