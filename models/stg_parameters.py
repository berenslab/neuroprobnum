import numpy as np
import data_utils
from pathlib import Path

    
names = ['g_Na', 'g_CaT', 'g_CaS', 'g_A', 'g_KCa', 'g_Kd', 'g_H', 'g_leak']

neuron2gs = { # [mS/cm²]
    'ABPD1': [400, 2.5,  6, 50, 10,  100,  0.01,  0.0 ],
    'ABPD2': [100, 2.5,  6, 50,  5,  100,  0.01,  0.0 ],
    'ABPD3': [200, 2.5,  4, 50,  5,   50,  0.01,  0.0 ],
    'ABPD4': [200, 5.0,  4, 40,  5,  125,  0.01,  0.0 ],
    'ABPD5': [300, 2.5,  2, 10,  5,  125,  0.01,  0.0 ],
    'LP1'  : [100, 0.0,  8, 40,  5,   75,  0.05,  0.02],
    'LP2'  : [100, 0.0,  6, 30,  5,   50,  0.05,  0.02],
    'LP3'  : [100, 0.0, 10, 50,  5,  100,  0.0 ,  0.03],
    'LP4'  : [100, 0.0,  4, 20,  0,   25,  0.05,  0.03],
    'LP5'  : [100, 0.0,  6, 30,  0,   50,  0.03,  0.02],
    'PY1'  : [100, 2.5,  2, 50,  0,  125,  0.05,  0.01],
    'PY2'  : [200, 7.5,  0, 50,  0,   75,  0.05,  0.0 ],
    'PY3'  : [200, 10.,  0, 50,  0,  100,  0.03,  0.0 ],
    'PY4'  : [400, 2.5,  2, 50,  0,   75,  0.05,  0.0 ],
    'PY5'  : [500, 2.5,  2, 40,  0,  125,  0.01,  0.03],
    'PY6'  : [500, 2.5,  2, 40,  0,  125,  0.0 ,  0.02],
}
    
n1_panel2gs = { # [mS/cm²]
    'a': [  0,   5,  4, 10, 20,  100,  0.02,  0.03],
    'b': [400, 2.5, 10, 50, 20,    0,  0.04,  0.0 ],
}

n2_panel2neurons = {
    'a': ['ABPD3', 'LP2'],
    'b': ['ABPD3', 'LP5'],
    'c': ['ABPD3', 'PY4'],
    'd': ['ABPD3', 'PY3'],
}

n2_syngs_list = [0, 1, 3, 10, 30, 100] # [nS]

n3_panel2neurons = {
    'a': ['ABPD2', 'LP4', 'PY1'], # a-e are the same
    'b': ['ABPD2', 'LP4', 'PY1'],
    'c': ['ABPD2', 'LP4', 'PY1'],
    'd': ['ABPD2', 'LP4', 'PY1'],
    'e': ['ABPD2', 'LP4', 'PY1'],
    
    'f': ['ABPD4', 'LP5', 'PY5'],
    'g': ['ABPD1', 'LP4', 'PY6'],
    'h': ['ABPD5', 'LP2', 'PY1'],
    'i': ['ABPD1', 'LP4', 'PY5'],
    'j': ['ABPD4', 'LP2', 'PY1'],
}

n3_panel2syngs = {  # [nS]
    'a': np.array([10, 100, 10, 3, 30, 1, 3]),
    'b': np.array([3, 0, 0, 30, 3, 3, 0]),
    'c': np.array([100, 0, 30, 1, 0, 3, 0]),
    'd': np.array([3, 100, 10, 1, 10, 3, 10]),
    'e': np.array([30, 30, 10, 3, 30, 1, 30]),
    
    'f': np.array([3, 100, 10, 1, 10, 3, 10]), # f-j are the same
    'g': np.array([3, 100, 10, 1, 10, 3, 10]),
    'h': np.array([3, 100, 10, 1, 10, 3, 10]),
    'i': np.array([3, 100, 10, 1, 10, 3, 10]),
    'j': np.array([3, 100, 10, 1, 10, 3, 10]),
}        

n3_isslow_list = [0, 1, 0, 1, 0, 0, 0]        

__filename = f'{Path(__file__).parent.absolute()}/stg_neuron2y0.pkl'
try:
    neuron2y0 = data_utils.load_var(__filename)    
except:
    print(f'Could not initialize {__filename}.')

__filename = f'{Path(__file__).parent.absolute()}/stg_n3_panel2y0.pkl'
try:
    n3_panel2y0 = data_utils.load_var(__filename)
except:
    print(f'Could not initialize {__filename}.')
