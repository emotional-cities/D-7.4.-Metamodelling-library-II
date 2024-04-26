import numpy as np
from smt.sampling_methods import LHS
import pickle


#select the number of samples of the LHS
num_exp = 400

xlimits = np.array([[0.1, 2], [0.1, 2], [0.1, 2]])
sampling = LHS(xlimits=xlimits)

x = sampling(num_exp)

with open('Data/X400.2'+ str(num_exp) + '.pickle', 'wb') as handle: #labeled ones
    pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
