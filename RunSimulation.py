import numpy as np
import matplotlib.pyplot as plt
import TempSim
#%matplotlib qt
import time

#Model parameters
nCities = 20
nAgents = 10
nVois = 2*nCities
noTimeSteps = 1

voiPositions = np.ones(nCities)*nVois/nCities