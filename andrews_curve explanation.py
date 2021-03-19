# Last amended: 19th March, 2021
# Myfolder: C:\Users\Administrator\OneDrive\Documents\manifold_learning
# Objective:
#           i) Explaining Andreas curve
#
"""
Explaining Andrews plots
=========================

Ref: https://en.wikipedia.org/wiki/Andrews_plot

Suppose our data has just three features,
f1,f2,f3. Then at its simplest, andrews-curve is:

f1*sin(t) + f2 * cos(t) + f3 * sin(2t)
where t varies from -np.pi to +np.pi.

For any observation AND for any particular feature,
,say, f1, there is (only) one sine curve. The feature
 value is its amplitude. Once amplitude is decided,
sine-wave, is decided (its angle varies from -np.pi
to +np.pi.)

Another, observation, another value of feature, f1,
and another sine curve.

Now, consider just f1 * sin(t). Feature,
f1, if it is not random, will vary within certain
range, say, 0.2 to 0.4. Thus, the sin plots at
their peaks or troughs, may vary from 0.2 to 0.4.

But, if 'f1' is random, 'f1' can assume any value
and hence peaks will be anything from -0.4 to +0.4
The following experiment shows what happens if
amplitude is random.

"""




# 1.0 Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1.1 Our angle-values. equally spaced
t = np.linspace(-np.pi, np.pi, 100)
# 1.2 Basic sine curve
equ = np.sin(t)
# 1.3 Get one random number as amplitudes
amplitude = np.random.normal(loc = 0.1, scale =1, size = (1,))

# 1.4 Now plot 100 sine
#      plots, as if from, 100,
#       random observation

feature = []
for i in range(100):
    equ = np.sin(t)
    amplitude = np.random.normal(loc = 0.1, scale =1, size = (1,))
    feature.append(amplitude)
    plt.plot(t,amplitude * equ)

#
feature
########## I am done ###############
