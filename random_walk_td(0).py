#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error

sns.set()

# first implement just temporal diference learning cause experiment was run with values information
# the model is known

def optimize_value_function(policy, values, alpha, iterations):
    
    o_values = [1/6, 2/6, 3/6, 4/6, 5/6]
    history = [mean_squared_error(o_values, values)]

    for i in range(iterations):

        state = 2
        
        while state != -1 and state != 5: # episode ending states
            new_state = state + np.random.choice([-1, 1], p=policiy)

            reward = new_state == 5
            v_s = ( values[new_state] if (new_state != -1 and new_state != 5 ) else 0) 
            values[state] = values[state] + alpha*(reward + v_s - values[state])
            state = new_state

        history += [mean_squared_error(o_values, values)]

    return values, history

ITERARTIONS = int(1e2)

policiy = [0.5, 0.5] # going left or right with uniform probability
values = [0.5]*5

for i, alpha in enumerate([0.01, 0.03, 0.1, 0.3]):

    values = [0.5]*5
    _, history = optimize_value_function(policiy, values, alpha, ITERARTIONS)
    plt.plot(history, label = f'alpha = {alpha}', color = sns.color_palette()[i])
    
plt.legend()

# %%
