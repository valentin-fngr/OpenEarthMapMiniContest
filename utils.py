# utils 
import matplotlib.pyplot as plt 

def plot_history(train_hist, val_hist): 
    
    x_axis = range(len(train_hist))
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].plot(train_hist, x_axis)
    ax[0].plot(val_hist, x_axis, )