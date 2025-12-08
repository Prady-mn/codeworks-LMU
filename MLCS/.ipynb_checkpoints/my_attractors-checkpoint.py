import numpy as np
import matplotlib.pyplot as plt

def lorenz(xi = [0., 1., 1.05],dt = 0.02,steps = 10000, sigma = 10,rho = 28,beta = 2.66):
    X = np.empty((steps,3))
    X[0] = xi
    for i in range(steps-1):
        dx = sigma*(X[i,1] - X[i,0])
        dy = X[i,0]*(rho - X[i,2]) - X[i,1]
        dz = X[i,0]*X[i,1] - beta*X[i,2]
        X[i+1,0] = X[i,0] + dx*dt
        X[i+1,1] = X[i,1] + dy*dt
        X[i+1,2] = X[i,2] + dz*dt
        
    return X

def plotTimeSeries(data):
    """
    Plot multiple time series with x, y, z variables in stacked subplots.
    
    Parameters
    ----------
    data : list of lists
        Format: [[X1, args1], [X2, args2], ...]
        where Xi is a numpy array of shape (N, 3) or (N, 4)
        - (N, 3): columns are [x, y, z], time assumed as 0 to N-1
        - (N, 4): columns are [x, y, z, time]
        args is a dictionary of matplotlib plot arguments (e.g., {"alpha": 0.5, "ls": "--"})
        or None/empty dict for default behavior
    
    Returns
    -------
    fig : matplotlib. figure.Figure
        The created figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    variable_names = ['x', 'y', 'z']
    
    for series_data, args_dict in data:
        # Extract time and variables
        if series_data. shape[1] == 3:
            # No time column, use indices
            time = np.arange(len(series_data))
            x, y, z = series_data[:, 0], series_data[:, 1], series_data[:, 2]
        elif series_data.shape[1] == 4:
            # Last column is time
            x, y, z, time = series_data[:, 0], series_data[:, 1], series_data[:, 2], series_data[:, 3]
        else:
            raise ValueError(f"Expected array with 3 or 4 columns, got {series_data.shape[1]}")
        
        # Use empty dict if args_dict is None
        kwargs = args_dict if args_dict else {}
        
        # Plot each variable in its subplot
        variables = [x, y, z]
        for ax, var, var_name in zip(axes, variables, variable_names):
            ax.plot(time, var, **kwargs)
            ax.set_ylabel(var_name)
            ax.grid(True, alpha=0.3)
    
    # Set x-label only on bottom subplot
    axes[-1].set_xlabel('Time')
    
    # Create a single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    return fig