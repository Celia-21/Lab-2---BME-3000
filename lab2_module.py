# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:49:58 2023

@author: celia
"""
#import packages
import numpy as np
from matplotlib import pyplot as plt


#Part 2, define variable with nested for loops
def get_convolved_signal(input_signal, system_impulse):
    '''
    Parameters
    Function takes in two inputs and makes a 1D array of zeros that iterate through each elemlent in input_signal and system_impulse to convolve the signals
    input_signal : 1D array of floats
        this models the provided input signal, x(t)
    system_impulse : 1D array of floats
        system impulse function, h(t), a rectangular pulse function

    Returns
    ---
    my_convolved_signal : 1D array of floats
        convolved signal of input_signal and system_impulse
 

    '''
    my_convolved_signal=np.zeros((len(input_signal)+len(system_impulse)-1))

    for n_index in np.arange(len(my_convolved_signal)):
        for k_index in np.arange(len(input_signal)):
            if (n_index-k_index <0) | (len(system_impulse)<=(n_index-k_index)):
                continue
            my_convolved_signal[n_index]=(my_convolved_signal[n_index])+(input_signal[k_index])*(system_impulse[n_index-k_index])

    return my_convolved_signal


#Part 3
def run_drug_simulations(input_signal, system_impulse, dt, label):
    """

    Parameters
    function that takes four inputs that convolves input_signal and system_impulse and plots the result
    input_signal : 1D array of floats
        this models the provided input signal, x(t)
    system_impulse : 1D array of floats
        system impulse function, h(t), a rectangular pulse function
    dt : float
        time interval 
    label : string
        describes the current line we are plotting, string will appear in the legend

    Returns
    -------
    None.


    """
    system_output = np.convolve(input_signal, system_impulse)
    time_array = np.arange(0, len(system_output))*dt
    plt.figure(3)
    plt.plot(time_array, system_output, label=label)
    