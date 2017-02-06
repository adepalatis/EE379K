# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:04:38 2017

@author: shammakabir
"""
import numpy as np
import math
import matplotlib.pyplot as plt


def part_two(n):
    beta_naught = -3
    beta = 0
    numSamples = 2000

    distances = []
    for x in range(numSamples):
        x = np.random.normal(0, 1, n)
        e = np.random.normal(0, 1, n)

        # Generate y_i = beta_naught + e_i
        y = beta_naught + x * beta + e

        # Calculate beta_hat for the dataset
        beta_hat = np.dot(x, y) / np.dot(x, x)
        #beta_hat = (x.transpose() * y) * (x.transpose() * x)
        sd = beta_hat - beta
        distances.append(sd)

    sum = 0
    for num in distances:
        sum += math.pow(num, 2)
    std_dev = math.sqrt(sum / numSamples)
    return std_dev

def main():
    n = [125, 250, 500, 1000, 1500, 2000, 4000, 8500]
    plot_n = []
    for number in n: 
        plot_n.append(part_two(number))
    plot_sqrt = []
    for number in n: 
        temp = np.sqrt(number)
        temp = 1/temp
        plot_sqrt.append(temp)
    print(plot_sqrt)
    plt.plot(n, plot_n, 'r')
    #plt.plot(plot_n)
    plt.plot(n, plot_sqrt, 'g')
    plt.show()
    

    
    
if __name__=='__main__':main()