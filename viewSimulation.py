#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:05:34 2019

@author: majd
"""
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import gridspec
from matplotlib.widgets import Slider, Button, RadioButtons
import random
from IPython import get_ipython
import os
from random import shuffle
import json

get_ipython().run_line_magic('matplotlib', 'qt')
X_uint= 254
P_uint= 255


def array2axes(array,array_ax,colors_dict):
    array_ax.cla()
    
    n_voxels= np.zeros((array.shape), dtype=bool)
    n_voxels= ((array < P_uint)&(array >0))
    colors = np.zeros(n_voxels.shape+(4,))
    edgecolors = np.zeros(n_voxels.shape+(4,))

    for object_type in np.unique(array):
        if (object_type!=P_uint) & (object_type!=0):
            colors[np.where(array == object_type)]=colors_dict[object_type]
            edgecolors[np.where(array == object_type)]=[i * 0.9 for i in colors_dict[object_type]]

    array_ax.voxels(n_voxels, facecolors=colors,edgecolors=edgecolors)

    return
    
def interactive_plot(world_record):
#        fig = plt.subplots()
    fig = plt.figure(figsize=(12.80, 10.24),dpi=100)   
    gs = fig.add_gridspec(2, 1)
    gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1]) 
    array_ax = fig.add_subplot(gs[0], projection='3d')
    axslider = fig.add_subplot(gs[1])

    axslider.set_facecolor('lightgoldenrodyellow')
    sample_slider = Slider(axslider, 'Sample', 0, 100, valinit=0, valstep=1)
    
    array_ax.set_aspect('equal')
#        array_ax.set_aspect(3/20)
    
    plt.tight_layout()
    array_ax.set_xlim([0,world_record[0].shape[0]])
    array_ax.set_ylim([0,world_record[0].shape[1]])
    array_ax.set_zlim([0,world_record[0].shape[2]]) 
    array_ax.view_init(elev=90,azim=90)
   
    array2axes(world_record[0],array_ax,{1:[0,1,0,1],2:[0,0,1,1],3:[1,0,0,1]})

    def update(val):
        sample = int(sample_slider.val)
        array2axes(world_record[sample],array_ax,{1:[0,1,0,1],2:[0,0,1,1],3:[1,0,0,1]})
        
        
    sample_slider.on_changed(update)
    
    def press(event):
        sys.stdout.flush()
        current_sample = sample_slider.val
        if (event.key =="right")&(current_sample<100):           
            sample_slider.set_val(current_sample+1)
        elif (event.key =="left")&(current_sample>0):
            sample_slider.set_val(current_sample-1)

    fig.canvas.mpl_connect('key_press_event', press)
    plt.show(block=True)


world_record = np.load('/home/majd/IRIDIA/StigSim_1D/xxx/simulation.npz')
#print(world_record["simulation"])
interactive_plot(world_record["simulation"])