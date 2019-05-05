#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:40:28 2019

@author: majd
"""
import numpy as np
from StigSim import StigSim
import os
import time
start = time.time()
"the code you want to test stays here"

def generate_expected_results(path_to_directory):
    #loop the directory and enter every folder, check if there is a config.xml file, 
    #copy the full path, run simulator, dump array.nmpy to the folder.
    for subfolder in os.listdir(path_to_directory):
        test_folder_path=os.path.join(path_to_directory, subfolder)
        config_file_path=os.path.join(test_folder_path, "configuration.xml")
        if os.path.isfile(config_file_path):
            print("processing : ",test_folder_path)
            simulator= StigSim(config_file_path)
            simulator.run()
            array_path=os.path.join(test_folder_path, "array")
            simulator.world.dump(array_path,simulator.world.array)
    return
def check(path_to_directory):
    #loop the directory and enter every folder, check if there is a config.xml file and an array.npy,
    #copy the full path of the config file and run simulator, read the array.npy file and compare with result
    #print result for every test and an overall result
    for subfolder in os.listdir(path_to_directory):
        test_folder_path=os.path.join(path_to_directory, subfolder)
        config_file_path=os.path.join(test_folder_path, "configuration.xml")
        array_file_path=os.path.join(test_folder_path, "array.npy")
        if os.path.isfile(config_file_path) & os.path.isfile(array_file_path):
            print("processing : ",test_folder_path)
            simulator= StigSim(config_file_path)
            simulator.run()
            expected_array = np.load(array_file_path)
            if np.array_equal(expected_array,simulator.world.array):
                print('\x1b[6;30;42m' + 'Success!' + '\x1b[0m')
            else: print('\x1b[0;33;41m' + 'Failed!' + '\x1b[0m')
            
    return


#generate_expected_results("Tests")
#check("Tests")
#
#config_file_name="TestsWithRandom/variable_messaging_test/configuration.xml"
config_file_name="configuration.xml"
#config_file_name="moving_structure.xml"
simulator= StigSim(config_file_name)
#simulator.printSigSim()
##simulator.world.plot()
#simulator.world.savesvg("initial.svg")
simulator.run("xxx")
end = time.time()
print(end - start)
#print(simulator.report[1])
#simulator.run("MovingStructureImageSavingTest")
#		<save format="report" period="1" /> <!--it is always 1 -->
#		<save format="npz" period="1" />
#simulator.dump("aaaaaaaaaa")
#simulator.world.plot()
#simulator.world.savesvg("final.svg")
#res= simulator.world.array