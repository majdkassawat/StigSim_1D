import xml.etree.ElementTree as ET
#import os
#import numpy as np
#from joblib import Parallel, delayed
#import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  
#from matplotlib import gridspec
#from matplotlib.widgets import Slider, Button, RadioButtons
#import random
#from IPython import get_ipython
#from random import shuffle
#import json
from StigSim import StigSim 
#from __future__ import print_function
import sys
#from mpl_toolkits.mplot3d import Axes3D  

#libraries to be deleted later
#import textwrap as tw
#get_ipython().run_line_magic('matplotlib', 'qt')
#get_ipython().run_line_magic('matplotlib', 'inline')



Rules=["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "p9", "p10", "p11", "p12", "p13", "p14"]
Values=[0, 1]
def generate_combinations_file():
    with open("/home/majd/combinations.txt",'a+') as f:
        combination_index=0
        
        def add_combination_to_file(combination_index,combination):
            combination_string=str(combination_index)+":"
            combination_string=combination_string+" ".join(str(x) for x in combination)+"\n"
            f.write(combination_string)
        
    #        if  combination_index == 2:
    #            sys.exit()
        def generate_combination(seed_rule):
            global combination_index
            if(len(seed_rule)==len(Rules)):
                return 
            for Value in Values:
                old_seed_copy = list(seed_rule.copy())
                old_seed_copy.append(Value)
                new_seed=old_seed_copy
                if len(new_seed)==len(Rules):
                    add_combination_to_file(combination_index,new_seed)
                    combination_index=combination_index+1
                generate_combination(new_seed)
            return
        
        
        generate_combination([])
        
    f.close()
    
    
def generate_xml_files_folders(combinations_file,template_file_path,target_folder,processes_virtual_core):
    def xml_from_rule_combination(rule_combination_string,template_file_path):
        rule_combination_list =rule_combination_string.rstrip("\n\r").split(' ')
        tree = ET.parse(template_file_path)
        root = tree.getroot() #this is the xml node
        objects_node=root.find("objects")
        for obj in objects_node:
            if obj.attrib["type"]=="1":
                rules_node=obj.find("rules")
                for rule in rules_node:
                    rule_id=rule.attrib["id"]
                    index=Rules.index(rule_id)
                    rule.find("actions").set("p",rule_combination_list[index])
        return ET.tostring(root,encoding='utf-8', method='xml') 
    with open(combinations_file,'r') as f:
        def process_combination_string(combination_string):
            combination_number=combination_string.split(':')[0]
            rule_combination_string=combination_string.split(':')[1]
            xml_config_string=xml_from_rule_combination(rule_combination_string,template_file_path)
            simulator= StigSim(xml_config_string)
            simulator.run(target_folder+"/"+combination_number)
#                simulator.run()
#                last_world=simulator.world_record[100]
#                unique, counts = np.unique(last_world, return_counts=True)
#                final_string=str(combination_number)+" "+str(dict(zip(unique, counts)))+"\n"
#                resf.write(final_string)
            
#            xml_config_filepath = target_folder+"/"+combination_number+"/configuration.xml"
#            if not os.path.exists(target_folder+"/"+combination_number):
#                os.makedirs(target_folder+"/"+combination_number)
#            xml_file_handle = open(xml_config_filepath,"wb")
#            xml_file_handle.write(xml_config_string)
#            xml_file_handle.close()
#        Parallel(n_jobs=1)(delayed(process_combination_string)(combination_string) for combination_string in f)
        for combination_string in f:
            
           combination_number=combination_string.split(':')[0]
           if int(combination_number) % 6 == processes_virtual_core:
               print("processing: "+str(combination_number)+", Prcentage: "+ str(round(int(combination_number)*100/1048576,3))+"%")
               process_combination_string(combination_string)
        f.close()
    return
    
#generate_combinations_file()
    
processes_virtual_core=int(sys.argv[1])
generate_xml_files_folders("combinations.txt","template_configuration.xml","Experiments_1D",processes_virtual_core)