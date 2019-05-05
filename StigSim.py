#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:10:07 2019

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
import gzip

#from __future__ import print_function
import sys
#from mpl_toolkits.mplot3d import Axes3D  

#libraries to be deleted later
import textwrap as tw
get_ipython().run_line_magic('matplotlib', 'qt')
#get_ipython().run_line_magic('matplotlib', 'inline')

X_uint= 254
P_uint= 255


##########This function is for printing anything(but still not everything is implemented)##########
def object2str(OBJECT):
    OBJECT_str=""
    if isinstance(OBJECT, list):
        OBJECT_str="["
        for key, value in enumerate(OBJECT):
            temp_str="\n"+str(key)+":"+object2str(value)
            OBJECT_str=OBJECT_str+tw.indent(temp_str,"    ")
        OBJECT_str+="]"
    elif isinstance(OBJECT, dict):
        OBJECT_str="\n{"
        for attr, value in OBJECT.items():
            temp_str="\n"+object2str(attr)+":"+object2str(value)
            OBJECT_str=OBJECT_str+tw.indent(temp_str,"    ")
        OBJECT_str+="\n}"
    elif isinstance(OBJECT, tuple):         OBJECT_str="*tuple*"
    elif isinstance(OBJECT, range):         OBJECT_str="*range*"
    elif isinstance(OBJECT, str):   
        OBJECT_str='"'+OBJECT+'"'
    elif isinstance(OBJECT, (np.ndarray, np.generic)):
        OBJECT_str="*numpy array*"+str(OBJECT.dtype)
    elif isinstance(OBJECT, bool):#True and false are also integers! 
        OBJECT_str=str(OBJECT)+"(bool)"
    elif isinstance(OBJECT, int):
        OBJECT_str=str(OBJECT)+"(int)"
    elif isinstance(OBJECT, float):
        OBJECT_str=str(OBJECT)+"(float)"
    elif isinstance(OBJECT, complex):       OBJECT_str="*complex*"
    elif (isinstance(OBJECT, StigSim) or isinstance(OBJECT, Rule) or
         isinstance(OBJECT, Condition) or isinstance(OBJECT, Action) or 
         isinstance(OBJECT, World) or isinstance(OBJECT, Result) or 
         isinstance(OBJECT, Obj)):
         OBJECT_str="<'"+str(type(OBJECT)).split(".",1)[1]+"\n{"
         for attr, value in OBJECT.__dict__.items():
             temp_str="\n"+attr+":"+object2str(value)
             OBJECT_str=OBJECT_str+tw.indent(temp_str,"    ")
         OBJECT_str+="\n}"
    elif (callable(OBJECT)):
        OBJECT_str=OBJECT.__name__+"(function)"
    else: OBJECT_str="*not implemented*"
    return OBJECT_str

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
    
def getdeltas(offset):
    dx=dy=dz=0
    dxstr,dystr,dzstr=tuple(map(str, offset.replace(" ", "").split(",")))
    if dxstr == "X":dx=X_uint
    else: dx = int(dxstr)
    if dystr == "X":dy=X_uint
    else: dy = int(dystr)
    if dzstr == "X":dz=X_uint
    else: dz = int(dzstr)     
    return dx,dy,dz

def generate_offset_rotations(offset_string):
    
    generated_offset_rotations={}
    R=1#we are considering that the object can affect only the adjacent cells
    dx,dy,dz=getdeltas(offset_string)
    if dx == X_uint or dy == X_uint or dz ==X_uint:
        generated_offset_rotations[0]=[dx,dy,dz]
        return generated_offset_rotations
    dummy_array = np.zeros((2*R+1,2*R+1,2*R+1))
    dummy_array[dx+R][dy+R][dz+R]=1
    rotated_arrays=generate_all_rotations(dummy_array)  
    for key, value in rotated_arrays.items():
        offset=np.where(value == 1)
        rotated_offset = [int(offset[0])-R,int(offset[1])-R,int(offset[2])-R]
        generated_offset_rotations[key]=rotated_offset
#    print(offset_string,object2str(generated_offset_rotations))    
    return generated_offset_rotations
    
def str2np(string):
    elements = string.replace(" ", "").split(",")
    R=int(len(elements))
    string_numpy=np.asarray(elements, dtype=np.str).reshape(R)
    string_numpy=np.where(string_numpy=="X", str(X_uint),string_numpy)
    string_numpy=np.where(string_numpy=="P", str(P_uint),string_numpy)
    int_numpy=np.asarray(string_numpy,dtype=np.uint8)
    return int_numpy

def generate_all_rotations(numpy_array):
    generated_rotations={}

    generated_rotations[0]=numpy_array
    generated_rotations[2]=np.flip(numpy_array,0)
    return generated_rotations

class Result:
    def __init__(self,succeeded_bool,data_dict):
        self.succeeded=succeeded_bool
        self.data= data_dict

class World:
    def __init__(self,size_str,padding_str,insertions_dict,distributions_dict):
        self.R = int(padding_str) 
        size = tuple(map(int, size_str.split(",")))
        self.array= np.zeros((size[0],size[1],size[2]),dtype=np.uint8)
        self.Objects=[]
        
        #insertions
        for insertion in insertions_dict:
            X,Y,Z=tuple(map(int,insertion["index"].split(",")))
            self.array[X][Y][Z]=int(insertion["type"])
            self.Objects.append(Obj(np.asarray([X+self.R,Y+self.R,Z+self.R]),insertion["orientation"],insertion["id"]))
        #distributions

        for distribution in distributions_dict:
            empty_indeces=np.asarray(np.transpose(np.where(self.array == 0)),dtype=np.int)
            np.random.shuffle(empty_indeces)
            for i in range(int(distribution["count"])):
                self.array[empty_indeces[i][0]][empty_indeces[i][1]][empty_indeces[i][2]]=int(distribution["type"])
            

        #pad with R
        self.array=np.pad(self.array,self.R,"constant",constant_values=P_uint)
#        self.messages=np.zeros(self.array.shape,dtype=np.uint8)
        def set_action(Object,action_data,rotation_id):
            
            dx =action_data["offset"][rotation_id][0]
            dy =action_data["offset"][rotation_id][1]
            dz =action_data["offset"][rotation_id][2]
            if dx == X_uint:
                dx=random.choice([0,1,-1])
            if dy == X_uint:
                dy=random.choice([0,1,-1])
            if dz == X_uint:
                dz=random.choice([0,1,-1])
                
            x= Object.coords[0]+dx
            y= Object.coords[1]+dy
            z= Object.coords[2]+dz
            
            rotated_object = [int(x),int(y),int(z)]
            result=None
            if self.array[x,y,z]!=P_uint:
                self.array[x,y,z]=action_data["type"]
                for idx,TargetObject in enumerate(self.Objects):
                    if ((TargetObject.coords[0] == x)&(TargetObject.coords[1] == y)&(TargetObject.coords[2] == z)) and action_data["type"]==0:
#                        del self.Objects[idx]
                        self.Objects[idx].to_delete=True
                if action_data["type"]!=0:        
                    coordinations=np.asarray([x,y,z],dtype=np.uint8)
                    Object=Obj(coordinations,0)
                    self.Objects.append(Object)

                    
                result=Result(True,{"type":"set","details":{"Object":Object.coords.tolist(),"rotation_id":rotation_id,"rotated_Object":rotated_object,"type":action_data["type"]}})
            else: result=Result(False,{"type":"set","details":{"Object":Object.coords.tolist(),"rotation_id":rotation_id,"rotated_Object":rotated_object,"reason":"Found Padding","type":action_data["type"]}})
            return result
        def move_action(Object,action_data,rotation_id):
            dx =action_data["offset"][rotation_id][0]
            dy =action_data["offset"][rotation_id][1]
            dz =action_data["offset"][rotation_id][2]
            if dx == X_uint:
                dx=random.choice([0,1,-1])
            if dy == X_uint:
                dy=random.choice([0,1,-1])
            if dz == X_uint:
                dz=random.choice([0,1,-1])
                
            x= Object.coords[0]+dx
            y= Object.coords[1]+dy
            z= Object.coords[2]+dz
            
            rotated_object = [int(x),int(y),int(z)]
            result=None
            if self.array[x,y,z]==0:
                self.array[x,y,z]=self.array[Object.coords[0],Object.coords[1],Object.coords[2]]
                self.array[Object.coords[0],Object.coords[1],Object.coords[2]]=0
                Object.coords[0]=x
                Object.coords[1]=y
                Object.coords[2]=z
                result=Result(True,{"type":"move","details":{"Object":Object.coords.tolist(),"rotation_id":rotation_id,"rotated_Object":rotated_object,"type":self.array[x,y,z]}})
            else: result=Result(False,{"type":"move","details":{"Object":Object.coords.tolist(),"rotation_id":rotation_id,"rotated_Object":rotated_object,"reason":"Cell was not empty","type":self.array[x,y,z]}})
            return result
        def send_action(resolved_parameters_dict):
            Object=resolved_parameters_dict["Object"]
            rotation_id=resolved_parameters_dict["rotation"]
            action_data=resolved_parameters_dict["action_data"]
            dx =action_data["offset"][rotation_id][0]
            dy =action_data["offset"][rotation_id][1]
            dz =action_data["offset"][rotation_id][2]
            if dx == X_uint:
                dx=random.choice([0,1,-1])
            if dy == X_uint:
                dy=random.choice([0,1,-1])
            if dz == X_uint:
                dz=random.choice([0,1,-1])
                
            x= Object.coords[0]+dx
            y= Object.coords[1]+dy
            z= Object.coords[2]+dz
            
            rotated_object = [int(x),int(y),int(z)]
            result=None
            if (self.array[x,y,z]!=P_uint) & (self.array[x,y,z]!=0):
                for ReceiverObject in self.Objects:
                    if ((ReceiverObject.coords[0] == x)&(ReceiverObject.coords[1] == y)&(ReceiverObject.coords[2] == z)):
                        ReceiverObject.inbox.append(action_data["data"])
                result=Result(True,{"type":"send","details":{"Object":Object.coords.tolist(),"rotation_id":rotation_id,"rotated_Object":rotated_object,"type":action_data["data"]}})
            else: result=Result(False,{"type":"send","details":{"Object":Object.coords.tolist(),"rotation_id":rotation_id,"rotated_Object":rotated_object,"reason":"can not send to Padding or Empty","type":action_data["data"]}})
            return result
        def neighborhood_condition(Object,cond_data):
            #slice Object neighborhood from world 
            #match with all elements of cond_data
            #return result
            def match_neighbour_rule_element(actual, rule):
                if(rule==X_uint):return True  
                elif(rule==actual):return True
                elif(actual==1 and rule==0):return True
                else :return False
            neighborhood=None
            if Object.orient == 0:
                neighborhood=self.array[Object.coords[0]+1:Object.coords[0]+1+self.R,Object.coords[1],Object.coords[2]]
            elif Object.orient == 2:
                neighborhood=self.array[Object.coords[0]-self.R:Object.coords[0],Object.coords[1],Object.coords[2]]
            rotation_res=True
            for index_neighbor,neighbor in np.ndenumerate(cond_data[Object.orient]):
                rotation_res=match_neighbour_rule_element(neighborhood[index_neighbor],neighbor)
                if rotation_res == False:
                    break
            if  rotation_res == True:
                result = Result(True,{"type":"neighborhood","details":{"Object":Object.coords.tolist(),"rotation_id":Object.orient}})
                return result
         
            

            return Result(False,{"type":"neighborhood","details":{"Object":Object.coords.tolist()}})
        def p_condition(Object,cond_data):
            p=float(cond_data)
            percentage = random.random()
            res = percentage < p 
            result = Result(res,{"type":"p","details":{"Object":Object.coords.tolist(),"percentage":percentage}})
            return result
        def receive_condition(Object,cond_data):
            res = (cond_data in Object.inbox) or (cond_data == X_uint and len(Object.inbox) > 0)
            result = Result(res,{"type":"received","details":{"Object":Object.coords.tolist(),"received_data":Object.inbox[0] if res else None}})
            return result
        self.actions = {"set":set_action,
                        "move":move_action,
                        "send":send_action}
        self.conditions = {"neighborhood":neighborhood_condition,
                           "p":p_condition,
                           "received_data":receive_condition}
        
#        self.Objects=[]
#        coords=np.asarray(np.transpose(np.where((self.array > 0)&(self.array < 254))),dtype=np.uint8)
#        for coordinations in coords:
#            Object=Obj(coordinations)
#            self.Objects.append(Object)
            
        return
     
    def plot(self,theme):
        # set the colors of each object
        fig = plt.figure(figsize=(12.80, 10.24),dpi=100)
        array_ax = fig.gca(projection='3d')
        array_ax.set_aspect('equal')
        plt.tight_layout()
        array_ax.set_xlim([0,self.array.shape[0]])
        array_ax.set_ylim([0,self.array.shape[1]])
        array_ax.set_zlim([0,self.array.shape[2]]) 
        array2axes(self.array,array_ax,theme)
#        plt.show()
        return
    
    def saveimage(self,fname,array2save,image_format,theme,resolution,elevation,azimuth):
        # set the colors of each object
        plt.ioff()
        fig = plt.figure(figsize=resolution,dpi=100)
        array_ax = fig.gca(projection='3d')
        array_ax.set_aspect('equal')
        plt.tight_layout()
        array_ax.set_xlim([0,array2save.shape[0]])
        array_ax.set_ylim([0,array2save.shape[1]])
        array_ax.set_zlim([0,array2save.shape[2]]) 
        array_ax.view_init(elev=elevation,azim=azimuth)
        array2axes(array2save,array_ax,theme)
        fig.savefig(fname, dpi= 'figure' ,quality= 100, orientation='portrait', papertype=None, format=image_format,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
        fig.clf()
        plt.close()
        return
    
    def dump(self,fname,array2dump):
        np.save(fname, array2dump)
        return
    
class Action:
    def __init__(self,type_str,data_dict):
        self.action_type=type_str
        self.action_data={}
        if self.action_type == "set":
            self.action_data["offset"]=generate_offset_rotations(data_dict["offset"])
            self.action_data["type"]=int(data_dict["type"])
        elif self.action_type == "move":
            self.action_data["offset"]=generate_offset_rotations(data_dict["offset"])
        elif self.action_type == "send":
            self.action_data["offset"]=data_dict["offset"]
            self.action_data["data"]=data_dict["data"]
  
class Condition:
    def __init__(self,type_str,data_str):
        self.cond_type=type_str
        if self.cond_type == "neighborhood":
            self.cond_data=generate_all_rotations(str2np(data_str))
        elif self.cond_type == "p":
            self.cond_data=float(data_str)
        elif self.cond_type == "received_data":
            self.cond_data=int(data_str) if data_str != "X" else X_uint
            
class Rule:
    def __init__(self,conditions_list,actions_list,probability_of_actions,rule_id):
        self.conditions=conditions_list
        self.actions=actions_list
        self.p=probability_of_actions
        self.id=rule_id
class Obj:
    def __init__(self,coordinations,orientation,object_id=0):
        self.coords=coordinations
        self.inbox=[]
        self.timers=[]
        self.orient=orientation
        self.id=object_id
        self.to_delete=False
    
class StigSim:
    
    def __init__(self, config_file_path):
#    def __init__(self, xml_string):
        self.__xml_file_path=config_file_path
        tree = ET.parse(config_file_path)
#        root = ET.fromstring(xml_string)
        root = tree.getroot() #this is the xml node
        self.seed=int(root.attrib["seed"])
        self.length=int(root.attrib["length"])
        bool(root.attrib["debug"])
        self.debug=True if root.attrib["debug"]=="true" else False
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.behaviours={}
        self.visualization={}
        self.world= None
        self.world_record = None
        self.report = {}
        #initiating initiators and self data members
        for sub_node in root:
            node_type = sub_node.tag
            if (node_type == "world"):
                list_of_insertions=[]
                list_of_distributions=[]
                for sub_sub_node in sub_node:
                    sub_sub_nod_tag= sub_sub_node.tag
                    if (sub_sub_nod_tag == "insert"):
                        insert={"type":sub_sub_node.attrib["type"],"index":sub_sub_node.attrib["index"],"orientation":int(sub_sub_node.attrib["orientation"]),"id":int(sub_sub_node.attrib["id"])}
                        list_of_insertions.append(insert)
                    elif (sub_sub_nod_tag == "distribute"):
                        distribution={"type":sub_sub_node.attrib["type"],"count":sub_sub_node.attrib["count"]}
                        list_of_distributions.append(distribution)
                self.world = World(sub_node.attrib["size"],sub_node.attrib["range_of_neighborhood"],list_of_insertions,list_of_distributions)
            elif (node_type == "visualization"):
                if sub_node.attrib["interactive"]=="true":
                    self.visualization={"type":"interactive"}
                else:
                    self.visualization={"type":"non-interactive"}
                self.visualization["outputs"]={}
                self.visualization["viewpoint"]={"elevation":int(sub_node.attrib["elevation"]),"azimuth":int(sub_node.attrib["azimuth"])}
                for sub_sub_node in sub_node:
                    if sub_sub_node.tag == "save":
                        self.visualization["outputs"][sub_sub_node.attrib["format"]]= {}
                        self.visualization["outputs"][sub_sub_node.attrib["format"]]["period"]=int(sub_sub_node.attrib["period"])
                        if sub_sub_node.get("resolution") is not None:
                            self.visualization["outputs"][sub_sub_node.attrib["format"]]["resolution"]=tuple(map(float, sub_sub_node.attrib["resolution"].replace(" ", "").split(",")))
                            if sub_sub_node.get("elevation") is not None:
                                self.visualization["outputs"][sub_sub_node.attrib["format"]]["elevation"]=int(sub_sub_node.attrib["elevation"])
                            if sub_sub_node.get("azimuth") is not None:  
                                self.visualization["outputs"][sub_sub_node.attrib["format"]]["azimuth"]=int(sub_sub_node.attrib["azimuth"])
                    elif sub_sub_node.tag == "theme":
                        self.visualization["theme"]={}
                        for color_node in sub_sub_node:
                            self.visualization["theme"][int(color_node.attrib["type"])]=list(map(float, color_node.attrib["color"].replace(" ", "").split(",")))
                    
            elif (node_type == "objects"):
                for object_node in sub_node:
                    behaviour=[]
                    self.behaviours[int(object_node.attrib["type"])]=behaviour
                    rules_node = object_node.find("rules")
                    if rules_node is not None:
                        for rule_node in rules_node:
                            conditions_list=[]
                            actions_list=[]
                            rule_id=rule_node.attrib["id"]
                            for condition_type_attrib, condition_value_attrib in rule_node.find("conditions").items():
                                condition= Condition(condition_type_attrib,condition_value_attrib)
                                conditions_list.append(condition)
                            actions_node = rule_node.find("actions")
                            probability_of_actions = float(actions_node.attrib["p"])
                            for action_node in actions_node:
                                data_dict={}
                                for parameter_name_attrib, parameter_value_attrib in action_node.items():
                                    data_dict[parameter_name_attrib]=parameter_value_attrib
                                action = Action(action_node.tag,data_dict)
                                actions_list.append(action)
                            rule = Rule(conditions_list,actions_list,probability_of_actions,rule_id)
                            behaviour.append(rule)

        
        self.world_record = np.zeros((self.length+1,self.world.array.shape[0],self.world.array.shape[1],self.world.array.shape[2]),dtype=np.uint8)
    def printSigSim(self):
        print(object2str(self))
    def step(self,step_report,n):
#        if n==26 :print(object2str(self.world.Objects))
        if not self.debug:
            shuffle(self.world.Objects) 
#        if n==26 :
##            print(object2str(self.world.Objects))
#            print(self.world.array[self.world.Objects[32].coords[0],self.world.Objects[32].coords[1],self.world.Objects[32].coords[2]])
        for Object in self.world.Objects:
#            if (n==26): print(object2str(Object))
#            if ((n==26) and Object.id==2):print(object2str(Object))
#                print(self.world.array[Object.coords[0],Object.coords[1],Object.coords[2]])
#            if n==26 : print(Object.id)
            if (self.world.array[Object.coords[0],Object.coords[1],Object.coords[2]] == 0) or Object.to_delete == True:
                continue
            if self.world.array[Object.coords[0],Object.coords[1],Object.coords[2]] == 1:
#                if n==26 : print(object2str(self.world.Objects))
#                if n==26 : print(object2str(Object))
                
                step_report[Object.id]={}
            rules_short_list=[]

            for rule in self.behaviours[self.world.array[Object.coords[0],Object.coords[1],Object.coords[2]]]:
                conditions_results = {}
                rule_variables={}
                for key,condition in enumerate(rule.conditions):
                    conditions_results[key]=self.world.conditions[condition.cond_type](Object,condition.cond_data)     
                #if the conditions_result is okay, proceed to actions and get rotation id
                overall_result=True
                rotation_id=0
                for key,condition_result in conditions_results.items():
                    if condition_result.succeeded == False:
                        overall_result=False
                if overall_result == True:
                    for key,condition_result in conditions_results.items():
                        if condition_result.data["type"]=="neighborhood":
                            rule_variables["rotation"]=condition_result.data["details"]["rotation_id"]
                            rotation_id=condition_result.data["details"]["rotation_id"]
                        if condition_result.data["type"]=="received":
                            rule_variables["received_data"]=condition_result.data["details"]["received_data"]
                    rules_short_list.append(rule)
                    step_report[Object.id][rule.id]=False
            
            if len(rules_short_list) == 0:
                continue
            rule_to_execute = random.choice(rules_short_list)
            step_report[Object.id][rule_to_execute.id]=True
            actions_result={}
            #check probability
            
            if random.random()<rule_to_execute.p:
                for key,action in enumerate(rule_to_execute.actions):
                    if action.action_type=="send":
                        actions_result[key]=self.world.actions[action.action_type](self.ExpressionSolver(Object,{"type":action.action_type,"data":action.action_data},rule_variables))
                    else: actions_result[key]=self.world.actions[action.action_type](Object,action.action_data,Object.orient)
#            if ((n==26) and Object.id==1): print(object2str(self.world.Objects))   
            Object.inbox.clear()
   
        return
            
    def run(self, path=None):
            self.world_record[0]=np.copy(self.world.array)
            for i in range(self.length):
#                print("step: ",i)
                self.report[i+1]={}
                self.step(self.report[i+1],i+1)
                self.world_record[i+1]=np.copy(self.world.array)
            
            if self.visualization["type"]=="interactive":
#                json_report = json.dumps(self.report, indent=4)
#                print("saving : ",os.path.join(path,"report.json"))
#                f = open(os.path.join(path,"report.json"),"w")
#                f.write(json_report)
#                f.close()
                self.interactive_plot()
                print("reached interactive")

            elif (self.visualization["type"]=="non-interactive") & (path is not None):     
                if not os.path.exists(path):
                    os.makedirs(path)
                
                if "svg" in self.visualization["outputs"]:
                    for i in range(0,self.length,self.visualization["outputs"]["svg"]["period"]):
#                        print("saving : ","%04d.svg"% i)
                        self.world.saveimage(os.path.join(path,"%04d.svg"% i),self.world_record[i],"svg",self.visualization["theme"],self.visualization["outputs"]["svg"]["resolution"],self.visualization["viewpoint"]["elevation"],self.visualization["viewpoint"]["azimuth"])
                if "png" in self.visualization["outputs"]:
                    for i in range(0,self.length,self.visualization["outputs"]["png"]["period"]):
#                        print("saving : ","%04d.png"% i)
                        self.world.saveimage(os.path.join(path, "%04d.png"% i),self.world_record[i],"png",self.visualization["theme"],self.visualization["outputs"]["png"]["resolution"],self.visualization["viewpoint"]["elevation"],self.visualization["viewpoint"]["azimuth"])
                if "npy" in self.visualization["outputs"]:
                    for i in range(0,self.length,self.visualization["outputs"]["npy"]["period"]):
#                        print("saving : ","%04d.npy"% i)
                        self.world.dump(os.path.join(path, "%04d" % i),self.world_record[i])
                if "npz" in self.visualization["outputs"]:
                    array2save=self.world_record[0:self.length:self.visualization["outputs"]["npz"]["period"]]
#                    print("saving : ",os.path.join(path,"simulation"))
                    np.savez_compressed(os.path.join(path,"simulation"), simulation=array2save)
                if "report" in self.visualization["outputs"]:
                    json_report = json.dumps(self.report, indent=4)
#                    print("saving : ",os.path.join(path,"report.json"))
                    f = open(os.path.join(path,"report.json"),"w")
                    f.write(json_report)
                    f.close()
                if "compressed_report" in self.visualization["outputs"]:
                    json_report = json.dumps(self.report, indent=4)
                    json_bytes = json_report.encode('utf-8')          

                    with gzip.GzipFile(os.path.join(path,"report.gz"), 'w') as fout:   
                        fout.write(json_bytes)                       
                    
#                    print("saving : ",os.path.join(path,"compressed_report.json"))

            elif self.debug:
                print("debug mode activated")
#            else: print("path for non-interactive output was not defined")     
            return
    def interactive_plot(self):
#        fig = plt.subplots()
        fig = plt.figure(figsize=(12.80, 10.24),dpi=100)   
        gs = fig.add_gridspec(2, 1)
        gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1]) 
        array_ax = fig.add_subplot(gs[0], projection='3d')
        axslider = fig.add_subplot(gs[1])

        axslider.set_facecolor('lightgoldenrodyellow')
        sample_slider = Slider(axslider, 'Sample', 0, self.length, valinit=0, valstep=1)
        
        array_ax.set_aspect('equal')
#        array_ax.set_aspect(3/20)
        
        plt.tight_layout()
        array_ax.set_xlim([0,self.world_record[0].shape[0]])
        array_ax.set_ylim([0,self.world_record[0].shape[1]])
        array_ax.set_zlim([0,self.world_record[0].shape[2]]) 
        array_ax.view_init(elev=self.visualization["viewpoint"]["elevation"],azim=self.visualization["viewpoint"]["azimuth"])
       
        array2axes(self.world_record[0],array_ax,self.visualization["theme"])

        def update(val):
            sample = int(sample_slider.val)
            array2axes(self.world_record[sample],array_ax,self.visualization["theme"])
            
            
        sample_slider.on_changed(update)
        
        def press(event):
            sys.stdout.flush()
            current_sample = sample_slider.val
            if (event.key =="right")&(current_sample<self.length):           
                sample_slider.set_val(current_sample+1)
            elif (event.key =="left")&(current_sample>0):
                sample_slider.set_val(current_sample-1)

        fig.canvas.mpl_connect('key_press_event', press)
        plt.show(block=True)

        
    def ExpressionSolver(self,Object,expression_dict,rule_variables):
#        if target_output == "":
#            print()
        for k in rule_variables:
            exec("%s = %s" % (k, rule_variables[k]))
        resolved_parameters={"action_data":{}}
        if expression_dict["type"] == "send":
            resolved_parameters["action_data"]["offset"]=generate_offset_rotations(expression_dict["data"]["offset"])
            resolved_parameters["action_data"]["data"]=int(eval(expression_dict["data"]["data"]))
        resolved_parameters["Object"]=Object
#        resolved_parameters["action_data"]=expression_dict["data"]
        resolved_parameters["rotation"]=rule_variables["rotation"] if "rotation" in rule_variables else 0
        return resolved_parameters
        
        