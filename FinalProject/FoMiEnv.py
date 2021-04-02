# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:40:53 2020

@author: postd
"""

import tensorflow as tf
import numpy as np

from tf_agents.specs import array_spec
import pandas as pd
from glob import glob

class FoMiEnv:        
    def __init__(self,start_data_file,states_folder,date_list):
        
        self._start_data = pd.read_csv(start_data_file,
                                       index_col=0)
        self._start_data = self._start_data.sort_index()
        
        self._dates = [pd.to_datetime(x) for x in date_list]
        self._dates.sort()
        self.start_date = self._dates[0]
        
        self.locations = [x for x in self._start_data.index]
        
        self._ext_state_files = glob(states_folder+"/*.csv")
        ext_data = pd.read_csv(self._ext_state_files[0],
                               index_col=0).iloc[:,:]
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(len(self.locations),), dtype=np.int32,
            minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(len(self.locations),len(ext_data.columns)),
            dtype=np.int32, minimum=0,name='observation')
        
        self._timeRange = len(self._dates)
        self._location_step_counter = 0
        self._dateIdx = 0
        
        self._start_state = ext_data.copy()
        self._state = self._start_state.copy()
        
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reset(self):
        self._state = self._start_state.copy()
        self._dateIdx = 0
        self._location_step_counter=0
        self._episode_ended = False
        return self._state

    def step(self, action, location_name):
        loc_index = self.locations.index(location_name)
        if self._episode_ended:
            # The last action ended the episode. 
            # Ignore the current action and start
            # a new episode.
            return self.reset()
        
        # Make sure episodes don't go on forever.
        if self._dateIdx >= self._timeRange-1:
            self._episode_ended = True
        
        elif action.shape != (len(self.locations),):
            raise ValueError('`action` shape incorrect, should be ('+\
                             str(len(self.locations))+')')
    
        else:
            # Update the state of the environment
            movements = np.floor(action)
            self._state.loc[location_name,"Families"] = \
                int(self._state.loc[location_name,:].Families\
                    + sum(movements))
            self._location_step_counter += 1
        # If all locations have acted for that timestep increment the
        # date forward by one
        if self._location_step_counter == len(self.locations):
            self._location_step_counter = 0
            self._dateIdx += 1
            self._state.iloc[:,1:] = \
                pd.read_csv(self._ext_state_files[self._dateIdx],
                            index_col=0)\
                    .iloc[:,1:]
        # Return the final state at the end of the time period or the current
        # state if the time period has not ended
        if self._episode_ended:
            return self._state
        else:
            return self._state
            