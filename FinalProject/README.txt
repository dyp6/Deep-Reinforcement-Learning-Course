=======================================================================
Forced Migration Simulator (FoMiSimulator)
=======================================================================

This README.txt file is to explain each of the files in this directory and its sub-directories. It will go through each
file or drectory's contents, purpose, and use.

=======================================================================
Python Scripts (.py files)
=======================================================================

distanceCalculation.py
demonstrationDataPrep.py
FoMiEnv.py
TrainAgentNetwork.py
FoMi_Test1.py

=======================================================================
Raw Data Files
=======================================================================

IDP_counts_15-17.csv - Raw Iraq IDP biweekly counts of IDP Families fleeing from one location to a different location from 2015 to 2017
IDP_counts_18.csv (Not used yet) - Raw Iraq IDP biweekly counts of IDP Families fleeing from one locaiton to a different location in 2018
Governorate_state_info.csv - Other location attributes collected at a biweekly rate for each location from 2015 to 2017
locations_defs.csv - Names, Latitudes, and Longitudes of each location in the simulation
start_idp_pops.csv - The initial IDP populations at each of the locations in the simulation

=======================================================================
Processed Data Files
=======================================================================

Files:
    Grouped_IDP_counts_15-17.csv - Biweekly IDP movement counts grouped by origin-destination pairs for each time point and each
location in the simulation
    locations_dists.csv - Road distances between each location, obtained using Google Map API Distance Matrix

Directories:
    External_Environment_Data - Governorate_state_info.csv split into a file for each time point that is being simulated, providing
state information for the environment at each time step in the simulation
    Demonstration_Actions - Sequence of expert actions for each location in the simulation (real movements) for agents to learn from
    Demonstation_Observations - Sequence of expert states encountered to be paired with each of the action sequences

=======================================================================
Model and Results Files
=======================================================================

Models - Trained network structures for each location saved from TrainAgentNetwork.py
Model_Weights - Trained network weights for each location from TrainAgentNetwork.py
Model_Hist - Training histories for each of the networks trained in TrainAgentNetwork.py
TrainResults - Plots showing the simulated values versus the real values for the number of IDPs at each location for the time period 
               used to train the networks
TestResults - Plots showing the simulated values versus the real values for the number of IDPs at each location for the time period
              left out of the training of the networks

=======================================================================
Other Files (hidden)
=======================================================================

.credentials - This file should have a .txt file with your Google Maps API Key for the distance calculations

==========================================================================
Running the Simulation
==========================================================================

Step 1: 
    Define location_defs.csv with your location names, latitudes and longitudes. Then make sure you have the
    other data collected on IDP movements, and state features to be prepped using DemonstrationDataPrep.py     

  Command Line Commands:
    1. Switch current directory to FoMiSimulator directory (Mac/Linux - "cd /path/to/FoMiSimulator"
    2. Run demonstrationDataPrep.py (Mac/Linux - "python demonstrationDataPrep.py")

  OUTPUT:
    Recreates all of the files in each of the following directories, 
    External_Environment_Data, Demonstration_Observations, Demonstration_Actions

Step 2:
    Run DistanceCalculations.py to output the distance matrix for the road distances (km) for each of the locations  
    in the file location_dists.py

  Command Line Commands:
    1. Run distanceCalculation.py (Mac/Linux - "python distanceCalculation.py")

  OUTPUT:
    locations_dists.csv

Step 3:
    Run TrainNetworkAgent.py on the time period you have chosen as your training period to train your neural networks
    to mimic the demonstrated behavior

  Command Line Commands:
    1. Run TrainAgentNetwork.py ("python TrainAgentNetwork.py --batch_size=15 --num_epochs=2000 --learning_rate=0.01")
    NOTE: All command options have default values that are the defaults for the keras package, so they do not have to be
    specified if you would like to use the defaults which are the values specified in the command above

  OUTPUT:
    Replaces the files for all of the location agents in the following directories,
    Models, Model_Weights, Model_Hist

Step 4:
    Run FoMi_Test1.py to test the trained location policy networks in the simulation environment, which will return
    testing and training plots showing the simulated IDP counts at each location with the real values to look at the
    accuracy of the simulated results

  Command Line Commands:
    1. Run FoMi_Test1.py ("python FoMi_Test1.py <net_label>" where net label is some terminology to identify the
    Neural Network Architecture for the networks being trained. For example, the Neural Net Architecture currently specified
    is a simple RNN layer within multiple fully connected layers, and the net_label could be "SimpleRNN" making the command
    "python FoMi_Test1.py SimpleRNN")

  OUTPUT:
    Adds (or replaces, if same neural net architecture) the PNG images in both the TrainResults and TestResults directories