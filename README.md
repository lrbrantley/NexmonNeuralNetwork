# Identifying People with Wifi

This project was able to identify the 4 individuals and the empty room with 86 percent accuracy using only the channel state information of a Nexus 5 phone that had the Nexmon Firmware patch. A router was set in a specific configuration as to not hop channels (for ease of use) and then it sent out beaconing packets. Individuals walked one at a time past the cell phone (Nexus 5).

The data from the Nexus 5 was a .pcap file (see the pcap-data folder). This file was put into the csi_parser.py script to convert it to a .csv. In the original experiment the csv underwent normalization and was changed into an image in Matlab. However, Tensorflow/Keras can handle .csv files directly and can do normalization itself.

## Findings from Spring 2018

Our findings seem to indicate that the Neural Network identifies humans based on Gait. 

![Gait](/paper/images/Gait.png)

Here is the Neural Network that was most effective.

![Model](/paper/images/model_plot.png)

More information in the .pdf inside the paper folder

## Information about Spring 2018 Data

![Table and Patterns](/paper/images/table.jpg)

Each person walked for 30-40 minutes over all (After the data was gathered and fed into the NN this is clearly more then nessisary for 1 room of data, instead of sending in 30 seconds per frame it is likely the same results can be done with 5-8 seconds) with 5-7 minutes per method (multiple walking paths past the router cell phone for the fast/slow walks). 

