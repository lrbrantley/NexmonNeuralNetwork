# Identifying People with Wifi

This is a project that I worked on in Spring 2018. It needs additional data, but the current findings show that it was able to uniquely and consistantly identify people (individuals vs each other vs empty room) using only the channel state information of a Nexus 5 phone that had the Nexmon Firmware patch. A router was set in a specific configuration as to not hop channels (for ease of use) and then it sent out beaconing packets. Individuals walked one at a time past the cell phone (Nexus 5). The data was run through matlab to make it into a visual shape, but this is actually an unnessisary step as TensorFlow/Keras should be able to use the csv file directly. 

Further work on this project needs more data. 
More data would be: 
More people, we only had 4. 
Different routers, using the same configuration. 
Different rooms. 

We did use both different patterns walking past the router/cellphone and different gaits (fast walk, slow walk, and carrying a heavy backpack). Each person walked for 30-40 minutes over all (After the data was gathered and fed into the NN this is clearly more then nessisary for 1 room of data, instead of sending in 30 seconds per frame it is likely the same results can be done with 5-8 seconds) with 5-7 minutes per method (multiple walking paths past the router cell phone for the fast/slow walks). 
