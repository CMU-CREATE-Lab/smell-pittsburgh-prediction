# smell-pittsburgh-dataset-v1

This is the first version of the Smell Pittsburgh dataset from 10/31/2016 (month/day/year) to 9/30/2018, including the following zipcodes in the Pittsburgh region in Pennsylvania, USA:

15221, 15218, 15222, 15219, 15201, 15224, 15213, 15232, 15206, 15208, 15217, 15207, 15260, 15104

This dataset is used for the Smell Pittsburgh paper below:

Yen-Chia Hsu, Jennifer Cross, Paul Dille, Michael Tasota, Beatrice Dias, Randy Sargent, Ting-Hao (Kenneth) Huang, and Illah Nourbakhsh. 2020. Smell Pittsburgh: Engaging Community Citizen Science for Air Quality. ACM Transactions on Interactive Intelligent Systems. 10, 4, Article 32. DOI:https://doi.org/10.1145/3369397. Preprint:https://arxiv.org/pdf/1912.11936.pdf.

This dataset is released under the Creative Commons Zero (CC0) license. Please feel free to use this dataset for your own research. If you found this dataset and the code useful, we would greatly appreciate it if you could cite our paper above.

Below are descriptions about what each column means in the file that contains smell reports (the "smell_raw.csv"):
- EpochTime: the Epoch timestamp when the smell is experienced
- smell_value: the self-reported rating of the smell (described on the [Smell Pittsburgh website](https://smellpgh.org/how_it_works)) 
- smell_description: the self-reported description of the smell (e.g., woodsmoke)
- feelings_symptoms: the self-reported symptoms that may caused by the source of the smell (e.g., eye irritation)
- zipcode: the zipcode of the location where the smell is experienced

Information about the metadata (e.g., latitude, longitude, feed ID, channel name) of the sensor monitoring stations used in this dataset (all files in the "esdr_raw" folder) can be found on the [ESDR data visualization page](https://environmentaldata.org/#time=1642345888.849,1642950688.849&cursor=1642730480.675&plotHeight=5.000&plotAreaHeight=40.000&showFilters=true&showSettings=true&showResults=true&center=40.445982705178,-79.96401491796037&zoom=12). ESDR means the [Environmental Sensor Data Repository](https://esdr.cmucreatelab.org/), a service for hosting environmental data. The feed ID and the channel name in the [code for gettting the sensor data](/py/prediction/getData.py) corresponds to the metadata on the visualization page.
