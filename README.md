# PV_detection
This is a code repository for one of my graduation projects.

1.project description
====
1.1 Project name & tasks: 
--
* Photovoltaic(PV-arrays) recognition in aerial images based on deep learning<br>
* build deep neural networks to complete detection and segmentation tasks,which means calculate the accurate bounding box and binary segmentation of PV-arrays in a set of aerial images.<br>

1.2 Enviroment Requirements: 
--
* Python>=3.6
* Pytorch=1.0
* Matlab platform

2.Dataset description
====
2.1 Data source:<br>
--
  The dataset which is needed in this project comes form FigShare platform(Figshare is an online data repository based on cloud computing technology that allows researchers to save and share their research, including data, datasets, images, videos, posters and code.),the link is attached below.<br>
* https://figshare.com/<br>

2.2 Data structure:<br>
--
  The complete dataset is composed of the (1)aerial images(.tif format) which collected from four cities in California.USA and (2)the label file in .csv format which contains detailed coordinate of vertices of polygons that surround the PV-arrays.<br>
  This project uses the data of stockton city (id:3385804),you can download the raw images and the label file from the links below separately.You can also view the attached instructions on official website.<br>
  
raw images:<br>
* https://figshare.com/articles/Stockton_Aerial_USGS_Imagery_from_the_Distributed_Solar_Photovoltaic_Array_Location_and_Extent_Data_Set/3385804<br>

label files:<br>
* https://figshare.com/articles/Stockton_Aerial_USGS_Imagery_from_the_Distributed_Solar_Photovoltaic_Array_Location_and_Extent_Data_Set/3385780<br>

3.Project Structure
====
3.1 make label data from original label files
--
3.2 build end-to-end network model 
--

