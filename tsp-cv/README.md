# Travelling Salesman Problem

The Traveling Salesperson Problem (TSP) is a class problem of computer science that seeks to find the shortest route between a group of cities. It is an NP-hard problem in combinatorial optimization, important in theoretical computer science and operations research. Refer to original [kaggle problem](https://www.kaggle.com/competitions/tsp-cv)  

![Route](https://i.imgur.com/YqIizWP.jpg)

## Submission
|Layers||
|---|---|
|`Conv2D`| 32-64|
|`GlobalAverage2D`||
|`Dense`| 256-128-1|

|#|Score|
|---|---|
|5/41|492|

## Upsolve

3 channel * 256 bins of histograms = 768 features
Shuffle = False
|Layers||
|---|---|
|`Flatten`||
|`Dense`| 1504-752-1|

Improved Score: **138.23716**
