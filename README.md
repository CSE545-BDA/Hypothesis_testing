Preliminary analysis contains code for checking how much is the problems worth running after.

Hypothesis testing folder  
> Concepts used
>> Hypothesis Testing (specifically, ttest)  

> Framework used
>> Apache Spark  

> System specification used for running the code -   
>> Google Cloud Platform (Dataproc cluster with 1 Master and 2 Workers)  
>> Debain 9, Hadoop 2.9, Spark 2.4  

1. Data download part -  
    https://console.cloud.google.com/marketplace/details/united-states-census-bureau/acs?filter=solution-type%3Adataset&filter=category%3Abig-data&q=income&id=1282ab4c-78a4-4da5-8af8-cd693fe390ab  
- Use the above link to find the BigQuery datasets.  
- Export all blockgroup tables (name format: blockgroup_YYYY_5yr) into HDFS in JSON format in the specific 'hdfs:/data/' path

2. Run spark-submit main.py
