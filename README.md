# Operationalizing External Supervised Models with Eland and Elastic Search

elastic.co|https://elastic.co

eland repo - https://github.com/elastic/eland

This demo shows you how to create a Random Forest Classifier machine learning model outside of elasticsearch with python, load it into elasticsearch, then operationalize it with ingest pipelines.

The model is created and trained using the sklearn ml python library in a Jupyter notebook. 

The model is loaded into elasticsearch using eland .

The model is operationalized by creating an ingest pipeline that uses the inference and enrich processors attached to an index template. 
