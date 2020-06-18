#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 06.2020 - Jeff Vestal @ Elastic
# Created with v7.8 of elasticsearch
# Model Training borrowed from - https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/

# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load eland
import eland as ed

# Load numpy
import numpy as np

# Set random seed
np.random.seed(0)

# Don't do this
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Create an object called iris with the iris data
iris = load_iris()


# In[ ]:


# Pandas
# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# View the top 5 rows
df.head()


# In[ ]:


# Add a new column with the species names, this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# View the top 5 rows
df.head()


# In[ ]:


# Create a new column that for each row, generates a random number between 0 and 1, and
# if that value is less than or equal to .75, then sets the value of that cell as True
# and false otherwise. This is a quick and dirty way of randomly assigning some rows to
# be used as the training data and some as the test data.
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# View the top 5 rows
df.head()


# In[ ]:


# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]

# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))


# In[ ]:


# Create a list of the feature column's names
features = df.columns[:4]

# View features
features


# In[ ]:


# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y, labels = pd.factorize(train['species'])

# View target
y


# In[ ]:


# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train[features], y)


# In[ ]:


# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
clf.predict(test[features])


# In[ ]:


# View the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]

# [probability belongs to first class, probability belongs to second class, predicted class]


# In[ ]:


# Create actual english names for the plants for each predicted plant class
preds = iris.target_names[clf.predict(test[features])]


# In[ ]:


# View the PREDICTED species for the first five observations
preds[0:5]


# In[ ]:


# View the ACTUAL species for the first five observations
test['species'].head()


# In[ ]:


# Create confusion matrix
pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])


# In[ ]:


# View a list of the features and their importance scores
list(zip(train[features], clf.feature_importances_))


# In[ ]:


# eland and Elastic time
from eland.ml import ImportedMLModel

from elasticsearch import Elasticsearch
from elasticsearch.client.ml import MlClient
from elasticsearch.client import IngestClient
from elasticsearch.client import IndicesClient
from elasticsearch.client.enrich import EnrichClient
from elasticsearch.helpers import bulk

from datetime import datetime
from pprint import pprint


# In[ ]:


# create es connection
# need to put in user : pass creds
# Es URL
es = Elasticsearch('https://<ES_URL>:9243', http_auth=('<USERNAME>', '<PASSWORD>'))


# In[ ]:


# Serialise the trained RandomForestClassifier model to Elasticsearch

# pick short feature names for the docs
feature_names = ['sl', 'sw', 'pl', 'pw']

# name model
model_id = "jeffs-rfc-flower-type"

# load model into elasticsearch
es_model = ImportedMLModel(es, model_id, clf, feature_names, overwrite=True)


# In[ ]:


# verify model exists in es
MlClient.get_trained_models(es)


# In[ ]:


## Test out the pipeline and new model


# In[ ]:


# configure ingest pipeline with inference processor using our model
body = {
  "pipeline": {
    "processors": [
      {
        "inference": {
          "model_id": model_id,
          "inference_config": {
            "classification": {}
          },
          "field_map": {}
        }
      }
    ]
  },
  "docs": [
    {
      "_source": {
        "sl" : 4.2,
        "sw": 3.9,
        "pl": 1.9,
        "pw": 0.4
      }
    }
  ]
}

# simulate ingest pipeline
IngestClient.simulate(es, body)


# In[ ]:


# Lets include an english name to convert the predicted value back to english flower name

# set up an enrich index

# create docs mapping serialized values -> english name
mapping_index_name = model_id + '_mapping'
mapping_docs =[]
now = datetime.now()

for pos, name in enumerate(labels.categories):
    mapping_docs.append({"_index": mapping_index_name,
                         'mapped_value': pos,
                         'flower_name': name,
                         'updated_ts': now})


# index mapping into es
res = bulk(es, mapping_docs)
print(res)


# In[ ]:


# Verify mapping docs are in the index
res = es.search(index=mapping_index_name, body={"query": {"match_all": {}}})
for doc in res['hits']['hits']:
    print ('Predicted value: %s -> Flower Name: %s'% (doc['_source']['mapped_value'], doc['_source']['flower_name']))


# In[ ]:


# create enrich policy
policy_name = model_id + '_enrich'

#delete existing policy
for p in EnrichClient.get_policy(es, name=policy_name)['policies']:
    if policy_name == p['config']['match']['name']:
        print('deleting existing policy')
        EnrichClient.delete_policy(es, name=policy_name)


# In[ ]:


# put enrich policy
policy = { "match" :{
        "indices": mapping_index_name,
        "match_field": 'mapped_value',
        "enrich_fields": ["flower_name"]
            }
        }

EnrichClient.put_policy(es, name=policy_name, body=policy)


# In[ ]:


# verify enrich policy
EnrichClient.get_policy(es, name=policy_name)


# In[ ]:


#execute enrich policy
EnrichClient.execute_policy(es, name=policy_name)


# In[ ]:


# Test ingest pipeline / inference pipeline with the additional enrich processor
processors = [
      {
        "inference": {
          "model_id": model_id,
          "inference_config": {
            "classification": {}
          },
          "field_map": {}
        }
      },
        {
         "enrich": {
             'policy_name': policy_name,
             'field' : 'ml.inference.predicted_value',
             'target_field' : 'ml.inference.predicted_name',
                    "tag": "enriched"

         }
        }
    ]

body = {
  "pipeline": {
    "processors": processors
  },
  "docs": [
    {
      "_source": {
        "sl" : 4.2,
        "sw": 3.9,
        "pl": 1.9,
        "pw": 0.4
      }
    }
  ]
}

# simulate ingest pipeline
IngestClient.simulate(es, body)


# In[ ]:


# store the pipeline for use in prod
pipeline_name = model_id + '_ingest_pipeline'
body = {
    'description': 'predict flower type',
    'processors': processors
}

IngestClient.put_pipeline(es, id=pipeline_name, body=body)


# In[ ]:


# verify pipeline
IngestClient.get_pipeline(es, pipeline_name)


# In[ ]:


# create index template with our new pipeline as the default pipeline

settings = {
  "index_patterns": ["flower_measurements-*"],
  "settings": {
    "default_pipeline": "jeffs-rfc-flower-type_ingest_pipeline"
  }
}

template_name = 'flowers_measurement'
IndicesClient.put_template(es, name=template_name, body=settings)


# In[ ]:


#verify  template
IndicesClient.get_template(es, name=template_name)


# In[ ]:


# put a new document with different (definitely real) measurements

new_flower = {
        "sl" : 4.2,
        "sw": 3.9,
        "pl": 11.9,
        "pw": 10.4,
        'updated_ts': now
      }

final_leg_index = 'flower_measurements-magic'
new_doc = es.index(index=final_leg_index, body=new_flower)


# In[ ]:


# Verify the doc was created
new_doc['result'], new_doc['_id']


# In[ ]:


# Find out what the flower was predicted to be with the _source and a nice human readable output!
res = es.get(index=final_leg_index,id=new_doc['_id'])
pprint(res['_source'])
print('\nThis flower is predicted to be a %s !' % res['_source']['ml']['inference']['predicted_name']['flower_name'])


# In[ ]:





# In[ ]:


# cleanup


IngestClient.delete_pipeline(es, id=pipeline_name)
IndicesClient.delete_template(es, name=template_name)
EnrichClient.delete_policy(es, name=policy_name)
es.delete_by_query(index=final_leg_index, body={"query": {"match_all": {}}})
es.delete_by_query(index=mapping_index_name, body={"query": {"match_all": {}}})


# In[ ]:




