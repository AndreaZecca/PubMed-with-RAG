from pymilvus import connections
conn = connections.connect(
  alias="default",
  user='username',
  password='password',
  host="127.0.0.1",
  port='19530'
)
from pymilvus import utility
print(utility.has_collection("medmcqa"))

from pymilvus import Collection
collection = Collection("medmcqa")

print(collection.num_entities)