from pymilvus import connections,utility, Collection

_ = connections.connect(
  alias="default",
  user='username',
  password='password',
  host="127.0.0.1",
  port='19530'
)

if utility.has_collection("medqa"):
  print(f'medqa: {utility.has_collection("medqa")}')
  collection = Collection("medqa")
  print(f'collection: {collection.name} has {collection.num_entities} entities')
  print()
else:
  print(f'medqa collection does not exist')
  print()

if utility.has_collection("medmcqa_mmlu"):
  print(f'medmcqa_mmlu: {utility.has_collection("medmcqa_mmlu")}')
  collection = Collection("medmcqa_mmlu")
  print(f'collection: {collection.name} has {collection.num_entities} entities')
  print()
else:
  print(f'medmcqa collection does not exist')
  print()