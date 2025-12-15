from pymilvus import utility, model, MilvusClient, DataType
# from datasets import load_dataset_builder, load_dataset, Dataset
# from transformers import AutoTokenizer, AutoModel
import pandas as pd
import time
import numpy as np
from towhee import ops, pipe, DataCollection

df = pd.read_csv("./archive/New_Medium_Data.csv", converters={"title_vector": lambda x: eval(x)})
df.head()

# parameters
DATASET = "squad"  # Huggingface dataset to use
MODEL = "bert-base-uncased"
TOKENIZATION_BATCH_SIZE = 1000  # batch size for tokenizing operation
INFERENCE_BATCH_SIZE = 64
INSERT_RATIO = 0.001  # how many titles to embed and insert
COLLECTION_NAME = "search_article_in_medium"
LIMIT = 30  # how many results to research for
DIMENTION = 768  # embedding size
HOST = "127.0.0.1"  # IP for milvus
PORT = 19530

client = MilvusClient(uri="http://127.0.0.1:19530")
print("connect!")

# client.drop_collection(collection_name=COLLECTION_NAME)


flag = 0
# Milvus
def create_milvus_collection(collenciton_name, dim, data):
    # connect to date base

    # remove collection if it already exists
    if collenciton_name in client.list_collections():
        global flag
        flag = 1
        print("collection already exists, connect to it")
        # client.drop_collection(collection_name=collenciton_name)
        client.load_collection(collection_name=COLLECTION_NAME)
        time.sleep(5)  # 等待数据库创建
    else:
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="title_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="link", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="reading_time", datatype=DataType.INT64)
        schema.add_field(field_name="publication", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="claps", datatype=DataType.INT64)
        schema.add_field(field_name="responses", datatype=DataType.INT64)

        # 校验格式是否正确
        schema.verify()
        # 构建索引
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="title_vector", index_type="IVF_FLAT", metric_type="L2",
                               params={"nlist": 2048})
        client.create_collection(collection_name=collenciton_name, schema=schema, index_params=index_params)

        time.sleep(5)  # 等待数据库创建
        res = client.get_load_state(collection_name=collenciton_name)
        print(res)

        # data to milvus
        insert_pipe = (
            pipe.input("df")
            .flat_map("df", "data", lambda df: df.values.tolist())
            .map("data", "res", ops.ann_insert.milvus_client(
                host=HOST,
                port=str(PORT),
                collection_name=COLLECTION_NAME)
                 )
            .output("res")
        )
        # 执行插入操作
        start_time = time.time()
        _ = insert_pipe(data)
        end_time = time.time()
        print(f"cost time:{end_time - start_time}")


        print(f"collection_list:\n{client.list_collections()}")


# 创建collection

create_milvus_collection(COLLECTION_NAME, DIMENTION)

print("collection create")

'''
#  测试测数据存数据是否有问题
embedding_fn = model.DefaultEmbeddingFunction()

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
vectors = embedding_fn(docs)
print("Dim:", embedding_fn.dim, vectors[0].shape)

# data = [
#     {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"} \
#     for i in range(len(vectors))]

# 测试数据库能否正常连接并插入数据
data2 = df.iloc[[2,4,6]]
print(data2)
print("Data has:{} entities,each with fields:{}".format(len(data2), ",".join(data2.columns)))
# print(f"Vector dim:{len(data[0]['vector'])}")
data3 = [row.to_dict()for _, row in data2.iterrows()]
print("new data2:\\n", data2)


res = client.insert(collection_name=COLLECTION_NAME, data=data3)
print(res)

'''
# 检查数据库状态
#client.load_collection(collection_name=COLLECTION_NAME)
print(client.get_load_state(collection_name=COLLECTION_NAME))
print(client.get_collection_stats(collection_name=COLLECTION_NAME))

# 开始执行查找操作
search_pipe = (pipe.input("query")
               .map("query", "vec", ops.text_embedding.dpr(model_name="facebook/dpr-ctx_encoder-single-nq-base"))
               .map("vec", "vec", lambda x: x / np.linalg.norm(x, axis=0))
               .flat_map("vec", ("id", "score"), ops.ann_search.milvus_client(host=HOST,
                                                                              port=PORT,
                                                                              collection_name=COLLECTION_NAME))
               .output("query", "id", "score")
               )
res = search_pipe("funny python demo")
DataCollection(res).show()
