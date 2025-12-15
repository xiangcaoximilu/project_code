# import gdown
# url = "https://drive.google.com/uc?id=10_sVL0UmEog7mczLedK5s1pnlDOz3Ukf"
# output = "./white_house_2021_2022.zip"
# gdown.download(url,output)
#
# import zipfile
# with zipfile.ZipFile(output, "r")as zip_file:
#     zip_file.extractall(output[:-4])
import time
# clean the data
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
from pymilvus import MilvusClient, DataType

df = pd.read_csv("./white_house_2021_2022/The white house speeches.csv")
# 先丢弃nan值
df = df.dropna()
print(f"丢弃nan后的值:{df}")

cleaned_df = df.loc[(df["Speech"].str.len() > 50)]
print(f"去除掉Speech字段字符长度小于等于50的行:{cleaned_df}")

# Speech字段有许多空白行，\r\n在http协议中常用来表示该段落是一个空白行，将其内容
cleaned_df["Speech"] = cleaned_df["Speech"].str.replace("\r\n", "")
print(cleaned_df.iloc[0]["Speech"])

# 将Data_time字段格式转换成 datetime object
cleaned_df["Date_time"] = pd.to_datetime(cleaned_df["Date_time"], format="%B %d, %Y")

cleaned_df["unix_time"] = cleaned_df["Date_time"].apply(lambda x: int(x.timestamp()))
print(f"转换Date_time字段格式后的cleaned_df:{cleaned_df}")

# 创建向量数据库
COLLECTION_NAME = "text_search_origin_code"
DIMENTION = 384
BATCH_SIZE = 128
TOPK = 5

client = MilvusClient(uri="http://localhost:19530")
client.drop_collection(COLLECTION_NAME)


def get_collection(collection_name):
    if client.has_collection(collection_name):
        print(f"milvus collection already exists:\n{client.list_collections()}")
        client.load_collection(collection_name)

    else:
        print("创建新数据库")
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="date", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="location", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DIMENTION)

        schema.verify()

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="L2",
                               params={"nlist": 128})
        client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)
        time.sleep(5)
        res = client.get_load_state(COLLECTION_NAME)
        print(res)


get_collection(COLLECTION_NAME)

# 检查数据库状态
print("检查数据库状态：")
print(client.get_collection_stats(COLLECTION_NAME))

# data to milvus
transformer = SentenceTransformer("all-MiniLM-L6-v2")


def embed_insert(data: list):
    embeddings = transformer.encode(data[3])

    ins = [data[0],
           data[1],
           data[2],
           [x for x in embeddings]
           ]
    res = []
    for i in range(len(data[0])):
        res.append({"title": ins[0][i], "date": str(ins[1][i]), "location": ins[2][i], "embedding": ins[3][i]})

    count = client.insert(COLLECTION_NAME, data=res)
    print(count)

data_batch = [[], [], [], []]
for _, row in cleaned_df.iterrows():
    # data_batch[0].append({"title": row["Title"]})
    # data_batch[1].append({"date": row["Date_time"]})
    # data_batch[2].append({"location": row["Location"]})
    # data_batch[3].append(row["Speech"])

    data_batch[0].append(row["Title"])
    data_batch[1].append(row["Date_time"])
    data_batch[2].append(row["Location"])
    data_batch[3].append(row["Speech"])

    if len(data_batch[0]) % BATCH_SIZE == 0:
        try:
            embed_insert(data_batch)
        except Exception as e:
            print("error")
            exit()
        data_batch = [[], [], [], []]
# 把剩余部分导入数据库
if len(data_batch[0]) != 0:
    embed_insert(data_batch)

time.sleep(5)
# 检查数据库状态
print("检查数据库状态：")
print(client.get_collection_stats(COLLECTION_NAME))

search_terms = ["The President speaks about the impact of renewable energy at the National Renewable Energy Lab.",
                "The Vice President and the Prime Minister of Canada both speak."]


def embed_search(data):
    search_vector = transformer.encode(search_terms)
    return [x for x in search_vector]


search_data = embed_search(search_terms)

start_time = time.time()
res = client.search(collection_name=COLLECTION_NAME,
                    data=search_data,
                    anns_field="embedding",
                    limit=5,
                    search_params={"metric_type": "L2", "params": {"nprobe": 10}},
                    output_fields=['title'],
                    )
end_time = time.time()

for hits_i, hits in enumerate(res):
    print("Title:", search_terms[hits_i])
    print("Search Time:", end_time - start_time)
    print("Result:")
    for hit in hits:
        print(hit["entity"].get("title"), "-------", hit["distance"])
    print()

result = json.dumps(res, indent=4)
print(result)
