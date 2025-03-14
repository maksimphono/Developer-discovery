# developer discovery mongodb query utilities

from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient(f'mongodb://readonlyUser:cictest123456@114.212.84.247:27017/')
# Access the database
db = client.developer_discovery
# Access the collection
collection_events = db.events
collection_events_id = db.events_id
collection_user_info = db.user_info


def find_user_event_ids(user_id):
    if "github:" not in user_id:
        user_id = f"github:{user_id}"
    return collection_events_id.find({"user_id": user_id})


def find_proj_event_ids(proj_id):
    if "github:" not in proj_id:
        proj_id = f"github:{proj_id}"
    return collection_events_id.find({"proj_id": proj_id})


def find_event_details_by_ids(id_list):
    return collection_events.find({"id": {"$in": id_list}})


def check_user_is_bot(user_login):
    '''
    给定用户的login，注意是不带github:前缀的，返回其身份是"Bot"或"User"。如果查无此人则返回None。
    :param user_login:
    :return:
    '''
    found = False
    cursor = collection_user_info.find({"login": user_login}, {"type": 1})
    for rec in cursor:
        found = True
        if rec["type"] == "Bot":
            return "Bot"
    if found:
        return "User"
    else: # 查无此人
        return None


if __name__ == "__main__":

    # 根据用户名id查询一个人的活动记录
    id_list = []
    event_idx = find_user_event_ids("github:niyashiyas")
    for rec in event_idx:
        # 此处rec中包含 事件id（检索events collection时的主键）, type, user_id，proj_id, created_at
        # 针对应用需求，此处可对type、proj_id、created_at设置过滤条件
        id_list.append(rec["id"])
    print(id_list)

    # 使用事件id列表查询events集合，获取事件详细信息
    event_details = find_event_details_by_ids(id_list)
    cnt = 0
    for rec in event_details:
        print(rec)
        cnt += 1
    print(f"{len(id_list)} : {cnt}")

    # 根据项目id查询该仓库中的活动记录
    id_list = []
    event_idx = find_proj_event_ids("github:nodejs/node")
    for rec in event_idx:
        # 此处rec中包含 事件id（检索events collection时的主键）, type, user_id，proj_id, created_at
        # 针对应用需求，此处可对type、user_id、created_at设置过滤条件
        id_list.append(rec["id"])
    print(id_list)

    # 使用事件id列表查询events集合，获取事件详细信息
    event_details = find_event_details_by_ids(id_list)
    cnt = 0
    for rec in event_details:
        print(rec)
        cnt += 1
    print(f"{len(id_list)} : {cnt}")
