# import firebase_admin
# from firebase_admin import credentials,  db
# from helper.path import get_abs_path

# cred_path = get_abs_path("baby-monitor-58b3f-firebase-adminsdk-fbsvc-0d422dc245.json", __file__)
# cred = credentials.Certificate(cred_path)
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://baby-monitor-58b3f-default-rtdb.firebaseio.com/'
# })

# ref = db.reference('/') 

# def send_firebase(data):
#     ref.set(data)


# def listener(event):
#     print(event.event_type)  # can be 'put' or 'patch'
#     print(event.path)  # relative to the reference, it seems
#     print(event.data)  # new data at /reference/event.path. None if deleted

# db.reference('/users/user1').listen(listener)

# ref = db.reference('/users/user1')
# user_data = ref.get()
# print(user_data)