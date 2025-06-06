import time
import firebase_admin
from firebase_admin import credentials, db, messaging
from helper.path import get_abs_path

cred_path = get_abs_path("baby-sleep-3085d-firebase-adminsdk-fbsvc-40a2ca0158.json", __file__)
cred = credentials.Certificate(cred_path)

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://baby-monitor-58b3f-default-rtdb.firebaseio.com/'
})

# Biến toàn cục theo dõi thời điểm gửi gần nhất
_last_notification_time = 0

def send_notification_to_token(token, title, body):
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body
        ),
        token=token
    )

    response = messaging.send(message)
    print("Đã gửi thông báo:", response)

def send_notification_to_user(body):
    global _last_notification_time
    now = time.time()

    if now - _last_notification_time >= 10:
        send_notification_to_token(
            "fHT9_0hLTmOgi8Gpw4PBaG:APA91bHVn-y79PbcVCL7_tJOh9hhB7_Sglrf3JUeIviBTla58G1M1n61Dl6SRJ-KysJF-MRhUTCNPyIfmA-wbPNGuBd9fahU_ET7IFZkgsHSD2lmIZH-iBA",
            "Baby sleep",
            body
        )
        _last_notification_time = now
    else:
        print(f"Bỏ qua gửi: chưa đủ 10 giây. Còn lại: {round(10 - (now - _last_notification_time), 2)}s")


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

