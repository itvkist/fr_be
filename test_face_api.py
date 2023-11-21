import requests
# import cv2
# import base64
# import os

# def convert2Base64(image):
#     retval, buffer = cv2.imencode('.jpg', image)
#     jpg_as_text = base64.b64encode(buffer)
#     img_str = 'data:image/jpg;base64,' + str(jpg_as_text)[2:-1]
#     return img_str

# path = "../000"

# # Tạo cây với 3 ảnh đầu
# for i, dir_ in enumerate(os.listdir(path)):
#     for j, f in enumerate(os.listdir(path + '/' + dir_)[:3]):
#         print(path + '/' + dir_ + '/' + f)
#         image = cv2.imread(path + '/' + dir_ + '/' + f)
#         print(i, j)
#         r = requests.post('http://localhost:5001/facereg', json={'ann_id': 'abc', 'id': str((i + 1)*10 + j), 'img': convert2Base64(image), 'local_register': True})
#         print(r.json())

# # Query các ảnh còn lại
# for dir_ in os.listdir(path):
#     for f in os.listdir(path + '/' + dir_)[3:]:
#         image = cv2.imread(path + '/' + dir_ + '/' + f)
#         r = requests.post('http://localhost:5001/facerec', json={'ann_id': 'abc', 'img': convert2Base64(image), 'local_register': True})
#         print(r.json()['result']['ids'], r.json()['result']['accessories'])

r = requests.post('http://localhost:5001/facerec', json={'ann_id': 'abc', 'img': "http://localhost:8000/001/03.jpg", 'local_register': False})
print(r.json())
