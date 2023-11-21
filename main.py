import os
import numpy as np
import cv2
from PIL import Image
from numpy import dot, sqrt

from aiohttp import web
from aiohttp import client
import aiohttp
import asyncio

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

from app.arcface.backbone import Backbone
from app.vision.ssd.config.fd_config import define_img_size
from app.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

import hnswlib

import base64
import requests

app = web.Application()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Face Detection Model
class_names = [name.strip() for name in open('./app/vision/detect_RFB_640/voc-model-labels.txt').readlines()]
candidate_size = 1000
threshold = 0.7
input_img_size = 640
define_img_size(input_img_size)
model_path = "./app/vision/detect_RFB_640/version-RFB-640.pth"
net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
face_detection = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=device)
net.load(model_path)

#Face Recognition Model
input_size=[112, 112]
transform = transforms.Compose(
        [
            transforms.Resize(
                [int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)],
            ),  # smaller side resized
            transforms.CenterCrop([input_size[0], input_size[1]]),
            # transforms.Resize([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
)
face_recognition = Backbone(input_size)
face_recognition.load_state_dict(torch.load('./app/arcface/ms1mv3_arcface_r50_fp16/backbone_ir50_ms1m_epoch120.pth', map_location=torch.device("cpu")))
face_recognition.to(device)
face_recognition.eval()

#Face Mask/Glass Classification Model
data_transform = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.ToTensor()
])
accessories_classify = torch.load('./app/accessories_classification/shuffle0_0_epoch_47.pth', map_location=device)
accessories_classify.eval()

def loadBase64Img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def load_image(img):
	exact_image = False; base64_img = False; url_img = False

	if type(img).__module__ == np.__name__:
		exact_image = True
	elif len(img) > 11 and img[0:11] == "data:image/":
		base64_img = True
	elif len(img) > 11 and img.startswith("http"):
		url_img = True

	if base64_img == True:
		img = loadBase64Img(img)
	elif url_img:
		img = np.array(Image.open(requests.get(img, stream=True).raw))
	elif exact_image != True: #image path passed as input
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")
		img = cv2.imread(img)
    
	return img

def check_accessories(image):
    image_tensor = data_transform(image)
    image_tensor = image_tensor.unsqueeze_(0)
    input_ = Variable(image_tensor)
    input_ = input_.to(device)
    output = accessories_classify(input_)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)
    index = probabilities.data.cpu().numpy().argmax()
    return index

def cosine_similarity(x, y):
    return dot(x, y) / (sqrt(dot(x, x)) * sqrt(dot(y, y)))

def get_embeddings(img_input, ann_id, local_register = False, register=False):
    # img = []
    img = load_image(img_input)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if not local_register:
        boxes, labels, probs = face_detection.predict(image, candidate_size / 2, threshold)
        boxes = boxes.detach().cpu().numpy()
    else:
        boxes = np.array([[0, 0, image.shape[1], image.shape[0]]])

    feats_np = []
    feats = []
    images = []
    bboxes = []
    ids = []
    accessories = []
    distances_ = []
    if not register:
        p = hnswlib.Index(space = 'cosine', dim = 512)
        p.load_index("indexes/index_" + ann_id + '.bin')
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        xmin, ymin, xmax, ymax = box
        xmin -= (xmax-xmin)/18
        xmax += (xmax-xmin)/18
        ymin -= (ymax-ymin)/18
        ymax += (ymax-ymin)/18
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = image.shape[1] if xmax >= image.shape[1] else xmax
        ymax = image.shape[0] if ymax >= image.shape[0] else ymax
        boxes[i,:] = [xmin, ymin, xmax, ymax]
        infer_img = image[int(ymin): int(ymax), int(xmin): int(xmax), :]
        person_id = ['-1', '-1', '-1']
        distance = ['0', '0', '0']
        accessory_id = 2
        if infer_img is not None and infer_img.shape[0] != 0 and infer_img.shape[1] != 0:
            with torch.no_grad():
                feat = F.normalize(face_recognition(transform(Image.fromarray(infer_img)).unsqueeze(0).to(device))).cpu()
                accessory_id = check_accessories(Image.fromarray(infer_img))
                if not register:
                    try:
                        neighbors, distances = p.knn_query(feat.detach().cpu().numpy(), k=3)
                        if (distances[0][0] <= 0.55 and accessory_id != 1) or (distances[0][0] <= 0.75 and accessory_id == 1):
                            person_id = [str(n) for n in neighbors[0]]
                            distance = [str(d) for d in distances[0]]
                    except Exception as e:
                        print(e)
                        person_id = ['-1', '-1', '-1']
                        distance = ['0', '0', '0']
            feats.append(np.array2string(feat.detach().cpu().numpy()))
            feats_np.append(feat.detach().cpu().numpy())
            images.append(infer_img.copy())
            bboxes.append("{} {} {} {}".format(xmin, ymin, xmax, ymax))
            accessories.append(str(accessory_id))
            ids.append(person_id)
            distances_.append(distance)

    
    return feats_np, feats, images, bboxes, accessories, ids, distances_

async def facerec(request):
    """
    ---
    description: This end-point allow to recognize face identity.
    tags:
    - Face Recognition
    produces:
    - text/json
    responses:
        "200":
            description: successful operation
        "400":
            description: Vui lòng truyền secret key
        "400":
            description: Vui lòng truyền ảnh dưới dạng Base64
        "403":
            description: Secret key không hợp lệ
    """
    req = await request.json()

    ann_id = ""
    if "ann_id" in list(req.keys()):
        ann_id = req["ann_id"]

    img_input = ""
    if "img" in list(req.keys()):
        img_input = req["img"]

    validate_img = False
    if len(img_input) > 11 and len(ann_id) > 0 and (img_input[0:11] == "data:image/" or img_input[0:4] == "http"):
        validate_img = True

    if validate_img != True:
        return  web.json_response({"result": {'message': 'Vui lòng truyền ảnh dưới dạng Base64'}}, status=400)

    if 'local_register' in list(req.keys()):
        _, feats, images, bboxes, accessories, ids, distances = get_embeddings(img_input, ann_id, req['local_register'])
    else:
        _, feats, images, bboxes, accessories, ids, distances = get_embeddings(img_input, ann_id)

    return  web.json_response({'result': {"bboxes": bboxes, "feats": feats, "ids": ids, "distances": distances, "accessories": accessories}}, status=200)

async def facereg(request):
    """
    ---
    description: This end-point allow to recognize face identity.
    tags:
    - Face Recognition
    produces:
    - text/json
    responses:
        "200":
            description: successful operation
        "400":
            description: Vui lòng truyền secret key
        "400":
            description: Vui lòng truyền ảnh dưới dạng Base64
        "403":
            description: Secret key không hợp lệ
    """
    req = await request.json()

    ann_id = ""
    if "ann_id" in list(req.keys()):
        ann_id = req["ann_id"]

    img_input = ""
    if "img" in list(req.keys()):
        img_input = req["img"]

    id_ = ""
    if "id" in list(req.keys()):
        id_ = req["id"]

    validate_img = False
    if len(img_input) > 11 and len(id_) > 0 and len(ann_id) > 0 and (img_input[0:11] == "data:image/" or img_input[0:4] == "http"):
        validate_img = True

    if validate_img != True:
        return  web.json_response({"result": {'message': 'Vui lòng truyền ảnh dưới dạng Base64'}}, status=400)

    if 'local_register' in list(req.keys()):
        feats_np, feats, images, bboxes, accessories, ids, _ = get_embeddings(img_input, ann_id, req['local_register'], True)
    else:
        feats_np, feats, images, bboxes, accessories, ids, _ = get_embeddings(img_input, ann_id, True, True)

    p = hnswlib.Index(space = 'cosine', dim = 512)
    if not os.path.isfile("indexes/index_" + ann_id + '.bin'):
        p.init_index(max_elements = 1000, ef_construction = 200, M = 16)
        p.set_ef(10)
        p.set_num_threads(4)
        p.save_index("indexes/index_" + ann_id + '.bin')
    else:
        p.load_index("indexes/index_" + ann_id + '.bin', max_elements=1000)

    for feat in feats_np[:1]:
        try:
            p.unmark_deleted(int(id_))
        except:
            print("unmark no label")
        p.add_items(feat, np.array([int(id_)]))
        p.save_index("indexes/index_" + ann_id + '.bin')

    return  web.json_response({'result': {"bboxes": bboxes, "feats": feats, "accessories": accessories, "register_id": id_, "message": "success"}}, status=200)

async def facedel(request):
    """
    ---
    description: This end-point allow to recognize face identity.
    tags:
    - Face Recognition
    produces:
    - text/json
    responses:
        "200":
            description: successful operation
        "400":
            description: Vui lòng truyền secret key
        "400":
            description: Vui lòng truyền ảnh dưới dạng Base64
        "403":
            description: Secret key không hợp lệ
    """
    req = await request.json()

    ann_id = ""
    if "ann_id" in list(req.keys()):
        ann_id = req["ann_id"]

    ids = []
    if "ids" in list(req.keys()):
        ids = req["ids"]

    p = hnswlib.Index(space = 'cosine', dim = 512)
    p.load_index("indexes/index_" + ann_id + '.bin', max_elements=1000)

    id_ = None
    try:
        for id_ in ids:
            p.mark_deleted(int(id_))
        p.save_index("indexes/index_" + ann_id + '.bin')
    except Exception as e:
        print(e)
        return  web.json_response({'result': {"message": "failed", "id": str(id_)}}, status=500)

    return  web.json_response({'result': {"message": "success", "ids": ids}}, status=200)


app.router.add_route('POST',"/facerec", facerec)
app.router.add_route('POST',"/facereg", facereg)
app.router.add_route('POST',"/facedel", facedel)

if __name__ == "__main__":
    web.run_app(app, port=5001)