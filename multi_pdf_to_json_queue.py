# -*- coding: utf-8 -*-
# @Time    : 01/08/2025
# @File    : multi_pdf_to_json_queue.py.py
# @Software: PyCharm

"""
multi_pdf_to_json_queue.py - Description of the file/module

new pdf 解析

"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import logging
import pdf2image
import re
import numpy as np
import pdfplumber as pb
import html2text
import copy
import os
os.environ['YOLO_VERBOSE'] = str(False)
from ultralytics import YOLO
import torch
import math
from paddleocr import PPStructure
import fitz
from PIL import Image
from paddlex import create_pipeline
logging.getLogger('paddlex').setLevel(logging.CRITICAL)
import json, time

import multiprocessing
import queue
from multiprocessing import Manager, Process, Lock, Queue
import glob
from collections import Counter
import threading
import asyncio
from filelock import FileLock
import shutil
from pathlib import Path
from config import LAYOUT_PATH

# LAYOUT_PATH = "/root/autodl-tmp/GLS/model/"
SAVE_PATH = '/root/Data/GLS/evidence_card/'

class PDFParsing():
    def __init__(self, layout_path, **kwargs):

        self.model_dir_path = layout_path
        self.layout_engine = YOLO(f'{self.model_dir_path}yolo12x_best.pt')

        self.table_en = PPStructure(layout=False, lang='en', show_log=False)
        self.table_cn = PPStructure(layout=False, lang='ch', show_log=False)

        self.cn_ocr_pipeline = create_pipeline(pipeline=f"{self.model_dir_path}cn_ocr5.1.yaml")
        self.en_ocr_pipeline = create_pipeline(pipeline=f"{self.model_dir_path}en_ocr5.1.yaml")

        self.h = html2text.HTML2Text()
        self.h.ignore_links = True

        self.zh_patter = re.compile(u'[\u4e00-\u9fa5]+')


    def sorted_boxes(self, res, w):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            res(list):results
        return:
            sorted results(list)
        """
        num_boxes = len(res)
        if num_boxes == 1:
            return res

        sorted_boxes = sorted(res, key=lambda x: (x['xyxy'][1], x['xyxy'][0]))
        _boxes = list(sorted_boxes)

        new_res = []
        res_left = []
        res_right = []
        i = 0

        while True:
            if i >= num_boxes:
                break
            if i == num_boxes - 1:
                if _boxes[i]['xyxy'][1] > _boxes[i - 1]['xyxy'][3] and _boxes[i]['xyxy'][0] < w / 2 and _boxes[i]['xyxy'][2] > w / 2:
                    new_res += res_left
                    new_res += res_right
                    new_res.append(_boxes[i])
                else:
                    if _boxes[i]['xyxy'][2] > w / 2:
                        res_right.append(_boxes[i])
                        new_res += res_left
                        new_res += res_right
                    elif _boxes[i]['xyxy'][0] < w / 2:
                        res_left.append(_boxes[i])
                        new_res += res_left
                        new_res += res_right
                res_left = []
                res_right = []
                break
            elif _boxes[i]['xyxy'][0] < w / 4 and _boxes[i]['xyxy'][2] < 3 * w / 4:
                res_left.append(_boxes[i])
                i += 1
            elif _boxes[i]['xyxy'][0] > w / 4 and _boxes[i]['xyxy'][2] > w / 2:
                res_right.append(_boxes[i])
                i += 1
            else:
                new_res += res_left
                new_res += res_right
                new_res.append(_boxes[i])
                res_left = []
                res_right = []
                i += 1
        if res_left:
            new_res += res_left
        if res_right:
            new_res += res_right
        return new_res


    def post_process_layout(self, location):
        delete_index_lis = []
        new_location = copy.deepcopy(location)
        for i, bbox in enumerate(location[::-1]):
            xyxy = bbox['xyxy']
            if (xyxy[3] - xyxy[1])/(xyxy[2] - xyxy[0]) > 20:
                del new_location[len(location) - 1 - i]

        for i, bbox in enumerate(new_location):
            for j, second_bbox in enumerate(new_location):
                if j <= i:
                    continue
                box_iou = self.box_iou(torch.tensor([bbox['xyxy']]), torch.tensor([second_bbox['xyxy']]))
                if self.bbox_ioa(np.array([bbox['xyxy']]), np.array([second_bbox['xyxy']])) > 0.8:
                    if box_iou > 0.7:
                        if bbox['conf'] >= second_bbox['conf']:
                            delete_index_lis.append(j)
                        else:
                            delete_index_lis.append(i)
                    else:
                        delete_index_lis.append(j)
                elif self.bbox_ioa(np.array([second_bbox['xyxy']]), np.array([bbox['xyxy']])) > 0.8:
                    if box_iou > 0.7:
                        if bbox['conf'] >= second_bbox['conf']:
                            delete_index_lis.append(j)
                        else:
                            delete_index_lis.append(i)
                    else:
                        delete_index_lis.append(i)
        delete_index_lis = list(set(delete_index_lis))
        delete_index_lis = sorted(delete_index_lis, reverse=True)
        for k in delete_index_lis:
            del new_location[k]

        return new_location


    def layout(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        else:
            arr = np.array(image)
            if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
                raise ValueError(f"传入 layout 的图像为空，shape={arr.shape}")
            # 若已是 BGR numpy array（来自 pdf_parsing 的 [:,:,::-1] 转换），直接使用；
            # 若是 PIL Image（RGB），则转换为 BGR
            if isinstance(image, np.ndarray):
                image = arr
            else:
                image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        results = self.layout_engine(image, conf=0.3, save_crop=False, verbose=False)
        labels = ['text', 'title', 'figure', 'figure_caption', 'table', 'table_caption', 'header', 'footer',
                  'reference', 'equation']
        res = [{'xyxy': [math.floor(xyxy) if id < 2 else math.ceil(xyxy) for id, xyxy in enumerate(i.data.cpu().numpy().tolist())], 'type': labels[int(j.data.cpu().numpy().tolist())],
                'conf': k.data.cpu().numpy().tolist()} for i, j, k in
               zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf)]

        res = self.post_process_layout(res)
        h, w, _ = image.shape

        res = self.sorted_boxes(res, w)

        return res, image

    def ocr(self, images, lang='ch', batch_size=16):
        """
        识别文字，增加 batch_size 控制，防止 OOM
        """
        text_lis = []
        # 将传入的所有小图，按照 batch_size (比如每次 16 张) 切分处理
        for i in range(0, len(images), batch_size):
            batch_images = images[i: i + batch_size]
            batch_images = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in batch_images]

            if lang == 'en':
                result = self.en_ocr_pipeline.predict(
                    input=batch_images,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                )
            else:
                result = self.cn_ocr_pipeline.predict(
                    input=batch_images,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                )

            for res in result:
                if lang == 'en':
                    ocr_text = ' '.join(res['rec_texts']).replace('-  ', '').replace('- ', '').strip()
                else:
                    ocr_text = ''.join(res['rec_texts']).strip()
                text_lis.append(ocr_text)

        return text_lis

    # def ocr(self, images, lang='ch'):
    #     """
    #     识别文字
    #     :param lang: 目前支持的多语言语种. 例如`ch`, `en`,
    #     :return:
    #     """
    #
    #     images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in images]
    #
    #     if lang == 'en':
    #         result = self.en_ocr_pipeline.predict(
    #             input=images,
    #             use_doc_orientation_classify=False,
    #             use_doc_unwarping=False,
    #             use_textline_orientation=True,
    #         )
    #     else:
    #         result = self.cn_ocr_pipeline.predict(
    #             input=images,
    #             use_doc_orientation_classify=False,
    #             use_doc_unwarping=False,
    #             use_textline_orientation=True,
    #         )
    #     text_lis = []
    #     for res in result:
    #
    #         if lang == 'en':
    #             ocr_text = ' '.join(res['rec_texts']).replace('-  ', '').replace('- ', '').strip()
    #         else:
    #             ocr_text = ''.join(res['rec_texts']).strip()
    #
    #         text_lis.append(ocr_text)
    #
    #     return text_lis


    def table_recognize(self, image, lang='ch'):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if lang == 'ch':
            result = self.table_cn(image)
        elif lang == 'en':
            result = self.table_en(image)
        else:
            print("please select langulage 'en' or 'ch'")

        return result[0]['res']['html']


    def pdf_to_image(self, pdf_path, dpi=200):

        images = []
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):

            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)

            # if width or height > 3000 pixels, don't enlarge the image
            if pix.width > 3000 or pix.height > 3000:
                pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)

            # images.append(image)
            images.append(np.array(image)[:, :, ::-1])

            # input()
        return images

    def html_to_markdown(self, html):
        return self.h.handle(html)


    def pdf_parsing(self, pdf_path, lang=None, html2markdown=True, res_bbox=True):

        if not lang:
            lang = self.recongnize_pdf_language(pdf_path)

        # input()
        data_lis = []

        doc = fitz.open(pdf_path)
        dpi = 200

        # pdf_images = self.pdf_to_image(pdf_path)

        for page_id in range(len(doc)):
            print(f"Processing page: {page_id}", flush=True)

            # --- 以下为新增的逐页渲染图片逻辑，用完即毁 ---
            page = doc[page_id]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            image_pil = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)

            if pix.width > 3000 or pix.height > 3000:
                pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                image_pil = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)

            np_image = np.array(image_pil)[:, :, ::-1]

            # 及时释放 pymupdf 对象，防止内存泄露
            page = None
            pix = None
            # --- 逐页渲染结束 ---

            # 跳过空页（LibreOffice 转换的 PDF 偶尔会产生 0 尺寸页面）
            if np_image.size == 0 or np_image.shape[0] == 0 or np_image.shape[1] == 0:
                print(f"警告: 第 {page_id} 页渲染结果为空，已跳过。", flush=True)
                continue

            # 使用当前生成的单页图片进行排版分析
            res_structure, image = self.layout(np_image)


            res_structure_num = len(res_structure)
            structure_lis = []
            temp_lis = []
            for i in range(res_structure_num):
                line = res_structure[i]
                bbox = line['xyxy']
                # print(line['conf'])
                # if page_id == 9:
                #     print(line)

                if line['type'] in ['figure', 'header', 'footer']:
                    continue
                elif line['type'] == 'table':
                    ### text rec
                    if len(structure_lis) > 0:
                        res_texts = self.ocr(structure_lis, lang)
                        data_lis.extend(self.ocr_post_process(res_texts, temp_lis))
                        structure_lis = []
                        temp_lis = []
                    ### table rec
                    # if bbox[3]-bbox[1] > 1.5 * (bbox[2] - bbox[0]):
                    #     image_table = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    #     result_image = image_table[max(bbox[0]-5, 0):bbox[2]+3, -(bbox[3]+5):-max(bbox[1]-3, 0), ::-1]
                    # else:
                    #     result_image = image[max(bbox[1]-1, 0):bbox[3]+1, max(bbox[0]-5, 0):bbox[2]+3, ::-1]

                    if bbox[3] - bbox[1] > 1.5 * (bbox[2] - bbox[0]):
                        image_table = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                        # 获取旋转前原图的高度 (即现在的宽度)
                        h_orig = image.shape[0]
                        # 使用绝对正数索引替代负数索引
                        col_start = max(0, h_orig - int(bbox[3]) - 5)
                        col_end = h_orig - max(int(bbox[1]) - 3, 0)
                        result_image = image_table[max(bbox[0] - 5, 0):bbox[2] + 3, col_start:col_end, ::-1]
                    else:
                        result_image = image[max(bbox[1] - 1, 0):bbox[3] + 1, max(bbox[0] - 5, 0):bbox[2] + 3, ::-1]

                    # 🌟 新增容错拦截：如果因极端越界情况切图依然为空，跳过该区域，防止崩溃整批任务
                    if result_image is None or result_image.size == 0:
                        print(f"警告: 截取的图像区域为空，已跳过。bbox坐标: {bbox}", flush=True)
                        continue


                    html_text = self.table_recognize(result_image, lang)
                    if html2markdown:
                        markdown = self.html_to_markdown(html_text)
                        # print(markdown)
                        if res_bbox:
                            data_lis.append({'text': markdown, 'type': line['type'], 'bbox': bbox, 'page': page_id})
                        else:
                            data_lis.append({'text': markdown, 'type': line['type'], 'page': page_id})
                    else:
                        if res_bbox:
                            data_lis.append({'text': html_text, 'type': line['type'], 'bbox': bbox, 'page': page_id})
                        else:
                            data_lis.append({'text': html_text, 'type': line['type'], 'page': page_id})
                else:
                    # result_image = image[max(bbox[1]-1, 0):bbox[3]+1, max(bbox[0]-5, 0):bbox[2]+3, ::-1]
                    result_image = image[max(bbox[1]-2, 0):bbox[3]+3, max(bbox[0]-5, 0):bbox[2]+5, ::-1]
                    # cv2.imwrite('./temp/temp_{}_{}.jpg'.format(str(page_id), str(i)),
                    #             result_image)

                    # structure_lis.append(result_image)
                    # if res_bbox:
                    #     temp_lis.append({'text': '', 'type': line['type'], 'bbox': bbox, 'page': page_id})
                    # else:
                    #     temp_lis.append({'text': '', 'type': line['type'], 'page': page_id})

                    result_image = image[max(bbox[1] - 2, 0):bbox[3] + 3, max(bbox[0] - 5, 0):bbox[2] + 5, ::-1]
                    # 🌟 新增容错拦截
                    if result_image is not None and result_image.size > 0:
                        structure_lis.append(result_image)
                        if res_bbox:
                            temp_lis.append({'text': '', 'type': line['type'], 'bbox': bbox, 'page': page_id})
                        else:
                            temp_lis.append({'text': '', 'type': line['type'], 'page': page_id})
                    else:
                        print(f"警告: 截取的段落图像区域为空，已跳过。", flush=True)

            if len(structure_lis) > 0:
                res_texts = self.ocr(structure_lis, lang)
                data_lis.extend(self.ocr_post_process(res_texts, temp_lis))

            del np_image
            del image
            import gc
            gc.collect()

        doc.close()

        return data_lis, lang


    def ocr_post_process(self, texts_lis, meta_region_lis):
        ocr_post_lis = []
        for region_id, res_text in enumerate(texts_lis):
            if len(res_text) == 0:
                continue
            region_meta = meta_region_lis[region_id]
            region_meta["text"] = res_text
            ocr_post_lis.append(region_meta)

        return ocr_post_lis

    def recongnize_pdf_language(self, file):
        try:
            pdf = pb.open(file)
            if len(pdf.pages) == 0:
                return 'en'
            page_text = pdf.pages[0].extract_text()
        except:
            return 'en'
        if re.search(self.zh_patter, page_text[:300].replace('医脉通', '')):
            return 'ch'
        else:
            return 'en'


    def bbox_ioa(self, box1, box2, iou=False, eps=1e-7):
        """
        Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

        Args:
            box1 (np.ndarray): A numpy array of shape (n, 4) representing n bounding boxes.
            box2 (np.ndarray): A numpy array of shape (m, 4) representing m bounding boxes.
            iou (bool): Calculate the standard IoU if True else return inter_area/box2_area.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (np.ndarray): A numpy array of shape (n, m) representing the intersection over box2 area.
        """

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

        # Intersection area
        inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
            np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
        ).clip(0)

        # Box2 area
        area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        if iou:
            box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
            area = area + box1_area[:, None] - inter_area

        # Intersection over box2 area
        return inter_area / (area + eps)


    def box_iou(self, box1, box2, eps=1e-7):
        """
        Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

        Args:
            box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
            box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)





async def task(pdf_path_name, save_path, err_log_path, processed_log):

    pdf_tools = PDFParsing(LAYOUT_PATH)

    # pdf_tools = PDFParsing()
    memerr_content = []
    pdfpageerr_content = []
    othererr_content = []

    # 损坏的文件名字存入PDFPageCountError.json中，每次运行程序自动跳过损坏文件
    pdfpageerror_logpath = os.path.join(err_log_path, "PDFPageCountError.json")

    if os.path.exists(pdfpageerror_logpath):
        with open(pdfpageerror_logpath, "r", encoding="utf-8") as f:
            pdfpageerr_content = json.load(f)

    # for pdf_path_name in pdf_path:
    json_img_path = os.path.join(save_path,pdf_path_name.split("/")[-1].replace(".pdf", "").replace(".PDF",""))
    os.makedirs(json_img_path,exist_ok=True)

    save_path_file = os.path.join(json_img_path, pdf_path_name.split("/")[-1]).replace(".pdf", ".json").replace(".PDF",".json")


    if os.path.exists(save_path_file):
        print("文件已存在。。。")
        return
    # 跳过损坏文件
    if pdfpageerr_content and pdf_path_name+"***PDFPageCountError" in pdfpageerr_content:
        print(f"跳过损坏文件。。。。。。。。。{pdf_path_name}")
        return
    # # 下载临时pdf
    # tmp_pdf_path = os.path.join(json_img_path,pdf_path_name.split("/")[-1])
    # if not os.path.exists(tmp_pdf_path):
    #     tmp_pdf_path = download_pdf(pdf_path_name, json_img_path)

    try:
        data_lis, _ = pdf_tools.pdf_parsing(pdf_path_name, None, html2markdown=False)

        # data_lis, _ = pdf_tools.pdf_parsing(pdf_path_name , lang=lang,save_figure_dir=json_img_path)
        # data_lis, _ = pdf_tools.pdf_parsing(pdf_path_name,save_figure_dir=json_img_path)
        with open(save_path_file, "w",encoding="utf-8") as f:
            json.dump(data_lis,f,ensure_ascii=False,indent=2)

        # 处理成功后记录到日志
        save_processed_file(pdf_path_name, processed_log)

        # remove_file(pdf_path_name)
        print(pdf_path_name,"============================  完成！！！=====", flush=True)
        # print('进程编号：', p_number, '进程总pdf数:', len(pdf_path))

    except Exception as e:
        # print(type(e).__name__)
        err_str = pdf_path_name + "***" + type(e).__name__
        if type(e).__name__ == "MemoryError":
            memoryerror_logpath = os.path.join(err_log_path,"MemoryError.json")
            if os.path.exists(memoryerror_logpath):
                with open(memoryerror_logpath,"r",encoding="utf-8") as f:
                    memerr_content = json.load(f)
                if err_str not in memerr_content:
                    memerr_content.append(err_str)
                    with open(memoryerror_logpath, "w", encoding="utf-8") as fe:
                        json.dump(memerr_content,fe,ensure_ascii=False,indent=2)
            else:
                memerr_content.append(err_str)
                with open(memoryerror_logpath, "w", encoding="utf-8") as fe:
                    json.dump(memerr_content, fe, ensure_ascii=False, indent=2)
            remove_dir(json_img_path)
            print(f"MemoryError错误，移除{json_img_path}！！！，====  PDF文件太大！！！=====", flush=True)
            # sys.exit()

        elif type(e).__name__ == "PDFPageCountError":
            if pdfpageerr_content:
                if err_str not in pdfpageerr_content:
                    pdfpageerr_content.append(err_str)
                    with open(pdfpageerror_logpath, "w", encoding="utf-8") as fe:
                        json.dump(pdfpageerr_content, fe, ensure_ascii=False, indent=2)
            else:
                pdfpageerr_content.append(err_str)
                with open(pdfpageerror_logpath, "w", encoding="utf-8") as fe:
                    json.dump(pdfpageerr_content, fe, ensure_ascii=False, indent=2)
            remove_dir(json_img_path)
            print(f"PDFPageCountError错误，移除{json_img_path}，====  PDF格式错误！！！=====", flush=True)
            # sys.exit()
        elif type(e).__name__ == "OutOfMemoryError":
            """显存溢出，结束进程（子进程）"""
            remove_dir(json_img_path)
            # sys.exit()
        else:
            othererror_logpath = os.path.join(err_log_path,"OtherError.json")
            if os.path.exists(othererror_logpath):
                with open(othererror_logpath, "r", encoding="utf-8") as f:
                    othererr_content = json.load(f)
                if err_str not in othererr_content:
                    othererr_content.append(err_str)
                    with open(othererror_logpath, "w", encoding="utf-8") as fe:
                        json.dump(othererr_content, fe, ensure_ascii=False, indent=2)
            else:
                othererr_content.append(err_str)
                with open(othererror_logpath, "w", encoding="utf-8") as fe:
                    json.dump(othererr_content, fe, ensure_ascii=False, indent=2)
            remove_dir(json_img_path)
            print("OtherError:",type(e).__name__,f"移除{json_img_path}，====  其它错误！！！=====", flush=True)


# 检查文件是否存在
def create_processed_log(processed_log_pattern):
    if not os.path.exists(processed_log_pattern):
        try:
            # 如果文件不存在，创建一个空的 JSON 文件
            with open(processed_log_pattern, 'w') as f:
                pass
                # json.dump({}, f, ensure_ascii=False, indent=4)  # 创建空字典或列表
                # # f.write('')  # 写入一个空字符串
            print(f"{processed_log_pattern} 已创建！", flush=True)
        except Exception as e:
            print(f"创建文件 {processed_log_pattern} 时发生错误: {e}")
    else:
        print(f"{processed_log_pattern} 文件已经存在！", flush=True)

# 加载已处理的文件
def load_processed_files(processed_log):
    processed_files = set()

    if os.path.exists(processed_log):
        try:
            with open(processed_log, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())  # 逐行读取 JSONL
                        if isinstance(data, str):  # 确保是字符串路径
                            processed_files.add(data)
                    except json.JSONDecodeError:
                        print(f"跳过损坏的 JSON 行: {line.strip()}")
        except Exception as e:
            print(f"加载处理记录失败: {e}")

    return processed_files


# 监控队列中的数量
# def monitor_queue_size(pdf_queue):
#     while True:
#         # print(f"[监控] 当前队列大小: {pdf_queue.qsize()}", flush=True)
#         time.sleep(10)

# 批量添加数据到队列
def add_data_batch(pdf_files, processed_log, pdf_queue):
    # 加载已处理的文件
    processed_files = load_processed_files(processed_log)

    while pdf_files:
        if pdf_queue.qsize() <= 1000:
            batch = pdf_files[:1000]  # 获取前500条数据
            pdf_files = pdf_files[1000:]  # 更新剩余数据

            batch_to_add = [pdf_path for pdf_path in batch if pdf_path not in processed_files]

            # 如果有未处理的文件，加入队列
            if batch_to_add:
                for pdf_path in batch_to_add:
                    pdf_queue.put(pdf_path)
                print(f"过滤后的数据加入队列，当前队列大小: {pdf_queue.qsize()}", flush=True)
            else:
                print(f"当前批次没有未处理的文件", flush=True)

        # 每次等待 2 秒钟来控制队列的速率，避免过度填充
        time.sleep(2)

    print(f"所有数据已加入队列，最终队列大小: {pdf_queue.qsize()}", flush=True)

# 独立的线程添加数据
def start_data_addition(pdf_files, processed_log, pdf_queue):
    t = threading.Thread(target=add_data_batch, args=(pdf_files, processed_log, pdf_queue))
    t.start()

def run_async_in_process(batch, new_file_folder_path, err_log_path, processed_log):

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(task(batch, new_file_folder_path, err_log_path, processed_log))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

# 处理PDF文件的任务
def process_pdf_file(pdf_path, processed_log):
    processed_files = load_processed_files(processed_log)

    if pdf_path in processed_files:
        print(f"跳过已处理文件: {pdf_path}")
        return

    # # 按照 '/' 分割路径
    # parts = pdf_path.split('/')

    # 生成保存路径
    # save_path = os.path.join(f'/root/autodl-tmp/GLS/replenish/', 'json_img_output')
    save_path = os.path.join(SAVE_PATH, 'json_img_output')
    os.makedirs(save_path, exist_ok=True)
    # 错误日志路径

    # err_log_path = os.path.join(f'/root/autodl-tmp/GLS/replenish/', 'err_file')
    err_log_path = os.path.join(SAVE_PATH, 'err_file')
    os.makedirs(err_log_path, exist_ok=True)

    run_async_in_process(pdf_path, save_path, err_log_path, processed_log)

# 进程
def processes_worker(gpu_id, mutex, pdf_queue, processed_log):
    while True:
        # try:
            # 尝试获取锁
            acquired = mutex.acquire(timeout=5)
            if acquired:
                try:
                    if not pdf_queue.empty():
                        pdf_file = pdf_queue.get()
                        # print(f"GPU {gpu_id} 获取队列数据", flush=True)
                    else:
                        # print(f"GPU {gpu_id} 队列为空，等待数据...", flush=True)
                        time.sleep(2)
                        continue
                finally:
                    mutex.release()  # 确保最终释放锁
            else:
                # print(f"GPU {gpu_id} 无法获取锁，超时！", flush=True)
                time.sleep(1)  # 等待再尝试获取锁

            if pdf_file:
                # 处理 PDF 文件
                # print(f"GPU {gpu_id} 正在处理", flush=True)
                process_pdf_file(pdf_file, processed_log)

        # except Exception as e:
        #     print(f"GPU {gpu_id} 处理时发生错误: {e}")
        #     time.sleep(1)  # 等待一段时间后重启该进程

def start_and_monitor_process(gpu_id, mutex, pdf_queue, processed_log):
    while True:
        try:
            # 启动新的进程
            p = multiprocessing.Process(target=processes_worker,
                                        args=(gpu_id, mutex, pdf_queue, processed_log))
            p.start()
            print(f"启动进程: GPU {gpu_id}", flush=True)

            # 等待进程结束
            p.join()

        except Exception as e:
            print(f"进程 {gpu_id} 出现错误: {e}")
            time.sleep(1)  # 等待一段时间后重启进程

# 启动进程处理数据
def restart_processes(mutex, pdf_queue, processes_num, processed_log, available_gpus):
    # 为每个 GPU 启动和监控进程
    for gpu_id in available_gpus:
        for i in range(processes_num):
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
            # 启动每个进程并监控
            process_monitor = multiprocessing.Process(target=start_and_monitor_process, args=(gpu_id, mutex, pdf_queue, processed_log))
            process_monitor.start()
            print(f"开始监控进程: GPU {gpu_id}, 进程 {i+1}")

import os
import glob

def load_pdf( base_urls, processed_json_root="/root/Data/rlj/ocrV5/json_img_output", sort_by_size=True, reverse=False ):
    """
    从多个 base_urls 中加载「尚未处理过」的 PDF 文件
    判断依据：PDF 文件名（不含后缀）是否已存在于 processed_json_root 中
    """

    pdf_files_all = []

    # 1️⃣ 收集所有已处理 json 的 basename（不带后缀）
    processed_names = set()
    json_files = glob.glob(os.path.join(processed_json_root, "*"))

    for jf in json_files:
        name = os.path.splitext(os.path.basename(jf))[0]
        processed_names.add(name)

    # 2️⃣ 遍历每个 base_url 下的 PDF
    for base_url in base_urls:
        pdf_files = glob.glob(
            os.path.join(base_url, "*.[pP][dD][fF]")
        )

        for pdf_path in pdf_files:
            # 跳过 0 字节
            if os.path.getsize(pdf_path) == 0:
                continue

            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

            # 跳过已处理的
            if pdf_name in processed_names:
                continue

            pdf_files_all.append(pdf_path)

    return pdf_files_all



# 按照年份处理
def process_years(processed_log, base_urls, pdf_queue, mutex, progress_num, available_gpus):
    # 检查文件是否存在
    # create_processed_log(processed_log_pattern)
    if not os.path.exists(processed_log):
        open(processed_log, "w").close()

    # 加载当前年份的 PDF 文件
    pdf_files = load_pdf(base_urls)
    print("待处理 PDF 数量:", len(pdf_files), flush=True)

    # # 启动监控数量线程
    # monitor_thread = threading.Thread(target=monitor_queue_size, args=(pdf_queue,), daemon=True)
    # monitor_thread.start()

    # 启动数据添加的线程
    start_data_addition(pdf_files, processed_log, pdf_queue)

    # 等待队列有足够数据
    while pdf_queue.empty():
        print(f"等待队列中有数据...", flush=True)
        time.sleep(3)

    # 启动进程处理数据
    restart_processes(mutex, pdf_queue, progress_num, processed_log, available_gpus)

    # 确保等待所有进程完成
    for p in multiprocessing.active_children():
        p.join()

    print("所有年份的数据处理完成，程序结束。", flush=True)

# 保存已处理文件
def save_processed_file(pdf_path, processed_log):
    lock_path = f"{processed_log}.lock"
    lock = FileLock(lock_path)  # 使用文件锁防止并发冲突

    with lock:  # 确保只有一个进程在写入
        if pdf_path:  # 确保只有非空路径被写入
            with open(processed_log, "a") as f:  # 追加模式
                f.write(json.dumps(pdf_path, ensure_ascii=False) + "\n")
        else:
            print(f"跳过空路径: {pdf_path}")

def remove_dir(folder_path):
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        print(f"文件夹移除失败，Error: {e}")

if __name__ == '__main__':

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 只有在主程序开始时设置启动方式

    # 使用 Manager 创建锁对象
    manager = multiprocessing.Manager()
    pdf_queue = manager.Queue(maxsize=3000)  # 共享队列
    mutex = manager.Lock()

    base_urls = [
        "/root/Data/rlj/ocrV5/pdf"
    ]

    processed_log = "multi_pdf_to_json_pubmed_replenish.jsonl"

    progress_num = 1  # 每个GPU的进程数量
    available_gpus = [0]

    process_years(processed_log, base_urls, pdf_queue, mutex, progress_num, available_gpus)

