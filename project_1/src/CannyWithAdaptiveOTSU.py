import cv2
import numpy as np
from skimage.filters import threshold_otsu
from utils import median_based_bilateral_filter, switching_median_filter

class CannyWithAdaptiveOTSU:

    def __init__(self, img, block_size):
        self.img = img
        self.pos_ck = np.zeros(img.shape[:2], np.uint8)
        self.new_canny = np.zeros(img.shape[:2], np.uint8)
        self.height, self.width = img.shape
        self.block_size = block_size


    def nonmax_suppression(self, sobel, direct):			#비 최대치 억제 함수
        rows, cols = sobel.shape[:2]
        dst = np.zeros((rows, cols), np.float32)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):			# 행렬 처리를 통해 이웃 화소 가져오기
                values = sobel[i-1:i+2, j-1:j+2].flatten()
                first = [3, 0, 1, 2]
                id = first[direct[i, j]]
                v1, v2 = values[id], values[8-id]
                dst[i, j] = sobel[i, j] if (v1 < sobel[i , j] > v2) else 0
                
        return dst

    def trace(self, max_sobel, i, j, low):
        h, w = max_sobel.shape
        if (0 <= i < h and 0 <= j < w) == False: return  # 추적 화소 범위 확인
        if self.pos_ck[i, j] == 0 and max_sobel[i, j] > low:
            self.pos_ck[i, j] = 255
            self.new_canny[i, j] = 255

            self.trace(max_sobel, i - 1, j - 1, low)				# 추적 함수 재귀 호출 - 8방향 추적
            self.trace(max_sobel, i    , j - 1, low)
            self.trace(max_sobel, i + 1, j - 1, low)
            self.trace(max_sobel, i - 1, j    , low)
            self.trace(max_sobel, i + 1, j    , low)
            self.trace(max_sobel, i - 1, j + 1, low)
            self.trace(max_sobel, i    , j + 1, low)
            self.trace(max_sobel, i + 1, j + 1, low)

    def hysteresis_th(self, max_sobel, low, high):                # 이력 임계값 수행
        rows, cols = max_sobel.shape[:2]
        for i in range(1, rows - 1):  # 에지 영상 순회
            for j in range(1, cols - 1):
                if max_sobel[i, j] > high:  self.trace(max_sobel, i, j, low)  # 추적 시작  

    def get_result(self, blur='med_bilateral'):
        # 1. 필터링
        if blur=='median':
            filtered_img = cv2.medianBlur(self.img, 5)
        elif blur=='gaussian':
            filtered_img = cv2.GaussianBlur(self.img, (5,5),  1.4)
        elif blur=='med_bilateral':
            filtered_img = median_based_bilateral_filter(self.img, 5, 40, 10, 15)
        elif blur=='switching_med_bilateral':
            filtered_img = switching_median_filter(self.img, 3, 100, 30)
            # filtered_img = cv2.bilateralFilter(filtered_img, -1, 10, 5)

        # 2-1. 그래디언트 크기
        Gx = cv2.Sobel(np.float32(filtered_img), cv2.CV_32F, 1, 0, 3)  # x방향 마스크
        Gy = cv2.Sobel(np.float32(filtered_img), cv2.CV_32F, 0, 1, 3)  # y방향 마스크
        sobel = np.fabs(Gx) + np.fabs(Gy)  # 두 행렬 절댓값 덧셈

        # 2-2. 그래디언트 방향
        directs = cv2.phase(Gx, Gy) / (np.pi / 4)
        directs = directs.astype(int) % 4

        # 3. 비최대치 억제
        max_sobel = self.nonmax_suppression(sobel, directs)   # 비최대치 억제

        # 4. 블록 기반 OTSU: 높은 임계값, 낮은 임계값 결정
        max_var_param = 0

        for i in range(0, self.height, self.block_size):
            for j in range(0, self.width, self.block_size):

                block = max_sobel[i:i+self.block_size, j:j+self.block_size]

                threshold = threshold_otsu(block)   # 임계값 자동 결정

                binary_block = block > threshold

                # 분산을 이용한 기준 파라미터 계산
                block_var_param = np.var(block[binary_block])   
                if block_var_param > max_var_param:     # 최대가 되는 임계값 채택
                    max_var_param = block_var_param
                    global_thresh = threshold

        low_thresh = global_thresh * 0.4
        high_thresh = global_thresh * 1.5

        # 5. 윤곽선 연결성 분석
        self.hysteresis_th(max_sobel, low_thresh, high_thresh)     # 이력 임계값
        # self.new_canny = cv2.Canny(self.img, low_thresh, high_thresh)
        
        return self.new_canny