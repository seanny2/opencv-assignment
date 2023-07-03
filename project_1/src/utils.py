import cv2
import numpy as np
import random

def add_salt_and_pepper_noise(image, prob):
    output = np.copy(image)
    height, width = image.shape[:2]
    num_salt = np.ceil(prob * height * width)
    coordinates = tuple(np.random.randint(0, i - 1, int(num_salt)) for i in image.shape)
    output[coordinates[:-1]] = 255
    num_pepper = np.ceil(prob * height * width)
    coordinates = tuple(np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape)
    output[coordinates[:-1]] = 0
    return output

def add_gaussian_noise(img, mean=0, std=10):
    # 이미지 배열의 shape를 가져옴
    shape = img.shape
    
    # 이미지가 1채널(gray)이면, channel 축을 추가함
    if len(shape) == 2:
        img = img[:, :, np.newaxis]
    
    # 평균값이 0이고, 표준편차가 1인 정규분포로부터 난수 생성
    noise = np.random.normal(mean, std, size=shape)
    
    # 원본 이미지와 노이즈를 더한 이미지를 반환
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def median_based_bilateral_filter(img, d, sigma_color, sigma_space, sigma_median):
    # 출력 이미지 초기화
    filtered = np.zeros_like(img)

    # 이미지 패딩
    pad_size = d // 2
    padded = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    for i in range(pad_size, img.shape[0]+pad_size):
        for j in range(pad_size, img.shape[1]+pad_size):
            # 현재 픽셀 주변 값 가져오기
            window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]

            # 중간값 계산
            median = np.median(window)

            # 색생 필터 계산
            color_filter = np.exp(-(window - median)**2 / (2 * sigma_color**2))

            # 공간 필터 계산
            x, y = np.meshgrid(np.arange(-pad_size, pad_size+1), np.arange(-pad_size, pad_size+1))
            spatial_filter = np.exp(-(x**2 + y**2) / (2 * sigma_space**2))

            # 공간, 색상 및 중앙값 필터의 곱으로 가중치를 계산한다.
            weights = spatial_filter * color_filter * np.exp(-((img[i-pad_size, j-pad_size] - median)**2) / (2 * sigma_median**2))

            # 가중치 정규화
            weights /= np.sum(weights)

            # 필터링된 픽셀 값을 윈도우 픽셀의 가중 합으로 계산
            filtered[i-pad_size, j-pad_size] = np.sum(window * weights)

    return filtered

def switching_median_filter(img, window_size, max_threshold, min_threshold):
    # 출력 이미지 초기화
    filtered = np.zeros_like(img)

    # 이미지 패딩
    pad_size = window_size // 2
    padded = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    for i in range(pad_size, img.shape[0]+pad_size):
        for j in range(pad_size, img.shape[1]+pad_size):
            # 현재 픽셀 주변 값 가져오기
            window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]

            # 중간값 계산
            median = np.median(window)

            # 최대 및 최소 임계값 계산
            max_thresh = median + max_threshold
            min_thresh = median - min_threshold

            # 스위칭 중간값 필터 적용
            pixels = window.flatten()
            filtered_pixel = median
            if np.max(pixels) > max_thresh or np.min(pixels) < min_thresh:
                pixels = pixels[(pixels > min_thresh) & (pixels < max_thresh)]
                if len(pixels) > 0:
                    filtered_pixel = np.median(pixels)

            filtered[i-pad_size, j-pad_size] = filtered_pixel

    return filtered

def bg_removal(src, bg, edge):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_img = cv2.dilate(edge, kernel)

    contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 윤곽선 선택
    max_contour = max(contours, key=cv2.contourArea)

    # 윤곽선 내부 색칠
    obj_mask = np.zeros(edge.shape, np.uint8)
    cv2.drawContours(obj_mask, [max_contour], 0, 255, thickness=-1)

    # 이미지와 마스크를 AND 연산하여 배경과 물체를 분리
    obj = cv2.bitwise_and(src, src, mask=obj_mask)  
    return obj

    # bg = cv2.resize(bg, (src.shape[1], src.shape[0]))

    # # 물체 부분을 배경 이미지와 합성
    # bg_mask = cv2.bitwise_not(obj_mask)
    # bg = cv2.bitwise_and(bg, bg, mask=bg_mask)
    # dst = cv2.add(obj, bg)      

    # return dst




