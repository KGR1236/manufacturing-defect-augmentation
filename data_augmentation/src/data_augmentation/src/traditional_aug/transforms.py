"""전통적 데이터 증강 기법: 90도 회전(R), 밝기 조절(B), 가우시안 노이즈(N)

포스터 [그림 3]의 OR -> TR 이미지 셋 생성 과정에서 사용되는 개별 변환 함수들.
"""
import cv2
import numpy as np


def rotate_90_left(image: np.ndarray) -> np.ndarray:
    """이미지를 왼쪽으로 90도 회전"""
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_bbox_90_left(x_center: float, y_center: float, width: float, height: float):
    """YOLO 정규화 좌표(x_center, y_center, width, height)를 왼쪽 90도 회전에 맞춰 변환"""
    new_x = y_center
    new_y = 1 - x_center
    new_width = height
    new_height = width
    return new_x, new_y, new_width, new_height


def increase_brightness(image: np.ndarray, value: int = 30) -> np.ndarray:
    """HSV의 V(명도) 채널을 올려 밝기 증가"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 10) -> np.ndarray:
    """가우시안 노이즈 추가"""
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian
    return np.clip(noisy_image, 0, 255).astype(np.uint8)
