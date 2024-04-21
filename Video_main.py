import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import warnings

# Игнорирование всех предупреждений
warnings.filterwarnings("ignore")


# region 1
def apply_edge_detection_sobel(image):
    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение алгоритма Собеля для поиска краев
    edges_sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = cv2.magnitude(edges_sobel_x, edges_sobel_y)

    # Конвертирование изображения к типу uint8
    edges_sobel = cv2.convertScaleAbs(edges_sobel)

    return edges_sobel


def apply_edge_detection(image):
    # Преобразование изображения в оттенки серого
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение алгоритма Кэнни для поиска краев
    edges_canny = cv2.Canny(image, 50, 200)

    return edges_canny


def fill_closed_regions(edges_image):
    # Находим контуры на изображении
    contours, _ = cv2.findContours(edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем белое изображение такого же размера, как исходное
    filled_image = np.ones_like(edges_image)

    # Закрашиваем каждый контур белым цветом
    cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)

    return filled_image


def display_task_2(image, edges_sobel, segmented_image):
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(edges_sobel, cmap='gray')
    axes[1].set_title('Edges (Sobel)')
    axes[1].axis('off')

    axes[2].imshow(segmented_image, cmap='gray')
    axes[2].set_title('Segmented Image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# endregion


# region 2.1
# Функция для вычисления порога на основе P-tile
def compute_ptile_threshold(gray_image, percentile):
    # Получение яркостей пикселей изображения
    pixel_values = np.sort(gray_image.ravel())

    # Вычисление порога (P-tile)
    threshold_index = int(np.ceil((percentile / 100.0) * len(pixel_values)))
    threshold = pixel_values[threshold_index]

    return threshold


# Функция для бинаризации изображения на основе порога
def binarize_image(image, threshold):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


# Функция для отображения изображения
def display_images(images, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

    for i, (image, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


# endregion

# region 2.2
# Функция для определения порога методом последовательных приближений
def compute_global_threshold(image, initial_threshold=128, max_iterations=100, epsilon=1e-3):
    threshold = initial_threshold
    prev_threshold = 0
    iteration = 0

    while abs(threshold - prev_threshold) > epsilon and iteration < max_iterations:
        prev_threshold = threshold

        # Разделение пикселей на две группы по порогу
        below_threshold = image[image <= threshold]
        above_threshold = image[image > threshold]

        # Вычисление среднего значения пикселей в каждой группе
        below_mean = np.mean(below_threshold) if len(below_threshold) > 0 else 0
        above_mean = np.mean(above_threshold) if len(above_threshold) > 0 else 0

        # Вычисление нового порога как среднее значение средних значений двух групп
        threshold = (below_mean + above_mean) / 2

        iteration += 1

    return threshold


# endregion

# region 2.3

from sklearn.cluster import KMeans

"""

from sklearn.cluster import KMeans в Python используется для импорта класса KMeans из модуля sklearn.cluster,
 который предоставляет реализацию алгоритма KMeans для кластеризации данных.

KMeans - это один из наиболее популярных методов кластеризации в машинном обучении.
 Он разбивает набор данных на заранее определенное количество кластеров. 
 Алгоритм стремится минимизировать среднеквадратическое расстояние между точками данных и центроидами 
 (средними значениями) кластеров."""


# Функция для применения метода k-средних
def kmeans_segmentation(image, k_values):
    segmented_images = []

    for k in k_values:
        # Преобразование изображения в одномерный массив
        reshaped_image = image.reshape((-1, 1))

        # Применение метода k-средних
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(reshaped_image)

        segmented_image = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)

        segmented_images.append(segmented_image)

    return segmented_images


# Функция для отображения сегментированных изображений
def display_segmented_images(images, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

    for i, (image, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    for i in range(len(images), rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# endregion

# region 2.4
# Функция для отображения гистограммы изображения
def display_histogram(image):
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title('Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()


# Функция для сглаживания гистограммы
def smooth_histogram(histogram, smoothing_factors):
    smoothed_histograms = []

    for factor in smoothing_factors:
        smoothed_histogram = np.convolve(histogram, np.ones(factor) / factor, mode='same')
        smoothed_histograms.append(smoothed_histogram)

    return smoothed_histograms


def count_peaks(histogram):
    # Подсчитываем пики в гистограмме
    peaks = np.where((histogram[1:-1] > histogram[:-2]) & (histogram[1:-1] > histogram[2:]))[0]
    return len(peaks)


# endregion

# region 3
# Функция для применения адаптивного порогового метода
@jit(nopython=True, parallel=True)
def apply_adaptive_thresholding(image, k_size, c_method='mean', t_value=5):
    adaptive_image = np.zeros_like(image)
    rows, cols = image.shape

    # Применение порога для каждого пикселя на основе окрестности
    for i in prange(rows):
        for j in prange(cols):
            min_row = max(0, i - k_size)
            max_row = min(image.shape[0], i + k_size + 1)
            min_col = max(0, j - k_size)
            max_col = min(image.shape[1], j + k_size + 1)
            neighborhood_array = image[min_row:max_row, min_col:max_col]
            if c_method == 'mean':
                c_value = np.mean(neighborhood_array)
            elif c_method == 'median':
                c_value = np.median(neighborhood_array)
            elif c_method == 'min_max':
                c_value = (np.min(neighborhood_array) + np.max(neighborhood_array)) / 2
            else:
                c_value = 0
            if image[i, j] - c_value > t_value:
                adaptive_image[i, j] = 255

    return adaptive_image


# endregion


def segmentation_with_edge_detection(image_original):
    # Применение алгоритмов выделения краев
    edges_canny = apply_edge_detection(image_original)

    # Применение сегментации на основе обнаруженных краев
    segmented_image = fill_closed_regions(edges_canny)
    return segmented_image


def binarize_image_ptile_threshold(image_original):
    threshold = compute_ptile_threshold(image_original, 60)

    # Бинаризация изображения на основе порога
    binary_image = binarize_image(image_original, threshold)
    return binary_image


def binarize_image_global_threshold(image_original):
    global_threshold = compute_global_threshold(image_original)
    # Бинаризация изображения на основе глобального порога
    binary_image = binarize_image(image_original, global_threshold)
    return binary_image


def binarize_image_kmeans_segmentation(image_original):
    k_values = [4]
    segmented_images = kmeans_segmentation(image_original, k_values)
    # segmented_images = np.round(segmented_images).astype(int)
    return segmented_images[0]


def binarize_image_adaptive_thresholding_min_max(image_original):
    k_value = 9
    c_method = 'min_max'
    t_value = 10

    adaptive_image = apply_adaptive_thresholding(image_original, k_value, c_method, t_value)

    return adaptive_image


def binarize_image_adaptive_thresholding_median(image_original):
    k_value = 9
    c_method = 'median'
    t_value = 5

    adaptive_image = apply_adaptive_thresholding(image_original, k_value, c_method, t_value)

    return adaptive_image


def binarize_image_adaptive_thresholding_mean(image_original):
    k_value = 9
    c_method = 'mean'
    t_value = 5

    adaptive_image = apply_adaptive_thresholding(image_original, k_value, c_method, t_value)

    return adaptive_image


def process_video(video_path, process_method, frame_step_in=10, flag=False):
    # Чтение видео
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Определение кодека и создание объекта VideoWriter для сохранения видео
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(process_method.__name__ + ".mp4", fourcc, fps, (width, height))

    start_time = time.time()  # Засекаем время начала выполнения метода

    # Чтение каждого frame_step-го кадра из видео и обработка
    for i in range(0, frame_count, frame_step_in):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Применение функции process_method к кадру
            if flag:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_grayscale = process_method(frame)
            frame_grayscale_uint8 = cv2.convertScaleAbs(frame_grayscale)
            out.write(frame_grayscale_uint8)

            # Отображение кадра
            cv2.imshow(process_method.__name__, frame_grayscale_uint8)
            cv2.waitKey(1)  # Необходимо для корректного отображения кадра
        else:
            print("Failed to read frame")

    # Закрытие видео-файлов
    cap.release()
    out.release()

    end_time = time.time()  # Засекаем время окончания выполнения метода
    execution_time = end_time - start_time  # Вычисляем время выполнения
    print(f"Метод '{process_method.__name__}' обработал 5 секунд видео за {execution_time:.2f} секунд c "
          f"frame_step = {frame_step_in}")


if __name__ == "__main__":
    input_video_path = 'data/Kitty 5 sec.mp4'
    frame_step = 1  # каждый N-й кадр будет обработан
    # region 1
    process_video('data/Смерть клетки.mp4', segmentation_with_edge_detection, frame_step, True)
    # endregion

    # region 2.1
    # Загрузка изображения
    process_video(input_video_path, binarize_image_ptile_threshold, frame_step, True)
    # endregion

    # region 2.2
    process_video(input_video_path, binarize_image_global_threshold, frame_step, True)
    # endregion

    # region 2.3
    process_video(input_video_path, binarize_image_kmeans_segmentation, frame_step, True)
    # endregion

    # region 3
    process_video(input_video_path, binarize_image_adaptive_thresholding_mean, frame_step, True)
    process_video(input_video_path, binarize_image_adaptive_thresholding_median, frame_step, True)
    process_video(input_video_path, binarize_image_adaptive_thresholding_min_max, frame_step, True)
    # endregion
