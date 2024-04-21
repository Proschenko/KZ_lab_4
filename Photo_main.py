import cv2
import numpy as np
import matplotlib.pyplot as plt

#region 1

def apply_edge_detection(image):
    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение алгоритма Кэнни для поиска краев
    edges_canny = cv2.Canny(gray_image, 50, 150)

    return edges_canny



def apply_segmentation(image, edges_sobel):
    # Применение маски изображения на основе обнаруженных краев
    segmented_image = cv2.bitwise_and(image, image, mask=edges_sobel)

    return segmented_image

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


#endregion


#region 2.1
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
#endregion

#region 2.2
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

#endregion

#region 2.3

from sklearn.cluster import KMeans
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

    # Ensure all empty subplots are turned off
    for i in range(len(images), rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()
    plt.show()


#endregion

#region 2.4
# Функция для отображения гистограммы изображения
def display_histogram(image, T_value):
    histogram, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    peak_count = count_peaks(histogram)

    plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title(f"Image Histogram, T={int(T_value)}, Peaks={peak_count}")
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.axvline(x=T_value, color='red', linestyle='--')  # Добавляем вертикальную линию
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
#endregion

# region 3
# Функция для применения адаптивного порогового метода

def apply_adaptive_thresholding(image, k_size, c_method='mean', threshold_type=cv2.THRESH_BINARY, t_value=5):
    # Применение адаптивного порогового метода
    k2_size = k_size*2+1
    if c_method == 'mean':
        adaptive_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, threshold_type, k2_size, t_value)
    elif c_method == 'median':
        adaptive_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, k2_size, t_value)
    elif c_method == 'min_max':
        adaptive_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, threshold_type, k2_size, t_value)

        # Вычисление окрестности для каждого пикселя
        neighborhood = cv2.boxFilter(image, -1, (k2_size, k2_size))

        # Применение порога для каждого пикселя на основе окрестности
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                neighborhood_array = neighborhood[i:i+k2_size, j:j+k2_size]
                c_value = (int(np.min(neighborhood_array)) + int(np.max(neighborhood_array))) / 2
                if image[i, j] - c_value > t_value:
                    adaptive_image[i, j] = 255

    return adaptive_image

#endregion

if __name__ == "__main__":

    filename_img='data/Money.jpg'
    # Загрузка изображения
    image_original = cv2.imread(filename_img)

    # region 1
    # Применение алгоритмов выделения краев
    edges_sobel = apply_edge_detection(image_original)

    # Применение сегментации на основе обнаруженных краев
    segmented_image = apply_segmentation(image_original, edges_sobel)

    # Отображение изображений
    display_task_2(image_original, edges_sobel, segmented_image)

    # endregion

    #region 2.1
    # Загрузка изображения
    image = cv2.imread(filename_img, cv2.IMREAD_GRAYSCALE)

    # Список значений P-tile
    ptile_values = [30, 40, 45, 50, 55, 60]

    binary_images = []

    # Вычисление порога и бинаризация изображения для каждого значения P-tile
    for ptile in ptile_values:
        # Вычисление порога на основе P-tile
        threshold = compute_ptile_threshold(image, ptile)

        # Бинаризация изображения на основе порога
        binary_image = binarize_image(image, threshold)
        binary_images.append(binary_image)

    # Отображение всех бинаризованных изображений на одном графике
    display_images(binary_images, [f'2.1 img (P-tile={ptile})' for ptile in ptile_values], 2, 3)
    #endregion

    # region 2.2
    # Вычисление глобального порога методом последовательных приближений
    global_threshold = compute_global_threshold(image)

    # Бинаризация изображения на основе глобального порога
    binary_image = binarize_image(image, global_threshold)


    # Отображение бинаризованного изображения
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Image (Global Threshold)')
    plt.axis('off')
    plt.show()

    # Отображение гистограммы изображения
    display_histogram(image,global_threshold)

    # endregion

    # region 2.3
    # Загрузка изображения для сегментации методом k-средних
    image_segmentation = cv2.imread(filename_img, cv2.IMREAD_GRAYSCALE)

    # Список значений k для метода k-средних
    k_values = [2, 4, 8, 16]

    # Применение метода k-средних для сегментации изображения
    segmented_images = kmeans_segmentation(image_segmentation, k_values)

    # Отображение сегментированных изображений
    display_segmented_images(segmented_images, [f'2.3 img (k={k})' for k in k_values], 1, len(k_values))

    # Отображение гистограммы изображения
    display_histogram(image_segmentation,global_threshold)
    # endregion

    # region 2.4
    # Вычислим исходную гистограмму
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Количество сглаживаний для сравнения
    smoothing_factors = [3, 5, 9]
    smoothed_histograms = smooth_histogram(histogram.ravel(), smoothing_factors)

    # Получим количество пиков для каждой сглаженной гистограммы
    peak_counts = [count_peaks(histogram) for histogram in smoothed_histograms]

    # Выведем три гистограммы отдельно
    for i, (smoothed_histogram, peak_count) in enumerate(zip(smoothed_histograms, peak_counts)):
        plt.figure()
        plt.plot(smoothed_histogram)
        plt.title(f'Smoothed Histogram with Peak Count: {peak_count}, Smoothing Factor: {smoothing_factors[i]}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

    plt.show()
    # endregion

    # region 3
    # Задание значений для сравнения
    k_values = [3,  37]
    c_methods = ['median', 'min_max']
    t_values = [5,  10]


    # Применение адаптивного порогового метода с различными параметрами
    fig, axes = plt.subplots(len(k_values), len(c_methods) * len(t_values), figsize=(15, 10))

    for i, k in enumerate(k_values):
        for j, c_method in enumerate(c_methods):
            for l, t_value in enumerate(t_values):
                # Применение адаптивного порогового метода
                adaptive_image = apply_adaptive_thresholding(image, k, c_method, cv2.THRESH_BINARY, t_value)

                # Отображение результата
                axes[i, j * len(t_values) + l].imshow(adaptive_image, cmap='gray')
                axes[i, j * len(t_values) + l].set_title(f'k={k}, C={c_method}, T={t_value}')
                axes[i, j * len(t_values) + l].axis('off')


    plt.tight_layout()
    plt.show()

    k_values = [3, 37]
    c_methods = ['mean']
    t_values = [5, 10]

    fig, axes = plt.subplots(len(k_values), len(c_methods) * len(t_values), figsize=(15, 10))

    for i, k in enumerate(k_values):
        for j, c_method in enumerate(c_methods):
            for l, t_value in enumerate(t_values):
                # Применение адаптивного порогового метода
                adaptive_image = apply_adaptive_thresholding(image, k, c_method, cv2.THRESH_BINARY, t_value)

                # Отображение результата
                axes[i, j * len(t_values) + l].imshow(adaptive_image, cmap='gray')
                axes[i, j * len(t_values) + l].set_title(f'k={k}, C={c_method}, T={t_value}')
                axes[i, j * len(t_values) + l].axis('off')

    plt.tight_layout()
    plt.show()
    # endregion
