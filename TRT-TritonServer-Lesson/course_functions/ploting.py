import cv2
import matplotlib.pyplot as plt

def compare_images(*images, titles=None, figsize=(15, 5)):
    """
    显示多张图像的对比。
    
    参数:
        *images: 多张图像，可以是任意数量的图像。
        titles: 图像的标题列表，长度应与图像数量一致。如果为None，则不显示标题。
        figsize: 图像显示的大小，格式为 (宽, 高)。
    """
    num_images = len(images)
    if titles is not None and len(titles) != num_images:
        raise ValueError("The length of the titles list must match the number of images.")
    
    plt.figure(figsize=figsize)
    for i, img in enumerate(images, start=1):
        plt.subplot(1, num_images, i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        if titles:
            plt.title(titles[i-1])
    plt.show()