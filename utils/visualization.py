import matplotlib.pyplot as plt
import seaborn as sns
def tensor2img(tensor_data):
    image = tensor_data
    # 将张量从 (channels, height, width) 转换为 (height, width, channels)
    image = image.permute(1, 2, 0)
    # 将张量转换为 NumPy 数组
    image = image.cpu().numpy()
    # 对图像进行归一化处理，确保像素值在 [0, 1] 范围内
    image = (image - image.min()) / (image.max() - image.min())
    return image
    
def tensors_show(tensor_datas):
    for i in range(tensor_datas.shape[0]):
        # 获取当前图像的张量
        image =tensor2img(tensor_datas[i])
        # 创建一个子图来显示当前图像
        plt.subplot(1, len(tensor_datas), i + 1)
        plt.imshow(image)
        plt.axis('off')

    # 显示所有图像
    plt.show()
    plt.close()
def plot_confusion_matrix(cm):
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.rcParams.update({'font.size': 18})
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted labels",fontsize=18)
        ax.set_ylabel("True labels",fontsize=18)
        ax.set_title("Confusion Matrix",fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        
        return fig