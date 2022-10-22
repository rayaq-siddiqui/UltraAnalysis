import os


def calcClassWeight(n_classes=3, data_dir='cnn_data/'):
    class_names = os.listdir(data_dir)
    class_sizes = [len(os.listdir(f"{data_dir}{c}/")) for c in class_names]
    total_images = sum(class_sizes)

    class_weight = [total_images/(n_classes*n_samples) for n_samples in class_sizes]

    w = dict()
    for name, weight in zip(class_names, class_weight):
        w[name] = weight

    cw = [w['benign'], w['malignant'], w['normal']]
    return cw