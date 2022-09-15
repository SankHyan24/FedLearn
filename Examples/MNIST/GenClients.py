import numpy as np
from Data.Datasets import MNIST_Dataset, get_mnist_train, get_mnist_test


def get_iid_mnist(samples_per_client: int):
    """
    Get identically distributed mnist datasets
    i.e. Each client will have the same distribution of data
    This is done by simply random distribute the samples
    :param samples_per_client:
    :return:
    """
    mnist_train_data = get_mnist_train().data
    client_datasets = []
    for i in range(int(mnist_train_data.shape[0] / samples_per_client)):
        client_datasets.append(MNIST_Dataset(mnist_train_data[i * samples_per_client: (i + 1) * samples_per_client]))
    return client_datasets


def get_non_iid_mnist(classes_per_client: int, samples_per_class: int):
    """
    Get non-identically distributed mnist dataset
    Each client will have data of several digits
    :param classes_per_client:
    :param samples_per_class:
    :return:
    """
    mnist_train_data = get_mnist_train().data
    class_counts = [int(5000 / samples_per_class) for _ in range(10)]
    mnist_classified = [mnist_train_data[mnist_train_data[:, 0] == i] for i in range(10)]
    clients = []
    clients_classes=[]

    while sum(class_counts) >= 2:
        i = np.random.randint(10)
        client_classes = []
        while len(client_classes) != 2:
            if class_counts[i] != 0:
                j = class_counts[i]
                client_classes.append(mnist_classified[i][(j - 1) * classes_per_client: j * classes_per_client])
                class_counts[i] -= 1
            i = (i + 1) % 10
        clients.append(MNIST_Dataset(np.vstack(client_classes)))
    # 统计每个client中含有的类别
    for client in clients:
        client_classes=[]
        for i in range(10):
            if len(client.data[client.data[:,0]==i])!=0:
                client_classes.append(i)
        clients_classes.append(client_classes)
    return clients,clients_classes

# 生成非iid测试数据集
def get_non_iid_mnist_test(classes: list, samples_per_class: int):
    """
    Get non-identically distributed mnist dataset
    Each client will have data of several digits
    :param classes: classes that the test set contain:
    :param samples_per_class:
    :return:xs ys
    """
    mnist_test_data = get_mnist_test().data
    mnist_classified = [mnist_test_data[mnist_test_data[:, 0] == i] for i in range(10)]
    test_data = []
    # 测试集里只允许标签在classes中的类别
    for i in classes:
        test_data.append(mnist_classified[i][:samples_per_class])
    test_data = np.vstack(test_data)
    ys = test_data[:, 0].astype(int)
    ys=np.eye(10, dtype=np.float32)[ys]
    xs=test_data[:, 1:]
    return (xs - 128) / 128, ys

