import numpy as np
import torch
import torch.nn as nn
from FedLearn import FederatedScheme
from Data.Datasets import get_mnist_test
from Examples.MNIST.Model import MNIST_CNN, MNIST_2NN,MNIST_3NN
from Examples.MNIST.GenClients import get_iid_mnist, get_non_iid_mnist
import tqdm

mnist_test = get_mnist_test()


# Each client get 500 samples
# Totally 50000/500 = 100 clients
clients = get_iid_mnist(500)  


# Each client get 2 classes with 250 samples per class.
# Totally 50000/500 = 100 clients
# clients = get_non_iid_mnist(2, 250)  


model = MNIST_3NN


n_clients = 10  # number of clients picked for each FedAvg round
client_batch_size = 50  # batch size for client training
n_client_batches = 10  # number of batches client trains in one round




fed_scheme = FederatedScheme(clients, lambda: model(), nn.MSELoss(),drop_rate=0.01)

# Train for 1000 global iterations
for i in range(1000):
    # Calculate the accuracy on the test set of the global model every 100 iterations
    if i % 100 == 0:
        test_xs_np, test_ys_np = mnist_test.get_batch_xy(1000)
        test_xs = torch.from_numpy(test_xs_np).to(fed_scheme.device)
        test_ys = torch.from_numpy(test_ys_np).to(fed_scheme.device)

        pred_ys = fed_scheme.global_model(test_xs).detach().cpu().numpy()
        acc = np.mean(np.argmax(pred_ys, axis=-1) == np.argmax(test_ys_np, axis=-1))
        print(f"Global round {i:6d}, test accuracy {acc:.3f}")

    # The learning rate should not be too large, otherwise the loss will explode
    fed_scheme.fed_avg_one_step(n_clients, client_batch_size, n_client_batches, 0.0005)


# fed per algorithm
print("use fed per algorithm")
for i in tqdm.tqdm(range(1000)):
    if i % 100 ==0 and i!=0:
        test_xs_np, test_ys_np = mnist_test.get_batch_xy(1000)
        test_xs = torch.from_numpy(test_xs_np).to(fed_scheme.device)
        test_ys = torch.from_numpy(test_ys_np).to(fed_scheme.device)
        # test all the model in the client_models
        acc=0
        for model in fed_scheme.client_models.values():
            pred_ys = model(test_xs).detach().cpu().numpy()
            acc += np.mean(np.argmax(pred_ys, axis=-1) == np.argmax(test_ys_np, axis=-1))
        print(f"Global round {i:6d}, test accuracy {acc/len(fed_scheme.client_models):.3f}")


    # The learning rate should not be too large, otherwise the loss will explode
    fed_scheme.fed_per_one_step(100, client_batch_size, n_client_batches, 0.0005)
