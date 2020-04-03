from torchvision import datasets

if __name__ == "__main__":
    # download cifar10
    trainset = datasets.CIFAR10(root='./data', train=True, download=True)
    testset = datasets.CIFAR10(root='./data', train=False, download=True)

    # download mnist

