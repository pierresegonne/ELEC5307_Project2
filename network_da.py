'''
this script is for the Task 2 domain adaptation network of Project 2.

-------------------------------------------
'''
import torch
import torchvision
import torch.nn as nn

class Reverse(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x, lbd):
    ctx.lbd = lbd
    return x.view_as(x) # tensor.view_as(other) is equivalent to tensor.view(other.size())

  @staticmethod
  def backward(ctx, grad_output):
    output = grad_output.neg() * ctx.lbd
    return output, None

class Network_DA(nn.Module):
    def __init__(self, number_classes=10):
        super(Network_DA, self).__init__()
        # define the network layers

        model_alexnet = torchvision.models.alexnet(pretrained=True)

        self.features = model_alexnet.features

        self.fc = model_alexnet.classifier[:-1] # Remove last layer
        bottleneck_features = model_alexnet.classifier[-1].in_features

        self.bottleneck = nn.Sequential(
          nn.Linear(bottleneck_features, int(bottleneck_features / 2)),
          nn.ReLU(inplace=True),
        )

        self.class_classifier = nn.Sequential(
          nn.Linear(int(bottleneck_features / 2), number_classes)
        )

        self.domain_classifier = nn.Sequential(
          nn.Linear(int(bottleneck_features / 2), 1024), # 1024 is recommended in paper.
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(1024, 1024),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(1024, 2),
        )

    def forward(self, input, lbd=0.1):
        # define the network structure

        features = self.features(input)
        features = features.view(-1, 256*6*6)
        fc = self.fc(features)
        bottleneck = self.bottleneck(fc)
        reverse_bottleneck = Reverse.apply(bottleneck, lbd)

        predicted_classes = self.class_classifier(bottleneck)
        predicted_domains = self.domain_classifier(reverse_bottleneck)

        return predicted_classes, predicted_domains

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dann_test = Network_DA().to(device)
    INPUT_SIZE = (3, 227, 227)
    input_random = torch.rand((1, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])).to(device)
    print('\nForward Test\n', dann_test(input_random))
