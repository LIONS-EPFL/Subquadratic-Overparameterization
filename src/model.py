import os
import argparse
import pathlib
import pickle
import warnings

import matplotlib

matplotlib.use("TKAgg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class GELU(torch.nn.Module):
    r"""Pulled from pytorch=1.4 since Tesla K40c are no longer precompiled in the binaries after pytorch=1.3.
    Applies the Gaussian Error Linear Units function:

    .. math::
        \text{GELU}(x) = x * \Phi(x)
    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def forward(self, input):
        return F.gelu(input)


def prepare_output_file(filename, output_dir):
    file_path = os.path.join(output_dir, filename)
    open(file_path, 'w').close()
    return file_path

ACT = {
    'relu': lambda: torch.nn.ReLU(),
    'elu': lambda: torch.nn.ELU(alpha=0.1),
    'gelu': lambda: GELU(),
}

class Network(torch.nn.Sequential):
    def __init__(self, d0, d1, d2, softmax=True, activation_function='relu'):
        super().__init__()
        self.add_module('linear0', torch.nn.Linear(d0, d1))
        self.add_module('activation1', ACT[activation_function]())
        self.add_module('linear1', torch.nn.Linear(d1, d2))
        if softmax:
            self.add_module('logsoftmax', torch.nn.LogSoftmax(1))


class Runner(object):

    @classmethod
    def from_parser(cls):
        """Pass and return the required command line arguments.
        :return:
        """
        parser = argparse.ArgumentParser(description='Batch size testing')
        parser.add_argument('--exp-name', type=str, default='default')
        parser.add_argument('--mse-loss', action='store_true', default=False)
        parser.add_argument('--dataset', type=str, default='MNIST')
        parser.add_argument('--batch-size', type=int, default=16,
                            help='input batch size for training (default: 16)')
        parser.add_argument('--epochs', type=int, default=60,
                            help='number of epochs to train (default: 60)')
        parser.add_argument('--d1', type=int, default=100, help='hidden units')
        parser.add_argument('--lr', type=float, default=0.1,
                            help='learning rate (default: 0.1)')
        parser.add_argument('--w1', type=float, default=0.1,
                            help='weight 1 init')
        parser.add_argument('--activation', type=str, default='relu',
                            help='relu or elu')
        parser.add_argument('--w2', type=float, default=0.1,
                            help='weight 2 init')
        parser.add_argument('--seed', type=int, default=None,
                            help='Random seed')
        parser.add_argument('--teacher', action='store_true', default=False,
                            help='Use teacher')
        parser.add_argument('--cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--log-interval', type=int, default=80,
                            help='how many batches to wait before logging training status')
        parser.add_argument('--remote', action='store_true', default=False, help='execute on HPC.')
        parser.add_argument('--dummy-run', action='store_true', default=False,
                            help='Run without doing anything.')
        args = parser.parse_args()
        return args


    def __init__(self, args):
        self.args = args

        self.activation = args.activation
        self.exp_name = args.exp_name
        self.dataset = args.dataset
        self.use_mse_loss = args.mse_loss
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.w1 = args.w1
        self.w2 = args.w2
        self.log_interval = args.log_interval
        if self.dataset == 'MNIST':
            self.layer_units = (784, args.d1, 10)
        elif self.dataset == 'CIFAR10':
            self.layer_units = (3072, args.d1, 10)

        # Set random seed
        if args.seed is None:
            args.seed = np.random.randint(0, 1e8)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.seed = args.seed
        print("seed:", self.seed)

        # GPU
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.file_paths = self._prepare_output()
        self.train_loader, self.test_loader = self._load_datasets()
        self.model, self.optimizer, self.criterion = self._prepare_model()

        # Use teacher
        if args.teacher:
            try:
                model = Network(*self.layer_units, activation_function=args.activation, softmax=True)
                model.load_state_dict(torch.load('teacher_model.pth', map_location='cpu'))
                model = model.to(self.device)
                model.eval()
            except RuntimeError:
                model = None
                warnings.warn("Teacher model was not loaded. Might be no problem if its only a local test before pushing to the server.")

            self.teacher_model = model
        else:
            self.teacher_model = None

        super(Runner, self).__init__()

    def _prepare_output(self):
        """
        Helper to initialize.
        :return: dictionary of output file paths.
        """
        # Ensure output dir is created
        output_dir = os.path.join("output", self.exp_name)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        # Store arguments
        argument_file = os.path.join(output_dir, "arguments.p")
        fp = open(argument_file, 'wb')
        pickle.dump(self.args, fp)
        fp.close()

        # Prepare output locations
        uuid = "n-{}-bs-{}".format(self.layer_units, self.batch_size)
        return dict(
            loss_file_path = prepare_output_file("loss-{}.txt".format(uuid), output_dir),
            epoch_loss_file_path = prepare_output_file("epoch-loss-{}.txt".format(uuid), output_dir),
            test_loss_file_path = prepare_output_file("test-loss-{}.txt".format(uuid), output_dir),
            loss_plot_filepath = os.path.join(output_dir, "loss-plot-{}.png".format(uuid)),
            epochs_loss_plot_filepath = os.path.join(output_dir, "epochs-loss-{}.png".format(uuid)),
            epochs_accuracy_plot_filepath = os.path.join(output_dir, "epochs-train-accuracy-{}.png".format(uuid)),
            epochs_test_accuracy_plot_filepath = os.path.join(output_dir, "epochs-test-accuracy-{}.png".format(uuid)),
        )

    def _load_datasets(self):
        """
        Helper to initialize.
        :return: (train data loader, test data loader)
        """
        # Load the data set
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        if self.dataset == 'MNIST':
            test_loader = DataLoader(
                datasets.MNIST('../data', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=self.batch_size, shuffle=True,
                **kwargs)
            train_loader = DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=self.batch_size, shuffle=True,
                **kwargs)
        elif self.dataset == 'CIFAR10':
            train_loader = DataLoader(
                datasets.CIFAR10('../data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                 ])),
                batch_size=self.batch_size, shuffle=True,
                **kwargs)
            test_loader = DataLoader(
                datasets.CIFAR10('../data', train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                 ])),
                batch_size=self.batch_size, shuffle=True,
                **kwargs)
        else:
            raise ValueError('`dataset` has to be either `MNIST` or `CIFAR10`.')

        return train_loader, test_loader

    def _prepare_model(self):
        """
        Helper to initialize.
        :return: (model, optimizer, criterion)
        """
        # Setup model, optimizer and loss
        model = Network(*self.layer_units, activation_function=self.activation, softmax=not self.use_mse_loss).to(self.device)
        optimizer = torch.optim.SGD(list(model.parameters()), lr=self.lr)
        criterion = F.mse_loss if self.use_mse_loss else F.nll_loss

        # Init parameters
        def init_weights(m):
            layer_structure = (1, 1, 1)
            for name, param in m.named_parameters():
                for i in range(len(layer_structure) - 2):
                    if name == 'linear0.weight':
                        param.data.normal_(0, self.w1)
                if name == 'linear1.weight':
                    param.data.normal_(0, self.w2)

        model.apply(init_weights)
        return model, optimizer, criterion

    def target_transform(self, true_target, data):
        """
        Helper to label based on teacher network if available.
        :param true_target:
        :param data:
        :return:
        """
        if self.teacher_model is not None:
            target = self.teacher_model(data)
            target = target.argmax(dim=1, keepdim=False)
            return target
        else:
            return true_target

    def run(self):
        losses = []
        average_losses = []
        average_accuracies = []
        test_accuracies = []

        # Train
        for epoch in range(1, self.epochs + 1):
            self.model.train()

            average_loss = 0
            average_accuracy = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                loss, correct_pred = self._gradient_step(batch_idx, epoch, data, target)

                average_loss = average_loss + loss.item()
                losses.append(loss.item())

                average_accuracy += correct_pred

            # Record loss across all batches for evaluation.
            num_batches = len(self.train_loader)
            average_loss = average_loss / num_batches
            average_losses.append(average_loss)
            average_accuracy = average_accuracy / len(self.train_loader.dataset)
            average_accuracies.append(average_accuracy)

            print("Train Epoch: {}\t\tAverage Loss: {}".format(epoch, average_loss))
            with open(self.file_paths['epoch_loss_file_path'], 'a') as fd:
                fd.write("{} {} {}\n".format(epoch, average_loss, average_accuracy))

            # Record test accuracy
            average_test_accuracy = self._calc_test_performance()
            test_accuracies.append(average_test_accuracy)

            self._update_plot(losses, average_losses, average_accuracies, test_accuracies)

            if epoch % 50 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "model.pth"))

        self._calc_final_train_performance()
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "model.pth"))

    def _gradient_step(self, batch_idx, epoch, data, target):
        # Flatten
        data = data.reshape((data.shape[0], -1))
        data = data.to(self.device)

        orig_target = self.target_transform(target, data)

        if self.use_mse_loss:
            target = torch.nn.functional.one_hot(orig_target, num_classes=10)
            target_transformed = target.float()
        else:
            target = orig_target
            target_transformed = target
        target = target.to(self.device)
        target_transformed = target_transformed.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target_transformed)
        loss.backward()
        self.optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct_pred = pred.eq(orig_target.view_as(pred)).sum().item()

        if batch_idx % self.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(self.train_loader.dataset),
                       100. * batch_idx / len(self.train_loader), loss.item()))
            with open(self.file_paths['loss_file_path'], 'a') as fd:
                fd.write("{} {} {}\n".format(epoch, batch_idx, loss.item()))

        return loss, correct_pred

    def _calc_test_performance(self):
        # Testing error at every epoch
        self.model.eval()
        average_loss = 0
        average_accuracy = 0

        for batch_idx, (data, target) in enumerate(self.test_loader):
            # Flatten
            data = data.reshape((data.shape[0], -1))
            data = data.to(self.device)

            orig_target = self.target_transform(target, data)

            if self.use_mse_loss:
                target = torch.nn.functional.one_hot(orig_target, num_classes=10)
                target_transformed = target.float()
            else:
                target = orig_target
                target_transformed = target
            target = target.to(self.device)
            target_transformed = target_transformed.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, target_transformed)
            average_loss = average_loss + loss.item()

            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            average_accuracy += pred.eq(orig_target.view_as(pred)).sum().item()

        # Record loss across all batches for evaluation.
        num_batches = len(self.test_loader)
        average_loss = average_loss / num_batches
        average_accuracy = average_accuracy / len(self.test_loader.dataset)

        print("Average Test Loss: {}".format(average_loss))
        with open(self.file_paths['test_loss_file_path'], 'a') as fd:
            fd.write("9999 {} {}\n".format(average_loss, average_accuracy))

        return average_accuracy

    def _update_plot(self, losses, average_losses, average_accuracies, test_accuracies):
        # Save plots at every epoch
        fig, ax = plt.subplots()
        ax.plot(losses)
        fig.savefig(self.file_paths['loss_plot_filepath'])

        fig, ax = plt.subplots()
        ax.plot(average_losses)
        fig.savefig(self.file_paths['epochs_loss_plot_filepath'])

        fig, ax = plt.subplots()
        ax.plot(average_accuracies)
        fig.savefig(self.file_paths['epochs_accuracy_plot_filepath'])

        fig, ax = plt.subplots()
        ax.plot(test_accuracies)
        fig.savefig(self.file_paths['epochs_test_accuracy_plot_filepath'])

    def _calc_final_train_performance(self):
        # Final train error across all batches
        self.model.eval()
        average_loss = 0
        average_accuracy = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Flatten
            data = data.reshape((data.shape[0], -1))
            data = data.to(self.device)

            orig_target = self.target_transform(target, data)

            if self.use_mse_loss:
                target = torch.nn.functional.one_hot(orig_target, num_classes=10)
                target_transformed = target.float()
            else:
                target = orig_target
                target_transformed = target
            target = target.to(self.device)
            target_transformed = target_transformed.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, target_transformed)
            average_loss = average_loss + loss

            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            average_accuracy += pred.eq(orig_target.view_as(pred)).sum().item()

        # Record loss across all batches for evaluation.
        num_batches = len(self.train_loader)
        average_loss = average_loss / num_batches
        average_accuracy = average_accuracy / len(self.train_loader.dataset)

        print("Average training Loss: {}".format(average_loss))
        with open(self.file_paths['epoch_loss_file_path'], 'a') as fd:
            fd.write("9999 {} {}\n".format(average_loss, average_accuracy))


