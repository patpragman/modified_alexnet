import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model
from sklearn.metrics import classification_report
import yaml
from pprint import pprint
from pathlib import Path
from hashlib import sha256

HOME_DIRECTORY = Path.home()
SEED = 42  # for consistency - I want all of the models to get the same data


# Define the architecture of the MLP for image classification
class AlexNet(nn.Module):
    """
    https://blog.paperspace.com/alexnet-pytorch/

    the majority of the implementation comes from here and is slightly modified for the different size
    and readability.
    """
    def __init__(self, num_classes=2,  # we're doing binary classification, so no sense using all 10
                 activation_function='relu',
                 ):
        super(AlexNet, self).__init__()

        if activation_function.lower() == "relu":
            fn = nn.ReLU()
        elif activation_function.lower() == "leaky_relu":
            fn = nn.LeakyReLU(0.1)
        else:
            fn = nn.Tanh()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),  # batch norm is a modification
            fn,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer_2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            fn,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer_3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            fn)
        self.layer_4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            fn)
        self.layer_5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            fn,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.fully_connected_1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6400, 4096),
            fn)
        self.fully_connected_2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            fn)
        self.fully_connected_3 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        # flatten here
        x = x.reshape(x.size(0), -1)
        x = self.fully_connected_1(x)
        x = self.fully_connected_2(x)
        x = self.fully_connected_3(x)

        # now you can spit out the remaining x
        return x


with open("sweep.yml", "r") as yaml_file:
    sweep_config = yaml.safe_load(yaml_file)

sweep_id = wandb.sweep(sweep=sweep_config)


def find_best_model():
    # config for wandb

    # Initialize wandb
    config = wandb.config

    # creating the model stuff
    input_size = config.input_size  # AlexNet only accepts 224 x 224 sized images
    num_classes = 2  # this doesn't ever change either - we're doing binary classification
    learning_rate = config.learning_rate
    epochs = wandb.config.epochs

    # Create the MLP-based image classifier model
    model = AlexNet(num_classes,
                    activation_function=config.activation_function)


    path = f"{HOME_DIRECTORY}/data/0.35_reduced_then_balanced/data_{config.input_size}"
    dataset = FloatImageDataset(directory_path=path,
                                true_folder_name="entangled", false_folder_name="not_entangled"
                                )

    training_dataset, testing_dataset = train_test_split(dataset, train_size=0.75, random_state=SEED)
    batch_size = config.batch_size

    # create the dataloaders
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimzer parsing logic:
    if config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = train_and_test_model(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                                   model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs,
                                   device="cpu", wandb=wandb, verbose=False)

    # save the model
    model_hash = sha256(str(config))
    model_name = f"alexnet_{model_hash.hexdigest()}"
    if not os.path.isdir(f"models/{model_name}"):
        os.mkdir(f"models/{model_name}")


    y_true, y_pred = history['y_true'], history['y_pred']
    cr = classification_report(y_true=y_true, y_pred=y_pred)

    report = [
        model_name, cr, str(model)
    ]
    with open(f"models/{model_name}/report.md", "w") as report_file:
        report_file.writelines(report)


    # Log hyperparameters to wandb
    wandb.log(dict(config))


if __name__ == "__main__":
    wandb.agent(sweep_id, function=find_best_model)

    # Specify your W&B project and sweep ID
    project_name = "AlexNet"

    # Fetch sweep runs
    api = wandb.Api()
    sweep = api.sweep(f"{project_name}/{sweep_id}")
    runs = list(sweep.runs)