import torch
import torch.nn as nn
from preprocessing.data_preprocessing import MetalicDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_1 = torch.nn.Linear(64 * 61 * 61, 256)
        self.linear_2 = torch.nn.Linear(256, 256)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        pred = torch.sigmoid(self.output(x)).float()

        return pred


def main():
    # Load data
    my_transforms = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((255, 255)),
                                        transforms.RandomCrop(244),
                                        transforms.ColorJitter(brightness=0.5),
                                        transforms.RandomRotation(degrees=45),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.ToTensor()])
    dataset = MetalicDataset(csv_file="images_data.csv",
                             root_dir=r"*root dir to all images*",
                             transform=my_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    metallic_model = Model()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(metallic_model.parameters(), lr=0.001, betas=(0.9, 0.99999), eps=1e-08,
                                 weight_decay=0.001, amsgrad=True)
    no_epochs = 100
    train_loss = list()
    best_loss = 1
    for epoch in range(no_epochs):
        total_train_loss = 0

        metallic_model.train()
        # training
        for itr, (image, label) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = metallic_model(image)
            loss = criterion(pred.float(), label.float())
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        total_train_loss = total_train_loss / (itr + 1)
        train_loss.append(total_train_loss)
        print('\nEpoch: {}/{}, Train Loss: {:.8f}'.format(epoch + 1, no_epochs, total_train_loss))
        if total_train_loss < best_loss:
            best_loss = total_train_loss
            best_model = metallic_model

    print("model saved")
    torch.save(best_model, "./best_model")

    import matplotlib.pyplot as plt
    plt.plot(train_loss)
    plt.ylabel('Loss Value')
    plt.show()


if __name__ == "__main__":
    main()
