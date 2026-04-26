from dataset import SkinDataset, train_transform

dataset = SkinDataset("../splits/train.csv", transform=train_transform)

print("Dataset size:", len(dataset))

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)