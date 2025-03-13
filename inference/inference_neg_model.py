import pandas as pd #type:ignore
import torch #type:ignore
from PIL import Image #type:ignore
from torch.utils.data import DataLoader, Dataset #type:ignore
from torchvision.transforms import Resize, ToTensor #type:ignore
from tqdm import tqdm #type:ignore


class NegativeDataset(Dataset):
    def __init__(self, image_ids, img_dir, labels=None):
        self.image_ids = image_ids
        self.labels = labels
        self.img_dir = img_dir
        self.transforms = Resize((224, 224)), ToTensor()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.image_ids[idx]}"
        image = Image.open(img_path)
        for transform in self.transforms:
            image = transform(image)
        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image, self.image_ids[idx]


def inference(model, img_dir, test_csv):
    # Load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(test_csv)

    test_ids = test_df["Image_ID"].unique().tolist()

    test_dataset = NegativeDataset(test_ids, img_dir)

    test_loader = DataLoader(test_dataset, batch_size=32)

    # Inference on test set
    print("Evaluating on test set")
    model.eval()
    test_results = {}
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).tolist()
            for i, pred in enumerate(preds):
                test_results[ids[i]] = "NEG" if pred else "POS"

    return test_results
