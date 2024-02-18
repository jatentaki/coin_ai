import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from torchvision.transforms import v2 as transforms

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from data import CoinDataset, build_coin_types
from model import (
    DinoWithHead,
    AttentionReadout,
    SigmoidLoss,
    DotProductLoss,
    MarginLoss,
    AccuracyMetric,
)

if __name__ == "__main__":
    coin_types = build_coin_types(
        "/Users/jatentaki/Data/archeo/coins/krzywousty-cropped/images"
    )

    augmentation = transforms.Compose(
        [
            transforms.ColorJitter(),
            transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
            transforms.RandomResizedCrop((224, 224), scale=(0.75, 1.0)),
            transforms.Grayscale(num_output_channels=3),
        ]
    )

    coin_dataset = CoinDataset(
        coin_types,
        batch_size=128,
        min_in_class=8,
        n_batches=1_000,
        augmentation=augmentation,
    )
    dataloader = torch.utils.data.DataLoader(
        coin_dataset, batch_size=1, num_workers=4, collate_fn=coin_dataset.collate_fn
    )

    device = torch.device("mps")
    model = DinoWithHead(AttentionReadout()).to(device)
    # loss_fn = SigmoidLoss().to(device)
    # loss_fn = DotProductLoss().to(device)
    loss_fn = MarginLoss().to(device)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-3
    )

    # losses = []
    # with tqdm(dataloader, total=coin_dataset.n_batches) as pbar:
    #    for images, labels in pbar:
    #        images = images.to(device)
    #        labels = labels.to(device)
    #        embeddings = model(images)
    #        loss = loss_fn(embeddings, labels)
    #        losses.append(loss.item())
    #        pbar.set_postfix({'loss': loss.item()})
    #        optim.zero_grad()
    #        loss.backward()
    #        optim.step()

    # torch.save(model.state_dict(), 'model_margin.pth')

    # plt.plot(losses)
    # plt.show()

    model.load_state_dict(torch.load("model_dot.pth"))

    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        # embeddings = model(images)
        embeddings = torch.randn(128, 32).to(device)
    similarity = loss_fn.similarity(embeddings).cpu()

    accuracy_metric = AccuracyMetric(loss_fn.similarity)
    print("accuracy", accuracy_metric(embeddings, labels))

    gt_similarity = (labels.unsqueeze(1) == labels.unsqueeze(0)).cpu()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))
    a1.imshow(similarity)
    a2.imshow(gt_similarity)
    plt.show()

    masked_similarity = similarity.clone()
    masked_similarity[gt_similarity] = 0

    linear_argmax = masked_similarity.argmax()
    maxes, argmaxes = masked_similarity.ravel().topk(20)
    print(maxes[::2])
    argmaxes = argmaxes[
        ::2
    ]  # the matrix is symmetric, so we only need half of the argmaxes
    i_argmax, j_argmax = argmaxes // masked_similarity.size(
        0
    ), argmaxes % masked_similarity.size(0)

    for i, j in zip(i_argmax, j_argmax):
        image_1 = images[i].permute(1, 2, 0).cpu().numpy()
        image_2 = images[j].permute(1, 2, 0).cpu().numpy()

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        a1.imshow(image_1)
        a2.imshow(image_2)
    plt.show()
