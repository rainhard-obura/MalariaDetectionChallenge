from tqdm import tqdm

from .utils import initialize_optimizer, get_train_dataloader


def train_one_epoch(
    config, model, optimizer, scheduler, train_data_loader, device, ema
):
    model.train()
    total_loss = 0
    num_batches = 0

    with tqdm(train_data_loader, desc="Training") as pbar:
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            loss = loss / config["training"]["n_accum"]
            loss.backward()

            if (i + 1) % config["training"]["n_accum"] == 0:
                optimizer.step()

                if ema is not None:
                    ema.update()  # Update EMA

                scheduler.step()

            num_batches += 1

            pbar.set_postfix({"Avg loss": f"{total_loss / num_batches:.4f}"})

    return total_loss / num_batches


def train(config, model, processor, data_dir, device):
    """
    Train the model for multiple epochs.

    Args:
        cfg (DictConfig): The configuration object.
        model (torch.nn.Module): The model to train.
        train_data_loader (torch.utils.data.DataLoader): DataLoader containing the training data.
        device (torch.device): The device to use for training (CPU or GPU).
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
    """
    model.to(device)
    optimizer, scheduler, ema = initialize_optimizer(model, config)

    train_dataloader = get_train_dataloader(config, data_dir, processor)

    for epoch in range(config["training"]["num_epochs"]):
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        train_loss = train_one_epoch(
            config, model, optimizer, scheduler, train_dataloader, device, ema
        )
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")

    return train_loss
