import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from datasets import load_dataset
from typing import List, Tuple
import numpy as np


class Qwen3Classifier(nn.Module):
    def __init__(self, model_path: str, num_classes: int):
        super().__init__()
        self.base_model, self.tokenizer = load(model_path)
        self.num_classes = num_classes

        hidden_size = self.base_model.args.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array = None):
        hidden_states = self._get_hidden_states(input_ids)

        if attention_mask is not None:
            seq_lengths = attention_mask.sum(axis=1).astype(mx.int32) - 1
            batch_indices = mx.arange(hidden_states.shape[0])
            last_hidden = hidden_states[batch_indices, seq_lengths, :]
        else:
            last_hidden = hidden_states[:, -1, :]

        return self.classifier(last_hidden)

    def _get_hidden_states(self, input_ids: mx.array):
        h = self.base_model.model.embed_tokens(input_ids)
        T = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        mask = mask.astype(h.dtype)

        for layer in self.base_model.model.layers:
            h = layer(h, mask=mask)

        return self.base_model.model.norm(h)


class AGNewsDataset:
    """AG News: 4-class news classification (World, Sports, Business, Sci/Tech)"""

    def __init__(self, tokenizer, max_length: int = 256, split: str = "train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id or 0

        dataset = load_dataset("ag_news", split=split)
        self.texts = dataset["text"]
        self.labels = dataset["label"]

        self.num_classes = 4
        self.label_names = ["World", "Sports", "Business", "Sci/Tech"]

    def __len__(self):
        return len(self.texts)

    def get_batch(self, indices: List[int]) -> Tuple[mx.array, mx.array, mx.array]:
        texts = [self.texts[i] for i in indices]
        labels = [self.labels[i] for i in indices]

        encodings = [self.tokenizer.encode(t) for t in texts]

        padded, masks = [], []
        for enc in encodings:
            if len(enc) > self.max_length:
                enc = enc[:self.max_length]
            pad_len = self.max_length - len(enc)
            padded.append(enc + [self.pad_id] * pad_len)
            masks.append([1] * (self.max_length - pad_len) + [0] * pad_len)

        return mx.array(padded), mx.array(masks), mx.array(labels)


def batch_iterator(dataset: AGNewsDataset, batch_size: int, shuffle: bool = True):
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(dataset), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size].tolist()
        yield dataset.get_batch(batch_indices)


def loss_fn(model, input_ids, attention_mask, labels):
    logits = model(input_ids, attention_mask)
    return nn.losses.cross_entropy(logits, labels, reduction="mean")


def eval_accuracy(model, dataset: AGNewsDataset, batch_size: int = 32, max_batches: int = 50):
    correct, total = 0, 0

    for i, (input_ids, attention_mask, labels) in enumerate(
            batch_iterator(dataset, batch_size, shuffle=False)
    ):
        if i >= max_batches:
            break

        logits = model(input_ids, attention_mask)
        preds = mx.argmax(logits, axis=1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
        mx.eval(correct)

    return correct / total


def train(
        model_path: str = "Qwen/Qwen3-4B",
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        num_epochs: int = 3,
        max_length: int = 256,
        eval_every: int = 100,
        max_train_batches: int = 500,
):
    print("Loading model...")
    model = Qwen3Classifier(model_path, num_classes=4)

    # Freeze base, train only classifier head
    model.base_model.freeze()

    print("Loading AG News dataset...")
    train_dataset = AGNewsDataset(model.tokenizer, max_length, split="train")
    test_dataset = AGNewsDataset(model.tokenizer, max_length, split="test")

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    print(f"Classes: {train_dataset.label_names}")

    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    step = 0
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        epoch_losses = []

        for input_ids, attention_mask, labels in batch_iterator(train_dataset, batch_size):
            loss, grads = loss_and_grad_fn(model, input_ids, attention_mask, labels)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters())

            epoch_losses.append(loss.item())
            step += 1

            if step % eval_every == 0:
                avg_loss = np.mean(epoch_losses[-eval_every:])
                accuracy = eval_accuracy(model, test_dataset, batch_size)
                print(f"Step {step}: loss={avg_loss:.4f}, test_acc={accuracy:.4f}")

            if step >= max_train_batches:
                break

        if step >= max_train_batches:
            break

    print("\nFinal evaluation...")
    final_acc = eval_accuracy(model, test_dataset, batch_size, max_batches=100)
    print(f"Final test accuracy: {final_acc:.4f}")

    return model


def demo_inference(model, texts: List[str]):
    label_names = ["World", "Sports", "Business", "Sci/Tech"]

    pad_id = model.tokenizer.pad_token_id or 0
    encodings = [model.tokenizer.encode(t) for t in texts]
    max_len = max(len(e) for e in encodings)

    padded, masks = [], []
    for enc in encodings:
        pad_len = max_len - len(enc)
        padded.append(enc + [pad_id] * pad_len)
        masks.append([1] * len(enc) + [0] * pad_len)

    logits = model(mx.array(padded), mx.array(masks))
    probs = mx.softmax(logits, axis=1)
    preds = mx.argmax(logits, axis=1)
    mx.eval(probs, preds)

    print("\n=== Inference Demo ===")
    for text, pred, prob in zip(texts, preds.tolist(), probs.tolist()):
        print(f"\nText: {text[:80]}...")
        print(f"Prediction: {label_names[pred]} ({prob[pred]:.2%})")


if __name__ == "__main__":
    model = train(
        model_path="Qwen/Qwen3-4B",
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=1,
        max_train_batches=300,
    )

    demo_inference(model, [
        "The stock market surged today as tech companies reported strong earnings.",
        "Manchester United defeated Chelsea 2-1 in a thrilling Premier League match.",
        "NASA announced a new mission to explore the moons of Jupiter.",
        "Tensions rise in the Middle East as diplomatic talks break down.",
    ])