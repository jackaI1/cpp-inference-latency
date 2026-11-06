"""Train a small classifier on generated bars, export it and quantize its weights."""
from pathlib import Path
import numpy as np
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType

OUT = Path('artifacts')
OUT.mkdir(exist_ok=True)
torch.manual_seed(12)
rng = np.random.default_rng(42)

def images(n):
    labels = rng.integers(0, 2, n)
    x = rng.normal(0, .15, (n, 1, 8, 8)).astype('float32')
    for i, label in enumerate(labels):
        position = int(rng.integers(2, 6))
        if label == 0:
            x[i, 0, :, position] += 1
        else:
            x[i, 0, position, :] += 1
    return x, labels

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    def forward(self, x):
        return self.layers(x)

model = Net()
x, y = images(512)
optimizer = torch.optim.Adam(model.parameters(), lr=.01)
for _ in range(80):
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(torch.from_numpy(x)), torch.from_numpy(y))
    loss.backward()
    optimizer.step()
model.eval()
test_x, test_y = images(128)
with (OUT / 'images.txt').open('w') as f:
    for label, image in zip(test_y, test_x):
        f.write(str(label) + ' ' + ' '.join(f'{v:.9g}' for v in image.ravel()) + '\n')
np.save(OUT / 'pytorch.npy', model(torch.from_numpy(test_x)).detach().numpy())
torch.onnx.export(model, torch.from_numpy(test_x[:1]), OUT / 'float.onnx', input_names=['image'], output_names=['logits'], dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=17, dynamo=False)
quantize_dynamic(str(OUT / 'float.onnx'), str(OUT / 'int8.onnx'), weight_type=QuantType.QInt8)
print('Exported 128 held-out images and both models to artifacts/')
