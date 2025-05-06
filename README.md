import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import convolve2d
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# 数据加载函数
def load_data(directory, target_size=(32, 32), max_samples=100):
    images = []
    labels = []
    classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

    for label, cls in enumerate(classes):
        cls_path = os.path.join(directory, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_samples]

        for file in files:
            try:
                img = imread(os.path.join(cls_path, file))
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                img = resize(img, target_size, anti_aliasing=True)
                images.append(img.transpose(2, 0, 1))  # (C,H,W)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue

    if len(images) == 0:
        raise ValueError("No images found in the directory!")

    return np.array(images), np.eye(len(classes))[labels]


# 激活函数
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


# 卷积层
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size):
        scale = np.sqrt(2. / (in_channels * kernel_size ** 2))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
        self.kernel_size = kernel_size
        self.output_shape = None

    def forward(self, X):
        self.X = X
        batch_size, _, in_h, in_w = X.shape

        out_h = in_h - self.kernel_size + 1
        out_w = in_w - self.kernel_size + 1

        self.output = np.zeros((batch_size, self.W.shape[0], out_h, out_w))

        for i in range(batch_size):
            for j in range(self.W.shape[0]):
                for k in range(self.W.shape[1]):
                    self.output[i, j] += convolve2d(self.X[i, k], self.W[j, k], mode='valid')
                self.output[i, j] += self.b[j]

        self.output = relu(self.output)
        self.output_shape = self.output.shape[1:]
        return self.output

    def backward(self, dout, lr):
        dout = dout * (self.output > 0)
        batch_size = dout.shape[0]

        dW = np.zeros_like(self.W)
        db = np.sum(dout, axis=(0, 2, 3)) / batch_size

        for i in range(batch_size):
            for j in range(self.W.shape[0]):
                for k in range(self.W.shape[1]):
                    dW[j, k] += convolve2d(self.X[i, k], dout[i, j], mode='valid')

        dX = np.zeros_like(self.X)
        for i in range(batch_size):
            for j in range(self.W.shape[1]):
                for k in range(self.W.shape[0]):
                    dX[i, j] += convolve2d(dout[i, k], np.rot90(self.W[k, j], 2), mode='full')

        self.W -= lr * dW / batch_size
        self.b -= lr * db

        return dX


# 池化层
class MaxPoolLayer:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self.mask = None
        self.original_input_shape = None
        self.output_shape = None

    def forward(self, X):
        self.original_input_shape = X.shape
        batch, channels, h, w = X.shape

        new_h = h - (h % self.pool_size)
        new_w = w - (w % self.pool_size)
        X = X[:, :, :new_h, :new_w]
        self.X = X

        out_h = new_h // self.pool_size
        out_w = new_w // self.pool_size

        X_reshaped = X.reshape(batch, channels, out_h, self.pool_size, out_w, self.pool_size)
        self.output = np.max(X_reshaped, axis=(3, 5))

        self.mask = np.zeros_like(X)
        argmax = np.argmax(X_reshaped.reshape(batch, channels, out_h, self.pool_size, out_w, self.pool_size), axis=3)
        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        idx = np.unravel_index(argmax[b, c, i, j], (self.pool_size, self.pool_size))
                        self.mask[b, c, i * self.pool_size + idx[0], j * self.pool_size + idx[1]] = 1

        self.output_shape = self.output.shape[1:]
        return self.output

    def backward(self, dout):
        dout = np.repeat(np.repeat(dout, self.pool_size, axis=2), self.pool_size, axis=3)

        dX = np.zeros(self.original_input_shape)
        valid_h = self.mask.shape[2]
        valid_w = self.mask.shape[3]

        dX[:, :, :valid_h, :valid_w] = dout * self.mask
        return dX


# 全连接层
class FCLayer:
    def __init__(self, input_size, output_size):
        scale = np.sqrt(2. / input_size)
        self.W = np.random.randn(input_size, output_size) * scale
        self.b = np.zeros(output_size)

    def forward(self, X):
        self.X = X
        return X.dot(self.W) + self.b

    def backward(self, dout, lr):
        dW = self.X.T.dot(dout) / self.X.shape[0]
        db = np.mean(dout, axis=0)
        dX = dout.dot(self.W.T)

        self.W -= lr * dW
        self.b -= lr * db

        return dX


# 卷积神经网络
class ConvNet:
    def __init__(self, input_shape=(3, 32, 32)):
        self.input_shape = input_shape

        self.conv1 = ConvLayer(3, 8, 3)
        self.pool1 = MaxPoolLayer(2)
        self.conv2 = ConvLayer(8, 16, 3)
        self.pool2 = MaxPoolLayer(2)

        test_input = np.random.randn(1, *input_shape)
        test_output = self.pool2.forward(
            self.conv2.forward(
                self.pool1.forward(
                    self.conv1.forward(test_input)
                )
            )
        )
        self.fc_input_size = np.prod(test_output.shape[1:])
        self.fc = FCLayer(self.fc_input_size, 2)

        print("\nNetwork architecture:")
        print(f"Input: {input_shape}")
        print(f"Conv1 -> Output: {self.conv1.output_shape}")
        print(f"Pool1 -> Output: {self.pool1.output_shape}")
        print(f"Conv2 -> Output: {self.conv2.output_shape}")
        print(f"Pool2 -> Output: {self.pool2.output_shape}")
        print(f"FC input size: {self.fc_input_size}")

    def forward(self, X):
        if X.shape[1:] != self.input_shape:
            raise ValueError(f"Expected input shape {self.input_shape}, got {X.shape[1:]}")

        out = self.conv1.forward(X)
        out = self.pool1.forward(out)
        out = self.conv2.forward(out)
        out = self.pool2.forward(out)
        out = out.reshape(out.shape[0], -1)
        return softmax(self.fc.forward(out))

    def backward(self, X, y, lr):
        output = self.forward(X)
        grad = (output - y) / X.shape[0]

        grad = self.fc.backward(grad, lr)
        grad = grad.reshape(X.shape[0], *self.pool2.output_shape)
        grad = self.pool2.backward(grad)

        if grad.shape != self.conv2.output.shape:
            grad = grad[:, :, :self.conv2.output.shape[2], :self.conv2.output.shape[3]]

        grad = self.conv2.backward(grad, lr)
        grad = self.pool1.backward(grad)
        self.conv1.backward(grad, lr)

        loss = -np.sum(y * np.log(output + 1e-10)) / X.shape[0]
        return loss

    def train(self, X, y, epochs=10, batch_size=32, lr=0.01):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                loss = self.backward(X_batch, y_batch, lr)
                epoch_loss += loss * X_batch.shape[0]

            losses.append(epoch_loss / len(X))
            print(f"Epoch {epoch + 1}/{epochs} Loss: {losses[-1]:.4f}")

        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        return losses


# 评估函数
def evaluate_model(model, X, y, set_name="Dataset"):
    output = model.forward(X)
    y_pred = np.argmax(output, axis=1)
    y_true = np.argmax(y, axis=1)

    accuracy = np.mean(y_pred == y_true)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    rmse = np.sqrt(np.mean((output - y) ** 2))
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{set_name} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rmse': rmse,
        'confusion_matrix': cm
    }


# 主程序
if __name__ == "__main__":
    try:
        # 配置参数
        DATA_DIR = r"C:\Users\20531\Desktop\DATASET"
        TARGET_SIZE = (32, 32)
        EPOCHS = 10
        BATCH_SIZE = 32
        LR = 0.01

        # 加载数据
        print("Loading data...")
        X_train, y_train = load_data(os.path.join(DATA_DIR, "train"), TARGET_SIZE)
        X_test, y_test = load_data(os.path.join(DATA_DIR, "test"), TARGET_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # 打印数据信息
        print("\nData shapes:")
        print(f"Train: {X_train.shape} {y_train.shape}")
        print(f"Val: {X_val.shape} {y_val.shape}")
        print(f"Test: {X_test.shape} {y_test.shape}")

        # 初始化模型
        print("\nInitializing model...")
        model = ConvNet(input_shape=(3, 32, 32))

        # 训练模型
        print("\nTraining...")
        model.train(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)

        # 评估模型
        print("\nEvaluating...")
        train_metrics = evaluate_model(model, X_train, y_train, "Training Set")
        val_metrics = evaluate_model(model, X_val, y_val, "Validation Set")
        test_metrics = evaluate_model(model, X_test, y_test, "Test Set")

        # 绘制指标对比
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        train_values = [train_metrics[m] for m in metrics]
        val_values = [val_metrics[m] for m in metrics]
        test_values = [test_metrics[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.25

        plt.figure(figsize=(10, 5))
        plt.bar(x - width, train_values, width, label='Train')
        plt.bar(x, val_values, width, label='Validation')
        plt.bar(x + width, test_values, width, label='Test')

        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback

        traceback.print_exc()
