import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import torch
import numpy as np
from matplotlib import pyplot as plt;
import time;

train_df = pd.read_csv("../new_train.tsv",sep='\t',header=None, names=['C1', 'C2'])
test_df = pd.read_csv("../new_test.tsv",sep = '\t',header=None, names=['C1', 'C2'])

x = train_df['C1'].astype(str)
y = train_df['C2']

x_train,x_val,y_train,y_val = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

x_test = test_df["C1"]
y_test = test_df["C2"]

def Tfidf():
    tfidf = TfidfVectorizer(
        max_features=10000,
    )
    X_train_vec = tfidf.fit_transform(x_train) #根据统计学生成词的向量，依据词典大小确定
    X_val_vec = tfidf.transform(x_val)
    X_test_vec = tfidf.transform(x_test)
    return X_train_vec,X_val_vec,X_test_vec;

def ngram():
    ngram_vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)  # 1-gram和2-gram
    )
    X_train_vec = ngram_vectorizer.fit_transform(x_train)
    X_val_vec = ngram_vectorizer.transform(x_val)
    X_test_vec = ngram_vectorizer.transform(x_test)
    return X_train_vec, X_val_vec, X_test_vec;

def wordBow():
    bow_vectorizer = CountVectorizer(
        max_features=10000,
    )
    X_train_vec = bow_vectorizer.fit_transform(x_train)
    X_val_vec = bow_vectorizer.transform(x_val)
    X_test_vec = bow_vectorizer.transform(x_test)
    return X_train_vec, X_val_vec, X_test_vec;

mode = [Tfidf,ngram,wordBow];
name = ["Tfidf","N-Gram","WordBow"];

def to_tensor(matrix):
    return torch.FloatTensor(matrix.toarray());

def cross_entropy(probs,target):
    correct_log_probs = -torch.log(probs.gather(1,target.view(-1,1))+1e-10); #计算交叉熵损失，计算的时候只取正确标签的索引
    return torch.mean(correct_log_probs); #一批当中取平均

def get_accuracy(probs,target):
    predictions = torch.argmax(probs,dim=1);#找概率最大的位置的索引
    return (predictions==target).float().mean().item();

def train(lr=1.0, epochs=50, batch_size=64,mode_name=""):
    W = torch.randn(num_features, num_classes) * (1.0 / np.sqrt(num_features));
    W.requires_grad_(True)
    b = torch.zeros(num_classes, requires_grad=True)

    def model(X):
        # 前向传播，计算Softmax
        logits = torch.matmul(X, W) + b;
        exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0])
        return exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
    ep=[];
    ls=[];
    ac = [];
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_t.size(0));
        for i in range(0,X_train_t.size(0),batch_size):
            indices = permutation[i:i+batch_size];
            batch_X,batch_y = X_train_t[indices],y_train_t[indices];
            probs = model(batch_X);
            loss = cross_entropy(probs,batch_y);
            loss.backward();
            with torch.no_grad(): #关闭梯度计算过程
                W -= lr*W.grad;
                b -= lr*b.grad;
                W.grad.zero_()
                b.grad.zero_()
        if (epoch + 1)%5 == 0:
            with torch.no_grad():
                val_probs = model(X_val_t);
                val_acc = get_accuracy(val_probs,y_val_t);
                ep.append(epoch+1);
                ls.append(loss.item());
                ac.append(val_acc);
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}, Val Acc: {val_acc:.4%}")

    plt.plot(ep, ls, marker='o', linestyle='-', color='b')
    plt.xlabel('epoch')  # X轴说明
    plt.ylabel('loss')  # Y轴说明

    # 4. 添加图表标题
    plt.title(f'batchSize_{batch_size}')


    # 7. 展示图表
    plt.savefig(f"graph/{mode_name}_loss_lr={lr}_epochs={epochs}_batchSize={batch_size}.png")
    plt.clf()

    plt.plot(ep, ac, marker='o', linestyle='-', color='b')
    plt.xlabel('epoch')  # X轴说明
    plt.ylabel('accuracy')  # Y轴说明

    # 4. 添加图表标题
    plt.title(f'batch_size_{batch_size}')

    # 7. 展示图表
    plt.savefig(f"graph/{mode_name}_accuracy_lr={lr}_epochs={epochs}_batchSize={batch_size}.png")
    plt.clf()
    with torch.no_grad():
        test_probs = model(X_test_t);
        test_acc = get_accuracy(test_probs, y_test_t);
        return test_acc;
start = time.time();

for l in range(len(mode)):
    X_train_vec, X_val_vec, X_test_vec = mode[l]();

    X_train_t = to_tensor(X_train_vec);
    y_train_t = torch.LongTensor(y_train.values);

    X_val_t = to_tensor(X_val_vec);
    y_val_t = torch.LongTensor(y_val.values);

    X_test_t = to_tensor(X_test_vec);
    y_test_t = torch.LongTensor(y_test.values);

    num_features = X_train_t.shape[1]
    num_classes = len(y_train.unique());
    test_acc = train(lr=0.1,epochs=100,batch_size=64,mode_name=name[l]);
    print(f"mode:{name[l]}, accuracy:{test_acc}")
end = time.time();

# 0.44363856315612793
# 0.4466606080532074
# 0.4780900478363037

