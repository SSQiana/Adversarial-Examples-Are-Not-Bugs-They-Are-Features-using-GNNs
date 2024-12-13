import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.nn.pytorch import GraphConv


def norm(Z):
    return torch.norm(Z.view(Z.shape[0], -1), dim=1)


def single_pgd_step_adv(model, g, X, y, alpha, epsilon, delta):
    delta.requires_grad = True  # Enable gradient computation for delta

    # Forward pass with the perturbed input
    logits = model(g, X + delta)
    loss = torch.nn.functional.cross_entropy(logits, y, reduction='none')  # Per-node loss

    # Compute gradient of loss w.r.t. delta
    grad = torch.autograd.grad(loss.sum(), delta, retain_graph=True)[0]

    # Normalize the gradient
    grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True) + 1e-10
    grad_normalized = grad / grad_norm.view(-1, 1)

    # Update delta
    delta = delta + alpha * grad_normalized

    # Project delta back to the L2 ball
    delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True)
    delta = delta * torch.min(torch.ones_like(delta_norm), epsilon / (delta_norm + 1e-10)).view(-1, 1)

    return delta.detach(), loss.mean()


def pgd_l2_adv(model, g, X, y, alpha, num_iter, epsilon=0, example=False):
    delta = torch.zeros_like(X, device=X.device)  # Initialize perturbation as zeros
    loss = 0

    for t in range(num_iter):
        delta, loss = single_pgd_step_adv(model, g, X, y, alpha, epsilon, delta)

    if example:
        print(f'{num_iter} iterations, final loss: {loss:.4f}')

    return delta


def onestep_pgd_linf(model, g, X, y, epsilon, alpha, delta):
    delta.requires_grad = True

    # Forward pass with the perturbed input
    logits = model(g, X + delta)
    loss = F.cross_entropy(logits, y, reduction='none')  # Compute loss for each sample

    # Compute gradient of the loss w.r.t. delta
    grad = torch.autograd.grad(loss, delta, grad_outputs=torch.ones_like(loss), create_graph=True)[0]

    # Update delta using the sign of the gradient and alpha scaling
    delta = delta + alpha * grad.sign()

    # Clip delta to be within the L∞ ball defined by epsilon
    delta = torch.clamp(delta, -epsilon, epsilon)

    return delta.detach()


def pgd_linf(model, g, features, labels, epsilon, alpha, num_iter):
    delta = torch.zeros_like(features, device=features.device)  # Initialize perturbation as zeros
    for t in range(num_iter):
        delta = onestep_pgd_linf(model, g, features, labels, epsilon, alpha, delta)
    return delta



def fgsm_attack(model, g, X, y, epsilon=0.1):
    delta = torch.zeros_like(X, device=X.device, requires_grad=True)

    # Forward pass with the perturbed input
    logits = model(g, X + delta)
    loss = F.cross_entropy(logits, y, reduction='none')  # Compute loss for each sample

    # Compute gradient of the loss w.r.t. delta
    loss.mean().backward()  # Backpropagate to get the gradient w.r.t. delta

    # Get the sign of the gradient and apply perturbation
    delta = epsilon * delta.grad.sign()

    return delta


class Net(nn.Module):
    def __init__(self, in_feats, n_layers, n_hidden, n_classes, activation, dropout):
        super(Net, self).__init__()
        gnn = GraphConv(in_feats, n_hidden)
        gnn1 = GraphConv(n_hidden, n_hidden)
        gnn2 = GraphConv(n_hidden, n_classes)

        self.layers = nn.ModuleList()
        self.layers.append(gnn)

        for i in range(n_layers - 1):
            self.layers.append(gnn1)
        self.layers.append(gnn2)
        self.drop_out = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.drop_out(h)
            h = layer(g, h)
            if h.dim() >= 3:
                h = h.squeeze(dim=1)
        return h


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def load_cora_data():
    dataset = CoraGraphDataset()
    g = dataset[0]
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]
    return g, features, labels, train_mask, test_mask


def load_citeseer_data():
    dataset = CiteseerGraphDataset()
    g = dataset[0]
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]
    return g, features, labels, train_mask, test_mask


def load_Pubmed_data():
    dataset = PubmedGraphDataset()
    g = dataset[0]
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]
    return g, features, labels, train_mask, test_mask


# CiteseerGraphDataset



def standard_training(model, g, features, labels, train_mask, test_mask, epochs=5, verbose=True, device='cuda'):
    model.to(device)
    g = g.to(device)
    features = features.to(device)
    labels = labels.to(device)
    res = []
    best_acc = 0
    criterion = nn.CrossEntropyLoss()

    # 使用 Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(50):
        if epoch >= 3:
            t0 = time.time()
        model.train()
        logits = model(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = evaluate(model, g, features, labels, train_mask)
        test_acc = evaluate(model, g, features, labels, test_mask)
        res.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model.state_dict()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Test Accuracy: {test_acc:.4f}, Train Accuracy: {train_acc:.4f}, Best Test Accuracy: {best_acc:.4f}")

    return best_model


def adversarial_training(model, g, features, labels, train_mask, test_mask, attack, attack_params=None, device='cuda', **kwargs):
    model.to(device)
    g = g.to(device)
    features = features.to(device)
    labels = labels.to(device)
    res = []
    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(50):
        if epoch >= 3:
            t0 = time.time()
        model.train()
        if attack_params is not None:
            delta = attack(model, g, features, labels, **attack_params)
        else:
            delta = attack(model, g, features, labels, **kwargs)
        features[train_mask] = features[train_mask] + delta[train_mask]

        logits = model(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = evaluate(model, g, features, labels, train_mask)
        test_acc = evaluate(model, g, features, labels, test_mask)
        res.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model.state_dict()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Test Accuracy: {test_acc:.4f}, Train Accuracy: {train_acc:.4f}, Best Test Accuracy: {best_acc:.4f}")
    return best_model


def run_adversarial_attack(model, g, features, labels, test_mask, attack, attack_params=None, device='cuda', **kwargs):
    model.to(device)
    model.eval()
    g = g.to(device)
    features = features.to(device)
    labels = labels.to(device)

    # Start timing the attack process
    t = time.time()
    test_labels = labels[test_mask]

    if attack_params is not None:
        delta = attack(model, g, features, labels, **attack_params)
    else:
        delta = attack(model, g, features, labels, **kwargs)
    features[test_mask] = features[test_mask] + delta[test_mask]
    acc = evaluate(model, g, features, labels, test_mask)
    print(f"Time: {(time.time() - t):0.2f}s, Test Acc: {acc:0.2f}")


def single_pgd_step_robust(model, g, X, y, alpha, delta):
    delta.requires_grad = True
    X_adv = X + delta
    output = model(g, X_adv)
    loss = F.mse_loss(y, output, reduction='none').mean()

    loss.backward(retain_graph=True)
    grad = delta.grad
    normgrad = grad.norm(p=2, dim=-1, keepdim=True)
    grad_normalized = grad / (normgrad + 1e-10)
    delta = delta - alpha * grad_normalized
    delta = torch.clamp(delta, min=-X, max=1 - X)
    return delta.detach(), loss


def pgd_l2_robust(model, g, X, y, alpha, num_iter):
    delta = torch.zeros_like(X, requires_grad=True).to(X.device)  # Initialize perturbation with zeros
    loss = 0
    for t in range(num_iter):
        delta, loss = single_pgd_step_robust(model, g, X, y, alpha, delta)
        print('iter:', t, 'loss:', loss)
    return delta


def robustify(robust_mod, g, features, labels, iters=1000, alpha=0.1, batch_size=32):
    device = torch.device('cuda:0')
    g = g.to(device)
    features = features.to(device)
    robust_mod = robust_mod.to(device)
    labels = labels.to(device)
    # Get the goal representation
    goal_representation = robust_mod(g, features)
    # Update the batch of images using PGD
    learned_delta = pgd_l2_robust(robust_mod, g, features, goal_representation, alpha, num_iter=iters)
    robust_update = features + learned_delta
    return robust_update, labels


def single_pgd_step_nonrobust(model, g, X, y, alpha, epsilon, delta):
    delta.requires_grad = True
    X_adv = X + delta
    output = model(g, X_adv)  # 通过模型预测
    loss = F.cross_entropy(output, y, reduction='none').mean()  # 计算损失
    loss.backward(retain_graph=True)
    grad = delta.grad
    normgrad = grad.norm(p=2, dim=1, keepdim=True)
    grad_normalized = grad / (normgrad + 1e-10)
    z = delta - alpha * grad_normalized
    normz = z.norm(p=2, dim=1, keepdim=True)
    delta = epsilon * z / torch.clamp(normz, min=epsilon + 1e-10)
    return delta.detach(), loss


def pgd_l2_nonrobust(model, g, X, y, alpha, num_iter, epsilon=0.5):
    delta = torch.zeros_like(X, requires_grad=True).to(X.device)  # 初始化扰动
    loss = 0
    epsilon = 0.07
    for t in range(num_iter):
        delta, loss = single_pgd_step_nonrobust(model, g, X, y, alpha, epsilon, delta)
        print(f"Iteration {t + 1}/{num_iter}, Loss: {loss}")
    return delta


def non_robustify(robust_mod, g, features, labels, iters=1000, alpha=0.1, batch_size=32):
    device = torch.device('cuda:0')
    g = g.to(device)
    features = features.to(device)
    robust_mod = robust_mod.to(device)
    labels = labels.to(device)

    # Update the batch of images using PGD
    random_labels = torch.randint(0, max(labels).cpu().numpy().min(), (features.shape[0],)).to(device)
    learned_delta = pgd_l2_nonrobust(robust_mod, g, features, random_labels, alpha, num_iter=iters)

    non_robust_update = features + learned_delta
    return non_robust_update, labels
