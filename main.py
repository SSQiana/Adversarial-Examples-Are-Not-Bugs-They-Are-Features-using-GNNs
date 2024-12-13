import torch
import torch.nn as nn
from utils import Net, standard_training, adversarial_training, run_adversarial_attack, robustify, non_robustify
from utils import pgd_l2_adv
from utils import load_cora_data

# Load data
g, features, labels, train_mask, test_mask = load_cora_data()

# Model parameters
n_layers = 3
n_hidden = 512
in_feats = features.shape[1]
n_classes = 7
activation = nn.PReLU(n_hidden)
dropout = 0.1

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Standard training
print('Standard training on the origin dataset')
model = Net(in_feats, n_layers, n_hidden, n_classes, activation, dropout)
model_path_standard = 'standard_training.pth'
best_model = standard_training(model, g, features, labels, train_mask, test_mask, epochs=50)
torch.save(best_model, model_path_standard)

# Adversarial training
print('Adversarial training on the origin dataset')
adversarial_model = Net(in_feats, n_layers, n_hidden, n_classes, activation, dropout)
adversarial_model_path = 'adversarial_training.pth'
adversarial_trained_model = adversarial_training(
    adversarial_model,
    g=g,
    features=features,
    labels=labels,
    train_mask=train_mask,
    test_mask=test_mask,
    attack=pgd_l2_adv,
    attack_params={'epsilon': 0.001, 'num_iter': 5, 'alpha': 0.5 / 5},
    device=device
)
torch.save(adversarial_trained_model, adversarial_model_path)

# Adversarial attack on standard model
print('Adversarial attack on the standard training model')
model.load_state_dict(torch.load(model_path_standard))
run_adversarial_attack(
    model=model,
    g=g,
    features=features,
    labels=labels,
    test_mask=test_mask,
    attack=pgd_l2_adv,
    attack_params={'epsilon': 0.01, 'num_iter': 7, 'alpha': 0.5 / 5},
    device=device
)

# Adversarial attack on adversarially trained model
print('Adversarial attack on the adversarial training model')
model.load_state_dict(torch.load(adversarial_model_path))
run_adversarial_attack(
    model=model,
    g=g,
    features=features,
    labels=labels,
    test_mask=test_mask,
    attack=pgd_l2_adv,
    attack_params={'epsilon': 0.01, 'num_iter': 7, 'alpha': 0.5 / 5},
    device=device
)

# Robust dataset generation
print('Generating robust dataset')
robust_model = Net(in_feats, n_layers, n_hidden, n_classes, activation, dropout)
robust_model.load_state_dict(torch.load(adversarial_model_path))
robust_train, _ = robustify(robust_model, g, features, labels, iters=200, alpha=0.1)
torch.save(robust_train, 'robust_train.pt')

# Train on robust dataset
print('Training robust model')
robust_train = torch.load('robust_train.pt')
robust_model_path = 'robust_model.pth'
robust_trained_model = standard_training(robust_model, g, robust_train, labels, train_mask, test_mask, epochs=50)
torch.save(robust_trained_model, robust_model_path)

# Adversarial attack on robust model
print('Adversarial attack on the robust model')
robust_model.load_state_dict(torch.load(robust_model_path))
run_adversarial_attack(
    model=robust_model,
    g=g,
    features=robust_train,
    labels=labels,
    test_mask=test_mask,
    attack=pgd_l2_adv,
    attack_params={'epsilon': 0.01, 'num_iter': 7, 'alpha': 0.5 / 5},
    device=device
)

# Generate non-robust dataset
print('Generating non-robust dataset')
model.load_state_dict(torch.load(model_path_standard))
non_robust_train, _ = non_robustify(model, g, features, labels, iters=200, alpha=0.1, batch_size=32)
torch.save(non_robust_train, 'non_robust_train.pt')

# Train on non-robust dataset
print('Training non-robust model')
non_robust_train = torch.load('non_robust_train.pt')
non_robust_model = Net(in_feats, n_layers, n_hidden, n_classes, activation, dropout)
non_robust_model_path = 'non_robust_model.pth'
non_robust_trained_model = standard_training(non_robust_model, g, non_robust_train, labels, train_mask, test_mask, epochs=50)
torch.save(non_robust_trained_model, non_robust_model_path)

# Adversarial attack on non-robust model
print('Adversarial attack on the non-robust model')
non_robust_model.load_state_dict(torch.load(non_robust_model_path))
run_adversarial_attack(
    model=non_robust_model,
    g=g,
    features=features,
    labels=labels,
    test_mask=test_mask,
    attack=pgd_l2_adv,
    attack_params={'epsilon': 0.01, 'num_iter': 7, 'alpha': 0.5 / 5},
    device=device
)
