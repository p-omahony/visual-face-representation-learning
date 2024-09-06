from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    # Model configuration
    config.model = ConfigDict()
    config.model.backbone = 'resnet18'
    config.model.pretrained = True  # Whether to use a pretrained model
    config.model.projector = ConfigDict()
    config.model.projector.active = False
    config.model.projector.embed_dim = 2048

    # Training configuration
    config.training = ConfigDict()
    config.training.batch_size = 64
    config.training.epochs = 3
    config.training.loss = ConfigDict()
    config.training.loss.name = 'triplet'
    config.training.loss.margin = 2.
    config.training.learning_rate = ConfigDict()
    config.training.learning_rate.start_value = 0.001
    config.training.learning_rate.reduce = True
    config.training.learning_rate.reduce_factor = 0.1
    config.training.learning_rate.reduce_patience = 4
    config.training.optimizer = 'adam'
    config.training.weight_decay = 1e-4

    # Dataset configuration
    config.dataset = ConfigDict()
    config.dataset.name = 'FaceScrub'
    config.dataset.train_split = 'train'
    config.dataset.image_size = 224  # ResNet typically uses 224x224 input size

    # Data augmentation (if needed)
    config.augmentation = ConfigDict()
    config.augmentation.random_crop = False
    config.augmentation.random_flip = 0.5
    config.augmentation.random_rotation = 15
    config.augmentation.gaussian_blur = False

    # Miscellaneous configurations
    config.seed = 42
    config.save_model = True

    return config

if __name__ == '__main__':
    cfg = get_config()
    if cfg.model.pretrained:
        
        print('cfg')