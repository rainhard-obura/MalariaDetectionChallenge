from transformers import AutoModelForObjectDetection, AutoImageProcessor, AutoConfig
from transformers.models.auto.modeling_auto import AutoBackbone

from .constants import ID2LABEL, LABEL2ID


def get_model(cfg):
    config = AutoConfig.from_pretrained(cfg["model_name"])

    config.num_labels = len(ID2LABEL)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID

    # Update config with values from cfg (now a dict)
    for key, value in cfg.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Assume the model is always DETA
    # Properly set up the backbone configuration for DETA
    if isinstance(config.backbone_config, dict):
        backbone_config_dict = config.backbone_config
        backbone_config = AutoConfig.for_model(backbone_config_dict["model_type"])
        for key, value in backbone_config_dict.items():
            if key != "model_type":
                setattr(backbone_config, key, value)
        config.backbone_config = backbone_config

    model = AutoModelForObjectDetection.from_pretrained(
        cfg["model_name"],
        config=config,
        ignore_mismatched_sizes=True,
    )

    # Manually set up the backbone
    backbone = AutoBackbone.from_config(config.backbone_config)
    model.model.backbone.backbone = backbone

    return model


def get_processor(cfg):
    # Convert the ProcessorConfig to a dictionary, excluding 'model_name'
    processor_kwargs = {k: v for k, v in cfg.items() if k != "model_name"}

    # Create the processor
    processor = AutoImageProcessor.from_pretrained(
        cfg["model_name"], **processor_kwargs
    )
    if "width" in cfg and "height" in cfg:
        processor.size = {"width": cfg["width"], "height": cfg["height"]}

    return processor
