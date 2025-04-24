import yaml
from omegaconf import OmegaConf
from yt_tools.utils import instantiate_from_config


def create_dataloader(dataloader_config_path: str, batch_size: int, skip_rows=0):
    with open(dataloader_config_path) as f:
        dataloader_config = OmegaConf.create(yaml.load(f, Loader=yaml.SafeLoader))
    # Set batch size
    dataloader_config["params"]["batch_size"] = batch_size
    return instantiate_from_config(dataloader_config, skip_rows=skip_rows)
