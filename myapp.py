import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf


with initialize(version_base=None, config_path="./config"):
    cfg = compose(config_name='config')
    # print(cfg)
print(cfg)




# logger = logging.getLogger(__name__)

# # conf = OmegaConf.load('config.yaml')
# # # print(conf.epochs)
# # @hydra.main(config_path="config", config_name="config.yaml")
# # def get_dataset(cfg: DictConfig):
# #     logger.info(cfg.exp.epochs)

# # if __name__ == "__main__":
# #     get_dataset()