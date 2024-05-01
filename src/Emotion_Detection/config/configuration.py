from src.Emotion_Detection.constants import *
from src.Emotion_Detection.utils.common import read_yaml,create_directories
from src.Emotion_Detection.entity.config_entity import DataIngestionConfig,PrepareBaseModelConfig
from src.Emotion_Detection.constants import *
from src.Emotion_Detection.utils.common import read_yaml, create_directories
# Defining Paths basically we are mergging our constants from config.yaml with entity dataingestion entity
class ConfigurationManager:
    def __init__( self,config_filepath = CONFIG_FILE_PATH,params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            push_URL=config.push_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )
# it will return all the paths 
        return data_ingestion_config   
      

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )
        return prepare_base_model_config