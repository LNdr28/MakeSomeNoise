from dataclasses import dataclass, asdict

import wandb
import dotenv
from enum import Enum

class Dataset(str, Enum):
    """Enum for the different datasets. Used to track dataset in wandb"""
    WIKIPEDIA2018_NQ_OPEN = "wikipedia2018_nq_open" # wikipedia dump from 2018 injected with golden documents from NQ

class Retriever(str, Enum):
    """Enum for the different retrievers. Used to track retriever in wandb"""
    BM25 = "bm25"
    CONTRIVER = "contriever"
    ADORE = "adore"


# For now use llm-id via huggingface model path
#class ModelLLM(str, Enum):
#    """Enum for the different LLMs. Used to track LLM in wandb"""
#    LLAMA_2_7B_CHAT = "llama-2-7b-chat"
#    PHI_2_0 = "phi-2"
#    MPT_7B_INSTRUCT = "mpt-7b-instruct"
#    FALCON_7B_INSTRUCT = "falcon-7b-instruct"


class JobType(str, Enum):
    """Enum for the different wandb job types."""
    TRAINING = "training"  # training our retriever/reranker
    BENCHMARK = "benchmark"  # benchmarking llm responses as described in README


class Tag(str, Enum):
    """Enum for wandb run tags"""
    CLOSED_BOOK = "closed_book"
    OPEN_BOOK = "open_book"
    FINE_TUNE = "fine_tune"
    FULL_TRAIN = "full_train"


@dataclass
class RunConfig:
    """Configuration to set key value pairs for wandb run. Needed to allow config tracking besides tags and job_type"""
    dataset: Dataset
    dataset_use_test_set: bool  # both options here in case we want to use both
    dataset_use_train_set: bool
    prompt_type: str # no enum needed as script (ex. read_generation_results.py) check checks strings themselves
    model_llm: str  # is a huggingface model id as string, thus strict/consistent format kind of guaranteed
    model_retriever: Retriever
    seed: str  # random seed

    # TODO: add benchmark confs:  gold doc position, num docs, use random irrelevant docs, etc....

    def to_dict(self):
        kwargs = asdict(self)
        kwargs['dataset'] = kwargs['dataset'].value
        kwargs['model_retriever'] = kwargs['model_retriever'].value
        return kwargs


def init_wandb(run_name: str, job_type: JobType, run_config: RunConfig, tags=None, notes: str = ""):
    """
    Initialize the wandb library for logging a specific run.
    It reads the wandb configuration from env and sets the config as well as the experiment name.
    """
    if tags is None:
        tags = []
    tags = [tag.value if hasattr(tag, 'value') else tag for tag in tags]

    if hasattr(job_type, 'value'):
        job_type = job_type.value

    wandb.init(
        # Project and entity are set via env vars
        name=run_name,
        job_type=job_type,
        tags=tags,
        notes=notes,
        config=run_config.to_dict()
    )


def load_env(env_file: str = ".env"):
    """
    Loads env file via python-dotenv. It is used to load the wandb project, the entity and api key for logging.
    Wandb will automatically use the env variables set (api-key, project and entity), no setup needed!
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)
