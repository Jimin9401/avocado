import json
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class CFExample(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 text,
                 y,
                 ):
        self.idx = idx
        self.text = text
        self.y = y


class ContrastiveExample(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 domain_text,
                 origin_text,
                 y):
        self.idx = idx
        self.domain_text = domain_text
        self.origin_text = origin_text
        self.y = y



class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


