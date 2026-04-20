from awm.tools import tools_jsonl_load, tools_jsonl_save
import random

random.seed(12345)
num_samples = 10

scenes = tools_jsonl_load('outputs/seed_scenario.jsonl')
random.shuffle(scenes)
tools_jsonl_save(scenes[:num_samples], 'outputs/seed_scenario10.jsonl')
