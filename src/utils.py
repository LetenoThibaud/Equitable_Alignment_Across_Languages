from unsloth import FastLanguageModel
from trl import DPOConfig, DPOTrainer
import torch
import pickle
import sys
from typing import Union, Any
import os
import glob
import pandas as pd
from transformers.data.data_collator import DataCollatorMixin
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
import random
import numpy as np
from PPLTrainer import DPOPPLTrainer
from trl.trainer.utils import pad

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def concat(example):
    example["prompt"] = example["norm"] + " " + example["situation"] + " " + example["intention"]
    return example

def concat_jigsaw(example):
    example['prompt'] = "Input: "
    return example

def load_h_data(args):
    if args.dataset_name == 'moral':
        en = pd.read_pickle('./data/train_en.pkl')
        fr = pd.read_pickle('./data/train_fr.pkl')

        n_ratio_en = int(8400*args.en_ratio)
        en_sample = en.sample(n=n_ratio_en, random_state=42)
        fr_sample = fr[~fr['guid'].isin(en_sample['guid'])]

        en_sample['language'] = 0
        fr_sample['language'] = 1

        en = Dataset.from_pandas(en_sample)
        fr = Dataset.from_pandas(fr_sample)
        dataset = concatenate_datasets([en, fr]).shuffle(seed=args.seed)

        dataset = dataset.map(concat)
        if args.align_to_moral:
            dataset = dataset.rename_column("moral_action", "chosen")
            dataset = dataset.rename_column("immoral_action", "rejected")
        else:
            dataset = dataset.rename_column("immoral_action", "chosen")
            dataset = dataset.rename_column("moral_action", "rejected")

        dataset = dataset.remove_columns(
            ['guid', '__index_level_0__', 'norm', 'situation', 'intention', 'moral_consequence', 'immoral_consequence'])

        return dataset
    elif args.dataset_name == 'unimoral':
        if args.languages == "en-ru":
            en = pd.read_csv('./data/train_en.csv')
            fr = pd.read_csv('./data/train_ru.csv')
        elif args.languages == "en-es":
            en = pd.read_csv('./data/train_en.csv')
            fr = pd.read_csv('./data/train_sp.csv')

        n_ratio_en = int(len(en)*args.en_ratio)
        en_sample = en.sample(n=n_ratio_en, random_state=42)
        fr_sample = fr[~fr['Scenario_id'].isin(en_sample['Scenario_id'])]

        en_sample['language'] = 0
        fr_sample['language'] = 1

        en = Dataset.from_pandas(en_sample)
        fr = Dataset.from_pandas(fr_sample)
        dataset = concatenate_datasets([en, fr]).shuffle(seed=args.seed)

        dataset = dataset.remove_columns(['Scenario_id'])
        return dataset
    elif args.dataset_name == 'jigsaw':
        if args.languages == "en-fr":
            en = pd.read_csv('./data/jigsaw/preference_toxicity_en_train.csv')
            fr = pd.read_csv('./data/jigsaw/preference_toxicity_fr_train.csv')
        elif args.languages == "en-ru":
            en = pd.read_csv('./data/jigsaw/preference_toxicity_en_train.csv')
            fr = pd.read_csv('./data/jigsaw/preference_toxicity_ru_train.csv')
        elif args.languages == "en-es":
            en = pd.read_csv('./data/jigsaw/preference_toxicity_en_train.csv')
            fr = pd.read_csv('./data/jigsaw/preference_toxicity_es_train.csv')

        en['language'] = 0
        fr['language'] = 1

        en_dataset = Dataset.from_pandas(en)
        fr_dataset = Dataset.from_pandas(fr)

        interleaved_data = []
        for en_row, fr_row in zip(en_dataset, fr_dataset):
            interleaved_data.append(en_row)
            interleaved_data.append(fr_row)

        dataset = Dataset.from_list(interleaved_data)
        dataset = dataset.map(concat_jigsaw)

        return dataset
    else:
        print('Dataset name must be jigsaw or moral')
        sys.exit(1)

def load_data_test(args):
    if args.dataset_name == 'moral':
        en = pd.read_pickle('./data/moral/test_en.pkl')
        fr = pd.read_pickle('./data/moral/test_fr.pkl')

        en = Dataset.from_pandas(en)
        fr = Dataset.from_pandas(fr)

        en = en.map(concat)
        fr = fr.map(concat)

        en = en.rename_column("moral_action", "chosen")
        en = en.rename_column("immoral_action", "rejected")

        fr = fr.rename_column("moral_action", "chosen")
        fr = fr.rename_column("immoral_action", "rejected")

        en = en.remove_columns(['norm', 'situation', 'intention', 'moral_consequence', 'immoral_consequence'])
        fr = fr.remove_columns(['norm', 'situation', 'intention', 'moral_consequence', 'immoral_consequence'])

        return en, fr

    elif args.dataset_name == 'unimoral':
        if args.languages == "en-ru":
            en = pd.read_csv('./data/unimoral/test_en.csv')
            fr = pd.read_csv('./data/unimoral/test_ru.csv')
        elif args.languages == "en-es":
            en = pd.read_csv('./data/unimoral/test_en.csv')
            fr = pd.read_csv('./data/unimoral/test_sp.csv')
        else:
            print('Languages must be en-es or en-ru')
            sys.exit(1)
        en = Dataset.from_pandas(en)
        fr = Dataset.from_pandas(fr)

        en = en.remove_columns(['Scenario_id'])
        fr = fr.remove_columns(['Scenario_id'])
        return en, fr
    elif args.dataset_name == 'jigsaw':
        if args.languages == "en-fr":
            en = pd.read_csv('./data/jigsaw/preference_toxicity_en_test.csv')
            fr = pd.read_csv('./data/jigsaw/preference_toxicity_fr_test.csv')
        elif args.languages == "en-ru":
            en = pd.read_csv('./data/jigsaw/preference_toxicity_en_test.csv')
            fr = pd.read_csv('./data/jigsaw/preference_toxicity_ru_test.csv')
        elif args.languages == "en-es":
            en = pd.read_csv('./data/jigsaw/preference_toxicity_en_test.csv')
            fr = pd.read_csv('./data/jigsaw/preference_toxicity_es_test.csv')
        else:
            print('Languages must be en-fr, en-es or en-ru')
            sys.exit(1)

        en_dataset = Dataset.from_pandas(en)
        fr_dataset = Dataset.from_pandas(fr)

        en_dataset = en_dataset.map(concat_jigsaw)
        fr_dataset = fr_dataset.map(concat_jigsaw)

        return en_dataset, fr_dataset
    else:
        print('Dataset name must be jigsaw or moral')
        sys.exit(1)


class LanguageDataCollatorForPreference(DataCollatorMixin):
    """
    Data collator used for preference data. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length. A special label is added for the language.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.
    """
    def __init__(self, pad_token_id=0, return_tensors="pt"):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.return_tensors = return_tensors

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]
        if "pixel_values" in examples[0]:
            pixel_values = [torch.tensor(example["pixel_values"]) for example in examples]
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [torch.tensor(example["pixel_attention_mask"]) for example in examples]
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            ref_chosen_logps = torch.tensor([example["ref_chosen_logps"] for example in examples])
            ref_rejected_logps = torch.tensor([example["ref_rejected_logps"] for example in examples])

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)
        if "pixel_values" in examples[0]:
            output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        if "pixel_attention_mask" in examples[0]:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor([example["image_sizes"] for example in examples])
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps

        output["language"] = [torch.tensor(example["language"]) for example in examples]

        return output


def qlora_training_reg_gdist(args, dataset, path = None):
    max_seq_length = 2048

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=args.hf_token
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )

    training_args = DPOConfig(
        output_dir="./output",
        beta=args.beta,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        loss_type=args.loss_type,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.n_epoch,
        logging_dir="./logs/" + args.log_id + path,
        logging_steps=1,
        logging_strategy="steps",
        report_to="tensorboard",
        evaluation_strategy="no",
        do_eval=False,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
    )

    gamma_step = args.gamma_step

    dpo_trainer = GDistDPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        data_collator=LanguageDataCollatorForPreference(pad_token_id=tokenizer.pad_token_id),
        train_dataset=dataset.shard(num_shards=int(8400 / args.nb_examples), index=0),
        processing_class=tokenizer,
        gamma=args.gamma,
        gamma_step=gamma_step,
    )
    dpo_trainer.train()

    log_dir = "./logs/" + args.log_id + path
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if event_files:
        os.rename(event_files[0], os.path.join(log_dir, "tensorboard_log.tfevents"))

    return dpo_trainer.model, tokenizer

def evaluate_model(model, tokenizer, dataset, args, name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_en = dataset['en']
    test_fr = dataset['X']

    names = ["en", "X"]
    result = {}
    for i, data in enumerate([test_en, test_fr]):
        count_moral = 0
        ppl_moral, ppl_immoral = [], []

        for dat in tqdm(data):
            if args.dataset_name == 'moral':
                input_all = tokenizer(dat["prompt"], return_tensors="pt")
                input = tokenizer(dat["chosen"], return_tensors="pt")
                input["labels"] = torch.hstack([torch.full_like(input_all["input_ids"], -100), input["input_ids"]])
                input["input_ids"] = torch.hstack([input_all["input_ids"], input["input_ids"]])
                input["attention_mask"] = torch.hstack([input_all["attention_mask"], input["attention_mask"]])
                input.to(device)
                output = model(**input)
                loss_chosen = output.loss.item()
                ppl_moral.append(loss_chosen)

                input = tokenizer(dat["rejected"], return_tensors="pt")
                input["labels"] = torch.hstack([torch.full_like(input_all["input_ids"], -100), input["input_ids"]])
                input["input_ids"] = torch.hstack([input_all["input_ids"], input["input_ids"]])
                input["attention_mask"] = torch.hstack([input_all["attention_mask"], input["attention_mask"]])
                input.to(device)
                output = model(**input)
                loss_rejected = output.loss.item()
                ppl_immoral.append(loss_rejected)

            else:
                input = tokenizer(dat["chosen"], return_tensors="pt", max_length=2048, truncation=True, add_special_tokens=False)
                input["labels"] = input["input_ids"]
                input.to(device)
                output = model(**input)
                loss_chosen = output.loss.item()
                ppl_moral.append(loss_chosen)

                input = tokenizer(dat["rejected"], return_tensors="pt", max_length=2048, truncation=True, add_special_tokens=False)
                input["labels"] = input["input_ids"]
                input.to(device)
                output = model(**input)
                loss_rejected = output.loss.item()
                ppl_immoral.append(loss_rejected)

            if loss_chosen < loss_rejected:
                count_moral += 1

        count_immoral_preferred, count_moral_preferred = 0, 0
        for a, b in zip(ppl_moral, ppl_immoral):
            if a > b:
                count_immoral_preferred += 1
            elif b > a:
                count_moral_preferred += 1

        print(count_moral / len(dataset))

        print("Model:", args.model_name)
        print("Dataset:", names[i])
        print("=" * 100)
        print('Count moral preferred | immoral preferred :', count_moral_preferred, ":", count_immoral_preferred)
        print('Average perplexity moral:', round(torch.mean(torch.tensor(ppl_moral)).item(), 2), "~",
              round(torch.std(torch.tensor(ppl_moral)).item(), 2))
        print('Average perplexity immoral:', round(torch.mean(torch.tensor(ppl_immoral)).item(), 2), "~",
              round(torch.std(torch.tensor(ppl_immoral)).item(), 2))
        print('Percentage moral preferred', count_moral / len(data))
        print("=" * 100)

        result[names[i]] = {'model': args.model_name,
                  'count_moral': count_moral_preferred,
                  'count_immoral': count_immoral_preferred,
                  'avg_ppl_moral': round(torch.mean(torch.tensor(ppl_moral)).item(), 2),
                  'std_ppl_moral': round(torch.std(torch.tensor(ppl_moral)).item(), 2),
                  'avg_ppl_immoral': round(torch.mean(torch.tensor(ppl_immoral)).item(), 2),
                  'std_ppl_immoral': round(torch.std(torch.tensor(ppl_immoral)).item(), 2),
                  'preference_rate': round(count_moral / len(data) * 100, 2)
                  }

    result_path = 'results/' + name + '.pickle'

    if not os.path.exists('results'):
        os.makedirs('results')

    with open(result_path, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

