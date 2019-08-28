import os
import json
import torch
import pathlib
import datetime

import numpy as np
import pandas as pd
import torch.nn.functional as F

from pytorch_transformers import BertForSequenceClassification, BertTokenizer
from pytorch_transformers import XLNetForSequenceClassification, XLNetTokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule
from sklearn.metrics import f1_score, accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from tqdm import tqdm, trange

tqdm.pandas()

home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUBSEQ_LENGTH = 1380
VALID_SUBSEQ_LENGTH = 1380

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelConfiguration:
    def __init__(self):
        self.num_labels = 3
        self.learning_rate = 1e-5
        self.num_train_epochs = 4
        self.warmup_proportion = 0.4
        self.batch_size = 1
        self.full_mode = True

        # XLnet
        self.is_scibert = False
        self.is_xlnet = True
        self.bert_model = "xlnet-base-cased"
        self.do_lower = True

        self.train_sub_sampling = 1
        self.model_directory = None

        if self.bert_model in ("scibert_scivocab_cased", "scibert_scivocab_uncased",):
            self.vocab_file = \
                os.path.join("/home", "deepcompute", "Tense", "NER", "embeddings", self.bert_model, "vocab.txt")
            self.bert_model = os.path.join("/home", "deepcompute", "Tense", "NER", "embeddings", self.bert_model)  # , "weights.tar.gz")
            if not os.path.exists(self.vocab_file):
                print("File Not Found", self.vocab_file)
                raise FileNotFoundError


class DataPreProcessor:
    def __init__(self, tokenizer, is_xlnet=False, mask_entities=True):
        self.train = pd.read_csv(os.path.join(home, "data", "train.csv"))
        self.test = pd.read_csv(os.path.join(home, "data", "test.csv"))
        self.is_xlnet = is_xlnet
        self.tokenizer = tokenizer
        self.mask_entities = mask_entities

    @staticmethod
    def get_closest_180(tokens, label, sub_sequence_length):
        def get_all_locations(tokens, label):
            i = 0
            result = []
            while i < len(tokens):
                if tokens[i:i+len(label)] == label:
                    result.append((i, i+len(label)))
                    i += len(label)
                else:
                    i += 1
            return result

        def get_middle_most_180(postion, total_length):
            if isinstance(postion, list):
                start_position = (postion[0][0] + postion[1][0]) // 2
                end_position = (postion[0][1] + postion[1][1]) // 2
            else:
                start_position = postion[0]
                end_position = postion[1]

            final_start_position = (start_position + end_position)//2 - sub_sequence_length//2
            final_end_position = (start_position + end_position)//2 + sub_sequence_length//2

            if final_start_position < 0:
                final_start_position = 0
                final_end_position = sub_sequence_length
            elif final_end_position > total_length:
                final_start_position = total_length - sub_sequence_length
                final_end_position = total_length

            return final_start_position, final_end_position

        if len(tokens) <= sub_sequence_length:
            return tokens

        locations = get_all_locations(tokens=tokens, label=label)
        if len(locations) != 0 and len(locations) % 2 == 0:
            start, end = get_middle_most_180(
                postion=[locations[(len(locations) - 1) // 2], locations[len(locations) // 2]],
                total_length=len(tokens)
            )
        elif len(locations) != 0:
            start, end = get_middle_most_180(postion=locations[len(locations) // 2], total_length=len(tokens))
        else:
            start, end = 0, sub_sequence_length

        result = tokens[start:end]
        return result

    def simple_mask_replacement(self, df, sub_sequence_length):
        def mask_entities(text, label_name):
            replacement = text.replace(label_name, "[ENTITY1]")
            replacement = replacement.replace(label_name.upper(), "[ENTITY1]")
            replacement = replacement.replace(label_name.title(), "[ENTITY1]")
            return replacement

        df["replacement"] = df.progress_apply(lambda row: mask_entities(row["text"], row["drug"]), axis=1)
        df["Tokens"] = df["replacement"].progress_apply(self.tokenizer.tokenize)
        print(f"Larger than {sub_sequence_length}", (df["Tokens"].apply(len) > sub_sequence_length).sum(), df.shape[0])
        # df["Labels"] = df["drug"].progress_apply(self.tokenizer.tokenize)

        if not self.is_xlnet:
            df["FullTokens"] = df.progress_apply(
                lambda row: ["[CLS]"] + DataPreProcessor.get_closest_180(row["Tokens"], "[ENTITY1]",
                                                                         sub_sequence_length=sub_sequence_length) +
                            ["[SEP]"],
                axis=1
            )
        else:
            df["FullTokens"] = df.progress_apply(
                lambda row: DataPreProcessor.get_closest_180(row["Tokens"], "[ENTITY1]",
                                                             sub_sequence_length=sub_sequence_length) +
                            ["[SEP]", "[CLS]"],
                axis=1
            )
        df["TokenIds"] = df["FullTokens"].progress_apply(self.tokenizer.convert_tokens_to_ids)
        df["LabelIds"] = 2 if "sentiment" not in df else df["sentiment"]
        return df

    def get_split_data(self, train_df):
        build_hash_ids, valid_hash_ids = \
            train_test_split(train_df["unique_hash"], test_size=0.2, shuffle=True, random_state=RANDOM_SEED)
        build_df = train_df[train_df["unique_hash"].isin(build_hash_ids)]
        build_df.reset_index(inplace=True, drop=True)

        valid_df = train_df[train_df["unique_hash"].isin(valid_hash_ids)]
        valid_df.reset_index(inplace=True, drop=True)
        return build_df, valid_df

    def process(self):
        train_df = self.simple_mask_replacement(self.train, sub_sequence_length=SUBSEQ_LENGTH)
        test_df = self.simple_mask_replacement(self.test, sub_sequence_length=VALID_SUBSEQ_LENGTH)
        build_df, valid_df = self.get_split_data(train_df)

        return build_df, valid_df, train_df, test_df


def collate_ner(batch):
    token_ids = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
    segment_ids = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
    input_masks = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True, padding_value=0)
    label_ids = torch.LongTensor([item[3] for item in batch])

    return token_ids, segment_ids, input_masks, label_ids


class TaskDataSetSimple(Dataset):
    def __init__(self, file_name=None, data=None, split_set="valid", sub_sampling=1):
        super().__init__()
        self.split_set = split_set
        self.sub_sampling = sub_sampling

        if file_name is None and data is None:
            raise FileNotFoundError
        if file_name is not None:
            self.data = pd.read_hdf(file_name, key="h5")
        else:
            self.data = data
        self.data.reset_index(inplace=True, drop=True)

    def __getitem__(self, index_num):
        token_ids_ = self.data.loc[index_num, "TokenIds"]
        label_ids_ = self.data.loc[index_num, "LabelIds"]

        input_mask_ids_ = [1] * len(token_ids_)

        segment_ids_ = torch.LongTensor([0] * len(token_ids_))
        token_ids_ = torch.LongTensor(token_ids_)
        label_ids_ = label_ids_
        input_mask_ = torch.LongTensor(input_mask_ids_)

        return token_ids_, segment_ids_, input_mask_, label_ids_

    def __len__(self):
        return len(self.data) // self.sub_sampling


class DataGeneratorsSimpler:
    def __init__(self, build_data, valid_data, train_data, test_data, batch_size=32,
                 train_sub_sampling=1, valid_sub_sampling=1):
        self.build_data = build_data
        self.valid_data = valid_data
        self.train_data = train_data
        self.test_data = test_data

        self.batch_size = batch_size
        self.data_sets = None
        self.train_sub_sampling = train_sub_sampling
        self.valid_sub_sampling = valid_sub_sampling
        self.data_loader = self.create_data_loader()

    def get_generator(self):
        return self.data_loader

    def get_num_steps(self, num_train_epochs, division="build"):
        total_steps = 0
        num_train_examples = len(self.data_sets[division])
        num_train_steps = round(num_train_examples * num_train_epochs / self.batch_size)
        total_steps += num_train_steps
        return total_steps

    def create_data_loader(self):
        self.data_sets = {
            "train": TaskDataSetSimple(file_name=None, data=self.train_data, sub_sampling=self.train_sub_sampling),
            "build": TaskDataSetSimple(file_name=None, data=self.build_data, sub_sampling=self.train_sub_sampling),
            "valid": TaskDataSetSimple(file_name=None, data=self.valid_data, sub_sampling=self.valid_sub_sampling),
            "test": TaskDataSetSimple(file_name=None, data=self.test_data, sub_sampling=1)
        }

        samplers = {
            "train": RandomSampler(self.data_sets["train"]),
            "build": RandomSampler(self.data_sets["build"]),
            "valid": SequentialSampler(self.data_sets["valid"]),
            "test": SequentialSampler(self.data_sets["test"])
        }

        data_loaders = {
            split_set: DataLoader(data_set, sampler=samplers[split_set],
                                  batch_size=self.batch_size, num_workers=7, collate_fn=collate_ner)
            for split_set, data_set in self.data_sets.items()
        }

        return data_loaders


class Experiment:
    def __init__(self, model_directory=None, identifier=datetime.datetime.now().strftime('%d-%H-%M')):
        self.model_configuration = ModelConfiguration()

        self.model_name = f"{self.model_configuration.bert_model}_{RANDOM_SEED}_{SUBSEQ_LENGTH}_{identifier}"
        self.model_configuration.model_directory = model_directory
        if self.model_configuration.model_directory is None:
            self.model_configuration.model_directory = os.path.join(home, "models", self.model_name)

        self.tokenizer = None
        self.load_tokenizer()
        self.model = self.create_model()

    def load_tokenizer(self):
        if self.model_configuration.is_xlnet:
            self.tokenizer = XLNetTokenizer.from_pretrained(self.model_configuration.bert_model,
                                                            do_lower_case=self.model_configuration.do_lower)
        elif not self.model_configuration.is_scibert:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_configuration.bert_model,
                                                           do_lower_case=self.model_configuration.do_lower)
        else:
            self.tokenizer = BertTokenizer(self.model_configuration.vocab_file,
                                           do_lower_case=self.model_configuration.do_lower)

    def create_model(self):
        if self.model_configuration.bert_model in ("xlnet-base-cased",):
            model = XLNetForSequenceClassification.from_pretrained(self.model_configuration.bert_model,
                                                                   num_labels=self.model_configuration.num_labels)
        else:
            model = BertForSequenceClassification.from_pretrained(self.model_configuration.bert_model,
                                                                  num_labels=self.model_configuration.num_labels)
        model.to(device)
        return model

    def save_model_config(self):
        model_config = {
            "bert_model": self.model_configuration.bert_model,
            "do_lower": self.model_configuration.do_lower,
            "is_scibert": self.model_configuration.is_scibert,
            "is_xlnet": self.model_configuration.is_xlnet,
            "num_labels": self.model_configuration.num_labels
        }
        pathlib.Path(self.model_configuration.model_directory).mkdir(exist_ok=True, parents=True)

        with open(os.path.join(self.model_configuration.model_directory, "model_config.json"), "w") as f:
            json.dump(model_config, f)

    def train(self):
        pre_processor = DataPreProcessor(self.tokenizer, is_xlnet=self.model_configuration.is_xlnet)
        build_data, valid_data, train_data, test_data = pre_processor.process()

        data_generator = DataGeneratorsSimpler(train_data=train_data, build_data=build_data, valid_data=valid_data,
                                               test_data=test_data, batch_size=self.model_configuration.batch_size,
                                               train_sub_sampling=self.model_configuration.train_sub_sampling)
        self.model = self.create_model()

        data_loaders = data_generator.get_generator()
        if self.model_configuration.full_mode:
            num_train_steps = data_generator.get_num_steps(self.model_configuration.num_train_epochs, division="train")
        else:
            num_train_steps = data_generator.get_num_steps(self.model_configuration.num_train_epochs, division="build")

        num_warmup_steps = self.model_configuration.warmup_proportion * num_train_steps
        optimizer = AdamW(self.model.parameters(), lr=self.model_configuration.learning_rate, correct_bias=False)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps)

        self.save_model_config()

        if self.model_configuration.full_mode:
            data_loader_iterators = [
                data_loaders["train"]
            ]
            eval_data_loaders = []
        else:
            data_loader_iterators = [
                data_loaders["build"]
            ]
            eval_data_loaders = [
                data_loaders["valid"]
            ]
        test_data_loaders = [
            data_loaders["test"]
        ]
        for i in trange(int(self.model_configuration.num_train_epochs), desc="Epoch"):
            self.model.train()
            torch.cuda.synchronize()
            train_loss = 0

            for data_loader in tqdm(data_loader_iterators, desc="DataLoaders"):
                for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)

                    input_ids, segment_ids, input_mask, label_ids = batch
                    if self.model_configuration.is_xlnet:
                        loss = self.model(input_ids, labels=label_ids)
                    else:
                        loss = self.model(input_ids, labels=label_ids)

                    loss[0].backward()

                    train_loss += loss[0].item()
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()

            print("TrainLoss:", train_loss)
            losses = self.do_evaluate(eval_data_loaders, identifier=i)
            torch.cuda.synchronize()

            model_identifier = f"model_{i}_{losses['MicroF1']}_{losses['CrossEntropyLoss']}"
            self.save_model(model_identifier)

        self.create_submission(test_data_loaders)

        return self.model

    def save_model(self, model_identifier):
        pathlib.Path(self.model_configuration.model_directory).mkdir(exist_ok=True, parents=True)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_file = os.path.join(home, self.model_configuration.model_directory, f"{model_identifier}.pt")
        torch.save(model_to_save.state_dict(), model_file)
        output_config_file = os.path.join(self.model_configuration.model_directory, "config.json")
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    def do_evaluate(self, data_loaders, identifier=0):
        self.model.eval()
        y_true = []
        y_pred = []
        y_prob = None

        for data_loader in tqdm(data_loaders, desc="DataLoaders"):
            for input_ids, _, _, _, label_ids in tqdm(data_loader, desc="Evaluating"):
                input_ids = input_ids.to(device)

                with torch.no_grad():
                    if self.model_configuration.is_xlnet:
                        logits = self.model(input_ids)
                    else:
                        logits = self.model(input_ids)
                probabilities = F.softmax(logits[0], dim=-1)
                _, predictions = torch.max(probabilities, dim=-1)

                probabilities = probabilities.detach().cpu().numpy()
                predictions = predictions.detach().cpu().numpy()
                label_ids = label_ids.numpy()

                y_true.extend(label_ids)
                y_pred.extend(predictions)
                if y_prob is None:
                    y_prob = probabilities
                else:
                    y_prob = np.concatenate([probabilities, y_prob], axis=0)

        if len(data_loaders) > 0:
            primary_score = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
            secondary_score = log_loss(y_true=y_true, y_pred=y_prob)
            metadata = {
                "BertModel": self.model_configuration.bert_model,
                "LearningRate": self.model_configuration.learning_rate,
                "WarmupProportion": self.model_configuration.warmup_proportion,
                "Epoch": identifier,
                "CrossEntropyLoss": secondary_score,
                "Accuracy": accuracy_score(y_true, y_pred),
                "MicroF1": f1_score(y_true=y_true, y_pred=y_pred, average='micro'),
                "MacroF1": primary_score,
                "WeightedF1": f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
            }
            print(metadata)
            report = classification_report(y_true, y_pred, digits=4)
            output_eval_file = \
                os.path.join(self.model_configuration.model_directory, f"eval_results_{identifier}_{primary_score}.txt")
            with open(output_eval_file, "w") as writer:
                writer.write(json.dumps(metadata))
                writer.write("\n")
                writer.write(report)

            print(report)
            metadata["Report"] = report
        else:
            metadata = {
                "BertModel": self.model_configuration.bert_model,
                "LearningRate": self.model_configuration.learning_rate,
                "WarmupProportion": self.model_configuration.warmup_proportion,
                "Epoch": identifier,
                "CrossEntropyLoss": 0,
                "Accuracy": 0,
                "MicroF1": 0,
                "MacroF1": identifier/self.model_configuration.num_train_epochs,
                "WeightedF1": 0
            }
        return metadata

    def create_submission(self, data_loaders):
        self.model.eval()
        y_pred = []
        y_prob = None

        for data_loader in tqdm(data_loaders, desc="DataLoaders"):
            for input_ids, _, _, _ in tqdm(data_loader, desc="Evaluating"):
                input_ids = input_ids.to(device)

                with torch.no_grad():
                    if self.model_configuration.is_xlnet:
                        logits = self.model(input_ids)
                    else:
                        logits = self.model(input_ids)
                probabilities = F.softmax(logits[0], dim=-1)
                _, predictions = torch.max(probabilities, dim=-1)

                probabilities = probabilities.detach().cpu().numpy()
                predictions = predictions.detach().cpu().numpy()

                y_pred.extend(predictions)
                if y_prob is None:
                    y_prob = probabilities
                else:
                    y_prob = np.concatenate([y_prob, probabilities], axis=0)

        result = pd.read_csv(os.path.join(home, "data", "test.csv"), usecols=["unique_hash"])
        for i in range(3):
            result[f"{RANDOM_SEED}_{i}"] = y_prob[:, i]

        results_dir = os.path.join(home, "results")
        pathlib.Path(results_dir).mkdir(exist_ok=True, parents=True)
        result.to_csv(os.path.join(results_dir, f"{self.model_name}.csv"), index=False)


if __name__ == "__main__":
    for seed in [21, 22, 23, 24, 25, 26]:
        RANDOM_SEED = seed
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        model1 = Experiment(identifier="seed").train()
