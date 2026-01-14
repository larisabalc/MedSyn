from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from diagnosis_engine.csv_logger_callback import CSVLoggerCallback
import torch
import os
import pandas as pd
import evaluate
import numpy as np

class ContextDiagnosisClassifier:
    def __init__(self, model_name="t5-small", dataset_path=None):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        self.dataset_path = dataset_path
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def load_local_dataset(self):
        """Loads a local CSV dataset with columns: input_text, target"""
        df = pd.read_csv(self.dataset_path)
        self.dataset = Dataset.from_pandas(df)

    def preprocess_data(self, examples):
        """
        Tokenizes a batch of examples.
        Assumes examples is a dictionary with keys: "input_text" and "target"
        """
        inputs = [str(x) for x in examples["input_text"]]
        targets = [str(x) for x in examples["target"]]

        model_inputs = self.tokenizer(
            inputs,
            max_length=256,
            truncation=True,
            padding="max_length"
        )

        labels = self.tokenizer(
            targets,
            max_length=32,
            truncation=True,
            padding="max_length"
        )["input_ids"]

        labels = [[l if l != self.tokenizer.pad_token_id else -100 for l in seq] for seq in labels]
        model_inputs["labels"] = labels

        return model_inputs

    def prepare_dataset(self, test_size=0.2):
        """Tokenizes the dataset and splits into train/test"""
        if not isinstance(self.dataset, Dataset):
            raise ValueError("Dataset not loaded. Call load_local_dataset() first.")

        tokenized = self.dataset.map(
            self.preprocess_data,
            batched=True,
            remove_columns=self.dataset.column_names
        )

        split = tokenized.train_test_split(test_size=test_size, seed=42)
        self.train_dataset = split["train"]
        self.test_dataset = split["test"]

    def train(self, num_train_epochs=5):
        """Trains the T5 model"""
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        csv_logger = CSVLoggerCallback(train_log_file="diagnosis_engine/trained_models/context/metrics/train_context_log.csv", eval_log_file="diagnosis_engine/trained_models/context/metrics/eval__context_log.csv")

        try:
            training_args = Seq2SeqTrainingArguments(
                output_dir="diagnosis_engine/trained_models/context",
                eval_strategy="epoch",
                logging_strategy="steps",
                logging_steps=10,  
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=num_train_epochs,
                save_total_limit=2,
                predict_with_generate=True,
                fp16=torch.cuda.is_available(),
                remove_unused_columns=False,
                report_to="none",
            )
        except TypeError:
            training_args = Seq2SeqTrainingArguments(
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=num_train_epochs,
                save_total_limit=2,
                predict_with_generate=True,
                fp16=torch.cuda.is_available(),
                remove_unused_columns=False,
                report_to="none",
            )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[csv_logger]
        )

        trainer.train()

    def evaluate(self, compute_metrics=True):
        """Evaluates the model on the test set and computes final metrics (BLEU, ROUGE, Exact Match)."""
        if self.test_dataset is None:
            raise ValueError("Test dataset not prepared. Call load_local_dataset() and prepare_dataset() first.")

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        args = Seq2SeqTrainingArguments(
            output_dir="diagnosis_engine/trained_models/context",
            per_device_eval_batch_size=16,
            predict_with_generate=True,
            report_to="none",
        )

        if compute_metrics:
            rouge_metric = evaluate.load("rouge")
            bleu_metric = evaluate.load("bleu")

            def compute_metrics_fn(eval_pred):
                predictions, labels = eval_pred
                decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(
                    np.where(labels != -100, labels, self.tokenizer.pad_token_id),
                    skip_special_tokens=True
                )

                rouge_result = rouge_metric.compute(
                    predictions=decoded_preds, references=decoded_labels, use_stemmer=True
                )

                bleu_result = bleu_metric.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )

                exact_matches = np.mean([
                    int(pred.strip().lower() == label.strip().lower())
                    for pred, label in zip(decoded_preds, decoded_labels)
                ])

                result = {
                    "rougeL": round(rouge_result["rougeL"], 4),
                    "bleu": round(bleu_result["bleu"], 4),
                    "exact_match": round(exact_matches, 4)
                }
                return result
        else:
            compute_metrics_fn = None

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn
        )

        metrics = trainer.evaluate()
        return metrics

    def generate_disease_name(self, patient_description):
        """Generates a diagnosis from a free-text patient description"""
        inputs = self.tokenizer(
            [patient_description],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        ).to(self.device)

        outputs = self.model.generate(inputs["input_ids"])
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_model(self, save_path="diagnosis_engine/trained_models/context"):
        """Saves model and tokenizer"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, load_path="diagnosis_engine/trained_models/context"):
        """Loads model and tokenizer from disk"""
        self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
