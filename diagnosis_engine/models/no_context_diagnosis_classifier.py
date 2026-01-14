from datasets import load_dataset
from diagnosis_engine.csv_logger_callback import CSVLoggerCallback
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch
import os
import evaluate
import numpy as np

class NoContextDiagnosisClassifier:
    def __init__(self, model_name="t5-small", dataset_name="QuyenAnhDE/Diseases_Symptoms"):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        self.dataset = load_dataset(dataset_name)
        self.tokenized_dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def preprocess_data(self, dataset):
        inputs = dataset["Symptoms"]
        targets = dataset["Name"]

        model_inputs = self.tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=32, truncation=True, padding="max_length")
        
        labels = labels["input_ids"]
        labels = [[label if label != self.tokenizer.pad_token_id else -100 for label in label_seq] for label_seq in labels]
        model_inputs["labels"] = labels

        return model_inputs

    def prepare_dataset(self):
        self.tokenized_dataset = self.dataset.map(self.preprocess_data, batched=True)

        columns_to_remove = ["Symptoms", "Name", "Code", "Treatments"]
        self.tokenized_dataset = self.tokenized_dataset.remove_columns(columns_to_remove)

        split = self.tokenized_dataset["train"].train_test_split(test_size=0.2, seed=42)
        self.train_dataset = split["train"]
        self.test_dataset = split["test"]

    def train(self, num_train_epochs=1000):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        csv_logger = CSVLoggerCallback(train_log_file="diagnosis_engine/trained_models/no_context/metrics/train_no_context_log.csv", eval_log_file="diagnosis_engine/trained_models/no_context/metrics/eval_no_context_log.csv")

        training_args = Seq2SeqTrainingArguments(
            output_dir="diagnosis_engine/trained_models/no_context",
            evaluation_strategy="epoch",
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
            output_dir="./results",
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


    def save_model(self, save_path="diagnosis_engine/trained_models/no_context"):  
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, load_path="diagnosis_engine/trained_models/no_context"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)

    def generate_disease_name(self, symptom_description):
        inputs = self.tokenizer(
            symptom_description,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        ).to(self.device)

        outputs = self.model.generate(inputs["input_ids"])
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)