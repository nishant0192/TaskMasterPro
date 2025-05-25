# ai-service/app/services/training_service.py
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_metric
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from app.core.config import settings

class TrainingService:
    def __init__(self):
        self.csv_path = os.path.join("data", "training_data.csv")
        self.output_dir = os.path.join(settings.MODEL_DIR, settings.FINE_TUNED_MODEL_NAME)

    def load_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path)

    def preprocess(self, tokenizer, examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    def optimize_model(self, model):
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    def fine_tune(self):
        df = self.load_csv()
        num_labels = df["label"].nunique()
        train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

        train_ds = Dataset.from_pandas(train_df)
        eval_ds  = Dataset.from_pandas(eval_df)

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )

        train_ds = train_ds.map(lambda x: self.preprocess(tokenizer, x), batched=True)
        eval_ds  = eval_ds.map(lambda x: self.preprocess(tokenizer, x), batched=True)

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="steps",
            save_steps=100,
            eval_steps=100,
            logging_dir="./logs",
            logging_steps=10,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=5e-5,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )

        metric = load_metric("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            pred = logits.argmax(-1)
            return metric.compute(predictions=pred, references=labels)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(self.output_dir)

        optimized = self.optimize_model(model)
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(optimized.state_dict(), os.path.join(self.output_dir, settings.OPTIMIZED_MODEL_NAME))

        return {"model_dir": self.output_dir}
