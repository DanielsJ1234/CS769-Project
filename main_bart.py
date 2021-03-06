from datasets import load_dataset, load_metric
from tqdm import tqdm

import json
from typing import Optional
import os
import sys
import numpy as np
import torch
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    BartForConditionalGeneration,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import (
    get_last_checkpoint,
)

@dataclass
class ModelArguments:
    model_name_or_path: str
    config_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    use_fast_tokenizer: bool = True

@dataclass
class DataTrainingArguments:
    task: str = "summarization"
    text_column: str = "table_array"
    summary_column: str = "answer"
    context_column: str = "question"
    train_file: str = "data/fetaQA-v1_train.json"
    validation_file: str = "data/fetaQA-v1_dev.json"
    test_file: str = "data/fetaQA-v1_test.json"
    overwrite_cache: bool = False
    max_source_length: int = 512
    max_target_length: int = 60
    pad_to_max_length: bool = True
    ignore_pad_token_for_loss: bool = True
    source_prefix: str = "summarize: "
    linearization_strategy: str = "simple"
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None



def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)

def preprocess(model_args, data_args, training_args, datasets,tokenizer):
    def default_linearize_table_context(table_array, question):
        return '[CLS]'+question+'[CLS]'+('[SEP]'.join([' '.join(row) for row in table_array]))
    linearize_method = default_linearize_table_context
    def preprocess_function(examples):
        prefix = data_args.source_prefix if data_args.source_prefix is not None else "summarize: "
        inputs = [linearize_method(table, question) for table,question in zip(examples[data_args.text_column],examples[data_args.context_column])]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[data_args.summary_column], max_length=data_args.max_target_length, padding="max_length", truncation=True)
        model_inputs["labels"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        return model_inputs

    train_dataset = None
    eval_dataset = None
    test_dataset = None
    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=None,
            remove_columns=datasets["train"].column_names,
            load_from_cache_file=True,
        )

    if training_args.do_eval:
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=None,
            remove_columns=datasets["train"].column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=None,
            remove_columns=datasets["train"].column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    return train_dataset, eval_dataset, test_dataset

def main(model_args, data_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=None,
        use_fast=model_args.use_fast_tokenizer,
        revision="main",
        use_auth_token=None,
    )
    tokenizer.add_special_tokens({'cls_token':'[CLS]','sep_token':'[SEP]'})
    data_files = {}
    train_file = "data/fetaQA-v1_train.json"
    data_files["train"] = train_file
    validation_file = "data/fetaQA-v1_dev.json"
    data_files["validation"] = validation_file
    test_file = "data/fetaQA-v1_test.json"
    data_files["test"] = test_file
    extension = train_file.split(".")[-1]
    datasets = load_dataset(extension, data_files=data_files,field='data')
    train_dataset, eval_dataset, test_dataset = preprocess(
        model_args, 
        data_args, 
        training_args, 
        datasets, 
        tokenizer)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
    )
    config.task_specific_params['summarization']['prefix'] = "summarize: "
    config.prefix =  "summarize: "

    # # config.d_model = 768
    # config.activation_function = "relu"
    # # config.decoder_attention_heads = 12
    # # config.encoder_attention_heads = 12
    # config.decoder_start_token_id = 0
    # config.eos_token_id = 1
    # config.pad_token_id = 0
    # config.task_specific_params['summarization']['max_length'] = 200
    # config.task_specific_params['summarization']['min_length'] = 30
    # # config.vocab_size = 32128
    # config.max_length = 200
    # config.min_length = 30

    model = BartForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=None,
                revision="main",
                use_auth_token=None,
            )
    if data_args.linearization_strategy != 'concat':
        model.resize_token_embeddings(len(tokenizer))
    trainer = Trainer(
            model = model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
    best_run = None
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
    
    all_metrics = {}
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint) if best_run is None else trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        if trainer.is_world_process_zero():
            metrics_formatted = trainer.metrics_format(metrics)
            print("***** train metrics *****")
            print(metrics_formatted)
            save_json(metrics, os.path.join(training_args.output_dir, "train_results.json"))
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
            all_metrics.update(metrics)

    # torch.cuda.empty_cache()
    if training_args.do_eval:
        print("*** Evaluate ***")
        metrics = trainer.evaluate(
            eval_dataset = eval_dataset, metric_key_prefix="eval"
        )
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        if trainer.is_world_process_zero():
            metrics_formatted = trainer.metrics_format(metrics)
            print("***** val metrics *****")
            k_width = max(len(str(x)) for x in metrics_formatted.keys())
            v_width = max(len(str(x)) for x in metrics_formatted.values())
            for key in sorted(metrics_formatted.keys()):
                print(str(key) + " " * (k_width - len(key)) + ": " + str(metrics_formatted[key]) + " " * (v_width - len(str(metrics_formatted[key]))))
            save_json(metrics, os.path.join(training_args.output_dir, "eval_results.json"))
            all_metrics.update(metrics)


    if training_args.do_predict:
        print("*** Test ***")

        raw_logits = None
        test_results = None
        # for i in range(int(len(test_dataset)/128)):
        #     test_dataset_batch = test_dataset.select(range(i*128,(i+1)*128))
        #     test_results = trainer.predict(test_dataset_batch, metric_key_prefix="test",)
        #     if raw_logits is None:
        #         raw_logits = test_results.predictions[0]
        #     else:
        #         raw_logits = np.vstack((raw_logits, test_results.predictions[0]))
        # torch.cuda.empty_cache()
        

        # test_dataset_batch = test_dataset.select(range(0,128))

        test_results = trainer.predict(test_dataset, metric_key_prefix="test",)

        if raw_logits is None:
            raw_logits = test_results.predictions[0]
        else:
            raw_logits = np.vstack((raw_logits, test_results.predictions[0]))

        metrics = test_results.metrics

        max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))


        if trainer.is_world_process_zero():
            metrics_formatted = trainer.metrics_format(metrics)
            print("***** test metrics *****")
            print(metrics_formatted)
            save_json(metrics, os.path.join(training_args.output_dir, "test_results.json"))
            all_metrics.update(metrics)
            print("generating predictions")
            raw_preds = np.argmax(raw_logits, axis = -1)
            test_preds = tokenizer.batch_decode(raw_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            test_preds = [pred.strip() for pred in test_preds]
            
            answers = []
            with open(test_file) as f:
                data = json.load(f)
                for i in data["data"]:
                    answers.append(i["answer"])
            output_test_preds_file = os.path.join(training_args.output_dir, "test_preds_seq2seq2.txt")
            print("writing output file")
            with open(output_test_preds_file, "a+") as writer:
                writer.write("\n".join(test_preds))
            with open(output_test_preds_file, "a+") as writer:
                writer.write("\n")
            print("running metrics")
            b_score(test_preds, answers) # unable to run on GPU, will need to process manually later
            sacrebleu_score(test_preds, answers)

def getCompareData(pred_file, test_file):
    preds = []
    questions = []
    answers = []
    with open(pred_file, encoding="utf-8") as f:
        for l in f:
            preds.append(l.strip("\n"))
    with open(test_file) as f:
        data = json.load(f)
        for i in data["data"]:
            answers.append(i["answer"])
            questions.append(i["question"])
    return preds, answers, questions

def b_score(preds, answers):
    metric = load_metric("bertscore")
    results = metric.compute(predictions=preds, references=answers[:len(preds)], lang="en")
    n = len(preds)
    print("bertscore:\nPrecision: {:.2f}\nRecall: {:.2f}\nF1: {:.2f}".format(
        sum(results["precision"])/n, sum(results['recall'])/n, sum(results['f1'])/n))
    return results

def sacrebleu_score(preds, answers):
    metric = load_metric("sacrebleu")
    results = metric.compute(predictions=preds, references=[[a] for a in answers[:len(preds)]])
    print("sacreBLEU:\nScore: {:.2f}\n".format(
        results['score']))
    return results

if __name__ == "__main__":
    set_seed(24)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        m_args, d_args, t_args = parser.parse_json_file(json_file=sys.argv[1])
        print("training arguments: \n", t_args)
        print("model arguments: \n", m_args)
        print("data arguments: \n", d_args)
        main(m_args, d_args, t_args)
    else:
        print("config file required")