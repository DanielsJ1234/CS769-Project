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
    T5Tokenizer,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    HfArgumentParser,
    Seq2SeqTrainer,
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
    model_1_name_or_path: str
    model_2_name_or_path: str
    config_1_name: Optional[str] = None
    config_2_name: Optional[str] = None
    tokenizer_1_name: Optional[str] = None
    tokenizer_2_name: Optional[str] = None
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


    tokenizer_bart = AutoTokenizer.from_pretrained(
        model_args.tokenizer_1_name if model_args.tokenizer_1_name else model_args.model_1_name_or_path,
        cache_dir=None,
        use_fast=model_args.use_fast_tokenizer,
        revision="main",
        use_auth_token=None,
    )
    tokenizer_bart.add_special_tokens({'cls_token':'[CLS]','sep_token':'[SEP]'})

    tokenizer_t5 = T5Tokenizer.from_pretrained('t5-small')
    tokenizer_t5.add_special_tokens({'cls_token':'[CLS]','sep_token':'[SEP]'})

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
        tokenizer_bart)


    config_bart = AutoConfig.from_pretrained(
        model_args.config_1_name if model_args.config_1_name else model_args.model_1_name_or_path,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
    )
    config_bart.task_specific_params['summarization']['prefix'] = "summarize: "
    config_bart.prefix =  "summarize: "

    model_bart = BartForConditionalGeneration.from_pretrained(
                model_args.model_1_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_1_name_or_path),
                config=config_bart,
                cache_dir=None,
                revision="main",
                use_auth_token=None,
            )

    # if data_args.linearization_strategy != 'concat':
    #     model_bart.resize_token_embeddings(len(tokenizer_bart))
    # trainer_bart = Trainer(
    #         model = model_bart,
    #         args=training_args,
    #         train_dataset=train_dataset if training_args.do_train else None,
    #         eval_dataset=eval_dataset if training_args.do_eval else None,
    #         tokenizer=tokenizer_bart,
    #         data_collator=default_data_collator,
    #     )

    trainer_bart = Seq2SeqTrainer(
            model = model_bart,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer_bart,
            data_collator=default_data_collator,
        )

    config_t5 = AutoConfig.from_pretrained(
        model_args.model_2_name_or_path,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
    )
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_2_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_2_name_or_path),
                config=config_t5,
                cache_dir=None,
                revision="main",
                use_auth_token=None,
            )
    
    trainer_t5 = Seq2SeqTrainer(
            model=model_t5,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=None,
            tokenizer=tokenizer_t5,
            data_collator=default_data_collator,
        )
    print("*** Test ***")

    test_dataset_batch = test_dataset.select(range(0,128))

    # test_results_bart = trainer_bart.predict(test_dataset_batch, metric_key_prefix="test",)
    test_results_bart = trainer_bart.predict(
        test_dataset_batch,
        metric_key_prefix="test",
        max_length=data_args.max_target_length,
        num_beams=None,
    )
    test_results_t5 = trainer_t5.predict(
        test_dataset_batch,
        metric_key_prefix="test",
        max_length=data_args.max_target_length,
        num_beams=None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_logits_bart = test_results_bart.predictions[0]
    raw_logits_t5 = test_results_t5.predictions[0]
    raw_logits_t5 = raw_logits_t5[:,:,: raw_logits_bart.shape[-1]]

    # # final_raw_logits = np.maximum(raw_logits_bart, raw_logits_t5)
    raw_logits_t5 = torch.from_numpy(raw_logits_t5).to(device)
    raw_logits_bart = torch.from_numpy(raw_logits_bart).to(device)
    

    log_probs_bart = torch.nn.functional.log_softmax(raw_logits_bart, dim=-1)
    log_probs_t5 = torch.nn.functional.log_softmax(raw_logits_t5, dim=-1)
    final_raw_logits = torch.max(log_probs_bart, log_probs_t5)

    print("generating predictions")
    raw_preds = torch.argmax(final_raw_logits, dim=-1)
    test_preds = tokenizer_t5.batch_decode(raw_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
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