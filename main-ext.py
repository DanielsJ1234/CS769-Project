from datasets import load_dataset, load_metric
from tqdm import tqdm

import json

import os
import sys
from dataclasses import dataclass

from transformers import (
    T5Tokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    Seq2SeqTrainer,
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



def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)

def preprocess(model_args, data_args, training_args, datasets,tokenizer):
    def default_linearize_table_context(table_array, question, page, section):
        return '[CLS]'+question+'[CLS]'+page+'[CLS]'+section+'[CLS]'+('[SEP]'.join([' '.join(row) for row in table_array]))
    linearize_method = default_linearize_table_context
    def preprocess_function(examples):
        prefix = data_args.source_prefix if data_args.source_prefix is not None else "summarize: "
        inputs = [linearize_method(table, question, page, section) 
                  for table,question,page,section in 
                  zip(examples[data_args.text_column],examples[data_args.context_column], 
                      examples["table_page_title"], examples["table_section_title"])]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[data_args.summary_column], max_length=data_args.max_target_length, padding="max_length", truncation=True)
        model_inputs["labels"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        return model_inputs

    train_dataset = None
    test_dataset = None
    if training_args.do_train:
        train_dataset = datasets["train"]
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=None,
            remove_columns=datasets["train"].column_names,
            load_from_cache_file=True,
        )

    if training_args.do_predict:
        test_dataset = datasets["test"]
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=None,
            remove_columns=datasets["train"].column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    return train_dataset, test_dataset

def main(model_args, data_args, training_args):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
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
    train_dataset, test_dataset = preprocess(
        model_args, 
        data_args, 
        training_args, 
        datasets, 
        tokenizer)
    
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=None,
                revision="main",
                use_auth_token=None,
            )
    
    trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
    best_run = None
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
    
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

    print("*** Test ***")

    test_results = trainer.predict(
        test_dataset,
        metric_key_prefix="test",
        max_length=data_args.max_target_length,
        num_beams=None,
    )
    metrics = test_results.metrics
    metrics["test_samples"] = len(test_dataset)
    if trainer.is_world_process_zero():
        metrics_formatted = trainer.metrics_format(metrics)
        print("***** test metrics *****")
        print(metrics_formatted)
        save_json(metrics, os.path.join(training_args.output_dir, "test_results.json"))

        print("writing base pred file")
        with open("test.txt", "w") as writer:
            writer.write("\n".join(test_results.predictions))

        print("generating predictions")
        test_preds = tokenizer.batch_decode(test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        test_preds = [pred.strip() for pred in test_preds]
        
        answers = []
        with open(test_file) as f:
            data = json.load(f)
            for i in data["data"]:
                answers.append(i["answer"])
        output_test_preds_file = os.path.join(training_args.output_dir, "test_preds_seq2seq.txt")
        print("writing output file")
        with open(output_test_preds_file, "w") as writer:
            writer.write("\n".join(test_preds))
        print("running metrics")
        #b_score(test_preds, answers) # unable to run on GPU, will need to process manually later
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
    results = metric.compute(predictions=preds, references=answers, lang="en")
    n = len(preds)
    print("bertscore:\nPrecision: {:.2f}\nRecall: {:.2f}\nF1: {:.2f}".format(
        sum(results["precision"])/n, sum(results['recall'])/n, sum(results['f1'])/n))
    return results

def sacrebleu_score(preds, answers):
    metric = load_metric("sacrebleu")
    results = metric.compute(predictions=preds, references=[[a] for a in answers])
    print("sacreBLEU:\nScore: {:.2f}\n".format(
        results['score']))
    return results

def category_metrics(test_preds_file, test_data_file):
    cats = ["ent", "sum", "nom", "mult", "list", "amb", "rep", "inf", "error"]
    test_preds_split = test_preds_file.split(".")
    test_preds_template = ".".join(test_preds_split[0:-1])+"_{}."+test_preds_split[-1]
    test_data_split = test_data_file.split(".")
    test_data_template = ".".join(test_data_split[0:-1])+"_{}."+test_data_split[-1]
    for c in cats:
        print(c)
        preds, answers, questions = getCompareData(
            test_preds_template.format(c), 
            test_data_template.format(c))
        res = sacrebleu_score(preds, answers)
        res = b_score(preds, answers)
        print("----------------\n\n")

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
    
    
    
    
