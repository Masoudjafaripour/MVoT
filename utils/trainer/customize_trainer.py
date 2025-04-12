import os
import torch
import importlib.metadata

from transformers import Trainer, Seq2SeqTrainer

from transformers.utils import is_peft_available
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
)

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset
from packaging import version
from torch import nn

from PIL import Image

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.file_utils import is_datasets_available

from transformers.trainer_utils import PredictionOutput, speed_metrics

from utils.training_arguments import WrappedSeq2SeqTrainingArguments
from utils.visualize import get_concat_h
from utils.postprocess_logits_utils import split_token_sequence

_is_torch_generator_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


from torch.utils.data import DataLoader, Dataset
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_utils import (
    EvalLoopOutput,
    PredictionOutput,
    has_length,
    speed_metrics,
)
from transformers.trainer_pt_utils import (
    EvalLoopContainer,
    IterableDatasetShard,
    find_batch_size
)
from transformers.utils import (
    is_peft_available,
    logging,
)

logger = logging.get_logger(__name__)


class MetricEvalPrediction(NamedTuple):
    predictions: List[dict]
    items: List[dict]
    sketches: Union[List[str], np.ndarray]     # the file path for the sketch


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    sketches: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]


if is_peft_available():
    from peft import PeftModel


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


class CustomizeSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
            self, 
            evaluator,
            *args: WrappedSeq2SeqTrainingArguments,
            eval_examples: Optional[Dataset] = None,
            ignore_pad_token_for_loss: bool = True,
            wandb_run_dir: Optional[str] = None,
            image_loss_func: Optional[torch.nn.Module],
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.eval_examples = eval_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.wandb_run_dir = wandb_run_dir

        self.image_loss_func = image_loss_func
    
    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            eval_examples: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs,
            # max_length: Optional[int] = None,
            # max_time: Optional[int] = None,
            # num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        # self._max_length = max_length if max_length is not None else self.args.generation_max_length
        # self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        # self._max_time = max_time
        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_new_tokens is not None
        ):
            gen_kwargs["max_new_tokens"] = self.args.generation_max_new_tokens
        
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        
        if hasattr(self.args, 'customize_gen_stopping_criteria'):
            if self.args.customize_gen_stopping_criteria:
                gen_kwargs['stopping_criteria'] = self.args.customize_gen_stopping_criteria
        
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if eval_examples is not None and eval_dataset is not None and self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                eval_examples,
                output.predictions,
                "eval_{}".format(self.state.epoch)
            )
            summary = self.compute_metrics(eval_preds, section="dev", finish=True)
            output.metrics.update(summary)

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
    
    def predict(
            self,
            test_dataset: Optional[Dataset],
            test_examples: Optional[Dataset],
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs
            # max_length: Optional[int] = None,
            # max_time: Optional[int] = None,
            # num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        # self._max_length = max_length if max_length is not None else self.args.generation_max_length
        # self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        # self._max_time = max_time
        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_new_tokens is not None
        ):
            gen_kwargs["max_new_tokens"] = self.args.generation_max_new_tokens

        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics


        if self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                test_examples,
                output.predictions,
                metric_key_prefix
            )
            output.metrics.update(self.compute_metrics(eval_preds, section="test", finish=True))

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def _post_process_function(
            self, 
            examples: Dataset, 
            predictions: np.ndarray, 
            # sketches: np.ndarray, 
            stage: str
    ) -> MetricEvalPrediction:
        # assert isinstance(examples, Dataset)
        if self.args.local_rank <= 0:
            print("*"*20)
            print(len(predictions))
            print(len(examples))
            print("*"*20)
        
        tokens = []
        sketches = []
        for r_ids in predictions:
            generated_results = split_token_sequence(
                tokens=torch.tensor(r_ids).unsqueeze(0).to(self.model.device), 
                image_seq_length=self.model.image_token_num,
                boi=self.model.config.boi_token_id, 
                eoi=self.model.config.eoi_token_id,
                max_length=predictions.shape[-1],
                pad_token_id=self.model.config.pad_token_id
            )
            tokens.append(generated_results['texts'])
            if generated_results["images"]:
                generated_imgs = torch.cat(generated_results["images"], dim=0).to(self.model.device)
                generated_imgs = self.model.decode_image_tokens(generated_imgs)
                generated_imgs = self.tokenizer.postprocess_pixel_values(generated_imgs)
            else:
                generated_imgs = None
            sketches.append(generated_imgs)
        
        predictions = torch.cat(tokens, dim=0)

        predictions[predictions == -100] = self.tokenizer.tokenizer.pad_token_id

        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)

        predictions = [i.split("<reserved08706>")[0] for i in predictions]

        sketch_dir = f"{self.args.output_dir}/sketch_{stage}"

        # Save locally.
        if self.is_world_process_zero():
            with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
                json.dump(
                    [dict(
                        **{
                            "idx": examples[idx]['idx'],
                            "prediction": predictions[idx],
                            "text": examples[idx]["input_text"],
                            "labels": examples[idx]['label_text'],
                            "predicted_sketch_paths": [os.path.join(sketch_dir, rf"{str(examples[idx]['idx'])}_{i}_{stage}.png") for i in range(len(sketches[idx]))] if sketches[idx] is not None else None, 
                            "label_img_paths": examples[idx]['label_img_paths'],
                            "input_img_paths": examples[idx]['input_img_paths']
                        }
                        ) for idx in range(len(examples))],
                    f,
                    indent=4,
                )
        
            if stage.startswith("eval"):
                
                if not os.path.exists(sketch_dir):
                    os.mkdir(sketch_dir)
                
                sketch_files = []
                for idx in range(len(examples)):
                    sketches_per_item = sketches[idx]
                    sketch_files_per_item = []
                    if sketches_per_item is not None:
                        for i in range(len(sketches_per_item)):
                            file_path = os.path.join(sketch_dir, rf"{str(examples[idx]['idx'])}_{i}_{stage}.png")
                            img = Image.fromarray(np.transpose((sketches_per_item[i, :, :, :]).cpu().detach().to(torch.uint8).numpy(), (1, 2, 0)).astype(np.uint8))
                            if len(examples[idx]["label_imgs"]) != 0:
                                concat_img = get_concat_h(im1=examples[idx]["label_imgs"][-1], im2=img)
                                concat_img = get_concat_h(im1=examples[idx]['input_imgs'][-1], im2=concat_img)
                            else:
                                concat_img = img
                            concat_img.save(file_path)

                            sketch_files_per_item.append(file_path)

                    sketch_files.append(sketch_files_per_item)

        # Save to wandb.
        if self.wandb_run_dir and self.is_world_process_zero():
            with open(f"{self.wandb_run_dir}/predictions_{stage}.json", "w") as f:
                json.dump(
                    [dict(
                        **{
                            "idx": examples[idx]['idx'],
                            "prediction": predictions[idx],
                            "text": examples[idx]["input_text"],
                            "labels": examples[idx]['label_text'],
                            "predicted_sketch_paths": [os.path.join(sketch_dir, rf"{str(examples[idx]['idx'])}_{i}_{stage}.png") for i in range(len(sketches[idx]))] if sketches[idx] is not None else None, 
                            "label_img_path": examples[idx]['label_img_paths'],
                            "input_img_path": examples[idx]['input_img_paths']
                        }
                        ) for idx in range(len(examples))],
                    f,
                    indent=4,
                )
        if not self.is_world_process_zero():
            sketch_files = []
            for idx in range(len(examples)):
                sketches_per_item = sketches[idx]
                sketch_files_per_item = []
                if sketches_per_item is not None:
                    for i in range(len(sketches_per_item)):
                        file_path = os.path.join(sketch_dir, rf"{str(examples[idx]['idx'])}_{i}_{stage}.png")
                        sketch_files_per_item.append(file_path)
                sketch_files.append(sketch_files_per_item)

        return MetricEvalPrediction(predictions=predictions, sketches=sketches, items=[examples[idx] for idx in range(len(examples))])

    def _compute_metrics(self, eval_prediction: MetricEvalPrediction, section, finish=False) -> dict:
        return self.evaluator.evaluate(eval_prediction.predictions, eval_prediction.items, eval_prediction.sketches, section, finish=finish)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # if self.label_smoother is not None and "labels" in inputs:
        if "labels" in inputs:
            labels = inputs.pop("labels")
            if "img_label" in inputs:
                img_label = inputs.pop("img_label")
            else:
                img_label = None
        else:
            labels = None
        outputs = model(**inputs, output_hidden_states=True)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # text-wise loss
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values() or model_name.endswith('ConditionalGeneration'):
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # perceptual loss
        # assume that I have image_mask now in form of (batch size, max length)
        image_mask = torch.isin(labels, torch.tensor(unwrapped_model.model.bpe_indices).to(labels.device))

        if torch.any(image_mask) and self.image_loss_func:
            # use image_mask to retrieve image tokens from labels and logits distribution from output.logits as well
            image_labels = labels[image_mask]
            image_logits = outputs.logits[:, :-1, :][image_mask[:, 1:], :]

            # for image tokens in the labels, we use model.model.model.convert_bpe2img_tokens to convert it back to visual token indices
            vis_img_tokens = unwrapped_model.model.model.convert_bpe2img_tokens(image_labels)
            # for logits distributions from outputs.logits, we retrieve the corresponding indices from 60k dimensions 1) using torch matmul or 2) just retrieve
            image_probs = torch.nn.functional.softmax(image_logits[:, unwrapped_model.model.bpe_indices], dim=-1)

            label_one_hot = torch.nn.functional.one_hot(vis_img_tokens.reshape(-1).to(torch.int64), num_classes=unwrapped_model.model.model.vqmodel.quantize.embedding.weight.shape[0]).to(torch.bfloat16)
            label_sim_matrix = torch.matmul(label_one_hot.to(unwrapped_model.device), unwrapped_model.model.codebook_sim_matrix)
            discrepancy_loss = torch.mean(torch.sum(label_sim_matrix * image_probs.to(torch.bfloat16), -1))

            loss += discrepancy_loss
            # log the image loss every logging step in addition to total loss
            if self.state.global_step == self._globalstep_last_logged and self.state.global_step != 0:
                self.log({"discrepancy_loss": float(discrepancy_loss)})

        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()

        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()

        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}

        if hasattr(self.model, "image_decoder"):
            generated_tokens, generated_sketch = self.model.generate(**inputs, **gen_kwargs)
        else:
            generated_tokens = self.model.generate(**inputs, **gen_kwargs)
            generated_sketch = None
        
        if hasattr(self.model, "image_postprocess"):
            if self.model.image_postprocess and generated_sketch is not None:
                generated_sketch["sketch"] = self.tokenizer.postprocess_pixel_values(generated_sketch["sketch"])
        
        if hasattr(generated_tokens, "sequences"):
            generated_tokens = generated_tokens.sequences

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                loss = self.compute_loss(
                    model=self.model,
                    inputs=inputs,
                    return_outputs=False
                )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        if generated_sketch is not None:
            return loss, (generated_tokens, generated_sketch), labels
        else:
            return loss, generated_tokens, labels
    

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            # comment it for now
            # if is_torch_xla_available():
            #     xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)
        
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # calculate the metric in a post-hoc way, so commented out here. 
            if self.args.batch_eval_metrics:
                # if self.compute_metrics is not None and logits is not None and labels is not None:
                #     is_last_step = self.accelerator.gradient_state.end_of_dataloader
                #     if args.include_inputs_for_metrics:
                #         metrics = self.compute_metrics(
                #             EvalPrediction(predictions=logits, label_ids=labels, inputs=inputs),
                #             compute_result=is_last_step,
                #         )
                #     else:
                #         metrics = self.compute_metrics(
                #             EvalPrediction(predictions=logits, label_ids=labels),
                #             compute_result=is_last_step,
                #         )

                del losses, logits, labels, inputs

                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs

                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        return EvalLoopOutput(
            predictions=all_preds, 
            label_ids=all_labels, 
            metrics=metrics, 
            num_samples=num_samples
            )