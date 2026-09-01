from typing import TYPE_CHECKING, Any, Optional, Union

import torch
import torch.distributed as dist
from transformers import Seq2SeqTrainer
from typing_extensions import override
from peft import LoraConfig, TaskType, get_peft_model

from spatiallm.tuner.framework import logging
from spatiallm.tuner.framework.utils import is_transformers_version_greater_than
from spatiallm.tuner.framework.callbacks import (
    LogCallback,
    ReporterCallback,
    get_swanlab_callback,
)
from spatiallm.tuner.framework.loader import load_tokenizer, load_model
from spatiallm.tuner.hparams_lora import get_train_args, read_args
from spatiallm.tuner.data import (
    IGNORE_INDEX,
    get_dataset,
    get_template_and_fix_tokenizer,
    register_spatiallm_templates,
    SFTDataCollatorWith4DAttentionMask,
)


if TYPE_CHECKING:
    from transformers import (
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )

    from spatiallm.tuner.hparams_lora import (
        DataArguments,
        FinetuningArguments,
        GeneratingArguments,
        ModelArguments,
    )

logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = (
                self.processing_class.pad_token_id
            )
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def create_optimizer(self):
        """
        Setup optimizer with different learning rates:
        - point_proj: use finetuning_args.proj_lr (or fallback to training_args.learning_rate)
        - LoRA params: use training_args.learning_rate
        """
        if self.optimizer is not None:
            return self.optimizer

        # Get all trainable parameters
        decay_parameters = self.get_decay_parameter_names(self.model)
        params = []
        point_proj_lr = getattr(self.finetuning_args, "proj_lr", self.args.learning_rate)

        point_proj_params = []
        lora_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "point_proj" in name:
                point_proj_params.append(param)
            else:
                lora_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": point_proj_params,
                "lr": point_proj_lr,
                "weight_decay": self.args.weight_decay if any(p in decay_parameters for p in [n for n, _ in self.model.named_parameters() if "point_proj" in n]) else 0.0,
            },
            {
                "params": lora_params,
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
        ]

        optimizer_cls, optimizer_kwargs = Seq2SeqTrainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    register_spatiallm_templates(
        cutoff_len=data_args.cutoff_len,
        num_bins=data_args.num_bins,
        do_augmentation=data_args.do_augmentation,
        random_rotation=data_args.random_rotation,
    )

    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(model_args, data_args, training_args)
    model = load_model(
        tokenizer, data_args, model_args, finetuning_args, training_args.do_train
    )

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LoRA <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if getattr(finetuning_args, "lora_target", None) is None:
        target_modules = ["q_proj", "v_proj"]
    else:
        target_modules = finetuning_args.lora_target.split(",") if isinstance(finetuning_args.lora_target, str) else finetuning_args.lora_target
    task_type = TaskType.SEQ_2_SEQ_LM if model.config.is_encoder_decoder else TaskType.CAUSAL_LM
    logger.info('{}, {}'.format(task_type, target_modules))

    modules_to_save = []
    if finetuning_args.modules_to_save:
        modules_to_save = finetuning_args.modules_to_save.split(",")
    if "point_proj" not in modules_to_save:
        modules_to_save.append("point_proj")

    lora_config = LoraConfig(
        r=finetuning_args.lora_rank,
        lora_alpha=finetuning_args.lora_alpha,
        lora_dropout=finetuning_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LoRA <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Proj <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    for name, param in model.named_parameters():
        if "point_proj" in name:
            param.requires_grad = True
            logger.info(f"Unfreezing {name} for training.")
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Proj <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print("=== Trainable Parameters ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name} | shape: {param.shape}")
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=(
            8 if training_args.do_train else None
        ),  # for shift short attention
        label_pad_token_id=(
            IGNORE_INDEX
            if data_args.ignore_pad_token_for_loss
            else tokenizer.pad_token_id
        ),
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [
        tokenizer.eos_token_id
    ] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.save_model()

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    model_args, data_args, training_args, finetuning_args, generating_args = (
        get_train_args(args)
    )

    callbacks: list[Any] = []
    callbacks.append(LogCallback())
    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    callbacks.append(
        ReporterCallback(model_args, data_args, finetuning_args, generating_args)
    )  # add to last

    run_sft(
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
        callbacks,
    )

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")


def run_exp(args: Optional[dict[str, Any]] = None) -> None:
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    _training_function(config={"args": args})


if __name__ == "__main__":
    run_exp()
