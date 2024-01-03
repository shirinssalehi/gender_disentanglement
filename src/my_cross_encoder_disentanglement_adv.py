
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from .. import SentenceTransformer, util
from ..evaluation import SentenceEvaluator
from .disentanglement_model_adv import DisentangledModel
from .hloss import entropy_loss
from .configs_disentanglement import LAMBDA_ATTRIBUTE, LAMBDA_RANKING, LAMBDA_ADV_RANKING


logger = logging.getLogger(__name__)


class DisentangledCrossEncoderAdv():
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = None, device:str = None, tokenizer_args:Dict = {},
                  automodel_args:Dict = {}, default_activation_function = None):
        """
        A CrossEncoder takes exactly two sentences / texts as input and either predicts
        a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
        on a scale of 0 ... 1.

        It does not yield a sentence embedding and does not work for individually sentences.

        :param model_name: Any model name from Huggingface Models Repository that can be loaded with AutoModel. We provide several pre-trained CrossEncoder models that can be used for common tasks
        :param num_labels: Number of labels of the classifier. If 1, the CrossEncoder is a regression model that outputs a continous score 0...1. If > 1, it output several scores that can be soft-maxed to get probability scores for the different classes.
        :param max_length: Max length for input sequences. Longer sequences will be truncated. If None, max length of the model will be used
        :param device: Device that should be used for the model. If None, it will use CUDA if available.
        :param tokenizer_args: Arguments passed to AutoTokenizer
        :param automodel_args: Arguments passed to AutoModelForSequenceClassification
        :param default_activation_function: Callable (like nn.Sigmoid) about the default activation function that should be used on-top of model.predict(). If None. nn.Sigmoid() will be used if num_labels=1, else nn.Identity()
        """

        self.config = AutoConfig.from_pretrained(model_name)
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        self.model = DisentangledModel(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length 

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
            except Exception as e:
                logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)) )
        elif hasattr(self.config, 'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
            self.default_activation_function = util.import_from_string(self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        relevance_labels = []
        gender_labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            relevance_labels.append(example.label["relevance_label"])
            gender_labels.append(example.label["gender_label"])

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        relevance_labels = torch.tensor(relevance_labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)
        gender_labels = torch.tensor(gender_labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, relevance_labels, gender_labels

    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized

    def fit(self,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct = None,
            activation_fct = nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """
        train_dataloader.collate_fn = self.smart_batching_collate
        # file_to_write = open("../checkpoints/adv_experiments/losses_disen_gender_dim100_adv_3_lambda_adv_0.01.txt", "a")

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        adv_attr_discriminator_params = [{'params': p for p in self.model.pre_adv_attribute.parameters()},
                                        {'params': p for p in self.model.adv_attribute.parameters()}]

        optimizer_adv_attribute = torch.optim.RMSprop(adv_attr_discriminator_params, lr=2e-5)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)


        loss_fct_ranking = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        loss_fct_attribute = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        loss_fct_adv_attribute = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        # loss_fct_adv_attribute = nn.MSELoss()


        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, relevance_labels, attribute_labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                if use_amp:
                    with autocast():
                        model_predictions = self.model(features)
                        ranking_logits = nn.Identity()(model_predictions["ranking_logits"])
                        if self.config.num_labels == 1:
                            ranking_logits = ranking_logits.view(-1)

                        loss_ranking = loss_fct_ranking(ranking_logits, relevance_labels)

                        attribute_logits = nn.Identity()(model_predictions["attribute_logits"])
                        if self.config.num_labels == 1:
                            attribute_logits = attribute_logits.view(-1)
                        adv_attribute_logits = nn.Identity()(model_predictions["adv_attribute_logits"])
                        if self.config.num_labels == 1:
                            adv_attribute_logits = adv_attribute_logits.view(-1)

                        loss_adv_attribute = loss_fct_adv_attribute(adv_attribute_logits, attribute_labels)
                        loss_attribute = loss_fct_attribute(attribute_logits, attribute_labels)
                        loss_entropy_attribute = entropy_loss(adv_attribute_logits)
                        loss_value = LAMBDA_RANKING *loss_ranking + \
                                    LAMBDA_ATTRIBUTE *loss_attribute + \
                                        LAMBDA_ADV_RANKING * loss_entropy_attribute


                    # if training_steps % 1000 ==0:
                        # print("loss ranking: {}".format(loss_ranking), "loss_attribute: {}".format(loss_attribute), "loss_entropy_attribute: ", str(loss_entropy_attribute.item()), "\n\n")
                        # print("loss ranking: {}".format(loss_ranking), "loss_attribute: {}".format(loss_attribute), "loss_adv_Attribute: {}".format(loss_adv_attribute), "loss_entropy_attribute: ", str(loss_entropy_attribute.item()), "\n\n")
                        # file_to_write.write("loss ranking: {} ".format(loss_value) + "loss_attribute: {}".format(loss_attribute) + "loss_adv_Attribute: {}".format(loss_adv_attribute)+ "loss_entropy_attribute: ".format(str(loss_entropy_attribute.item()))+"\n\n")
                    training_steps += 1
                    scale_before_step = scaler.get_scale()

                    # optimizer adv
                    scaler.scale(loss_adv_attribute).backward(retain_graph=True)
                    scaler.unscale_(optimizer_adv_attribute)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer_adv_attribute)
                    scaler.update()

                    # freeze the weights of the adversary network
                    # layers_to_freeze = [self.model.pre_adv_attribute, self.model.adv_attribute]
                    # for layer in layers_to_freeze:
                    #     for param in layer.parameters():
                    #         param.requires_grad = False

                    # optimizer loss total
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    print("use amp is False")

                # Unfreeze the weights of the adversary network
                    # layers_to_freeze = [self.model.pre_adv_attribute, self.model.adv_attribute]
                    # for layer in layers_to_freeze:
                    #     for param in layer.parameters():
                    #         param.requires_grad = True

                optimizer.zero_grad()
                optimizer_adv_attribute.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                
                
            #     if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
            #         self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

            #         self.model.zero_grad()
            #         self.model.train()

            # if evaluator is not None:
            #     self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)



    def predict(self, sentences: List[List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               num_workers: int = 0,
               activation_fct = None,
               apply_softmax = False,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               disen = True
               ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.

        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        pred_attributes = []
        pred_attributes_adv = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(features)
                # print(model_predictions["ranking_logits"])
                ranking_logits = activation_fct(model_predictions["ranking_logits"])
                # print(ranking_logits)
                if disen:
                    attribute_logits = activation_fct(model_predictions["attribute_logits"])
                    adv_attribute_logits = activation_fct(model_predictions["adv_attribute_logits"])
                    # adv_attribute_logits = nn.functional.sigmoid(model_predictions["adv_attribute_logits"])

                if apply_softmax and len(ranking_logits[0]) > 1:
                    ranking_logits = torch.nn.functional.softmax(ranking_logits, dim=1)
                    if disen:
                        attribute_logits = torch.nn.functional.softmax(ranking_logits, dim=1)
                        adv_attribute_logits = torch.nn.functional.softmax(adv_attribute_logits, dim=1)

                pred_scores.extend(ranking_logits)
                if disen:
                    pred_attributes.extend(attribute_logits)
                    pred_attributes_adv.extend(adv_attribute_logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]
            if disen:
                pred_attributes = [score[0] for score in pred_attributes]
                pred_attributes_adv = [score[0] for score in pred_attributes_adv]
        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
            if disen:
                pred_attributes = torch.stack(pred_attributes)
                pred_attributes_adv = torch.stack(pred_attributes_adv)

        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])
            if disen:
                pred_attributes = np.asarray([score.cpu().detach().numpy() for score in pred_attributes])
                pred_attributes_adv = np.asarray([score.cpu().detach().numpy() for score in pred_attributes_adv])

        if input_was_string:
            pred_scores = pred_scores[0]
            if disen:
                pred_attributes = pred_attributes[0]
                pred_attributes_adv = pred_attributes_adv[0]
        return pred_scores, pred_attributes, pred_attributes_adv


    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        torch.save(self.model, path)
        # self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path):
        """
        Same function as save
        """
        return self.save(path)
    
    def get_attribute_rep(self, x):
        return self.model.get_attribute_rep(x)


