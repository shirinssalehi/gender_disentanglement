import torch
from torch import nn
from transformers import AutoModel
from .configs_disentanglement import REP_DIMENSION, GENDER_DIMENSION

class DisentangledModel(nn.Module):
    """
    """
    def __init__(self, model_name):
        super(DisentangledModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.n_ranking = REP_DIMENSION - GENDER_DIMENSION
        self.n_attribute = GENDER_DIMENSION
        self.preranker = nn.Linear(self.n_ranking, self.n_ranking)
        self.ranking_layer = nn.Linear(self.n_ranking, 1, bias=True)
        self.preclassifier = nn.Linear(self.n_attribute, self.n_attribute)
        self.gender_classifier = nn.Linear(self.n_attribute, 1, bias=True)
        self.pre_adv_attribute = nn.Linear(self.n_ranking, self.n_ranking)
        self.adv_attribute = nn.Linear(self.n_ranking, 1)
        self.dropout_ranker = nn.Dropout(p=0.1)
        self.dropout_classifier = nn.Dropout(p=0.1)
        self.dropout_adv_attribute = nn.Dropout(p=0.1)
        print("n_ranking = {}".format(self.n_ranking))

    def forward(self, x):
        vector_representation = self.encoder(**x)
        vector_representation = vector_representation.last_hidden_state
        vector_representation = vector_representation[:, 0, :]
        # ranker
        vector_ranking = vector_representation[:, :self.n_ranking]
        ranking_rep = self.preranker(vector_ranking)
        ranking_rep = nn.ReLU()(ranking_rep)
        ranking_rep = self.dropout_ranker(ranking_rep)
        ranking_logits = self.ranking_layer(ranking_rep)
        # classifier
        attribute_rep = vector_representation[:, self.n_ranking:]
        attribute_rep = self.preclassifier(attribute_rep)
        attribute_rep = nn.ReLU()(attribute_rep)
        attribute_rep = self.dropout_classifier(attribute_rep)
        attribute_logits = self.gender_classifier(attribute_rep)
        # adv attribute
        adv_attribute_rep = self.pre_adv_attribute(vector_ranking)
        adv_attribute_rep = nn.ReLU()(adv_attribute_rep)
        adv_attribute_rep = self.dropout_adv_attribute(adv_attribute_rep)
        adv_attribute_logits = self.adv_attribute(adv_attribute_rep)
        # print(adv_attribute_logits)
        # adv_attribute_logits = torch.sigmoid(adv_attribute_logits)
        adv_attribute_logits = torch.nn.functional.softmax(adv_attribute_logits)

        out = {"ranking_logits": ranking_logits, "attribute_logits": attribute_logits, "adv_attribute_logits": adv_attribute_logits}
        return out
    
    def get_attribute_rep(self,x):
        vector_representation = self.encoder(**x)
        vector_representation = vector_representation.last_hidden_state
        vector_representation = vector_representation[:, 0, :]
        attribute_rep = vector_representation[:, self.n_ranking:]
        return attribute_rep




    

