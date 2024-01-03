from torch import nn
from transformers import AutoModel
from configs_disentanglement import REP_DIMENSION


class OriginalModel(nn.Module):
    """
    """

    def __init__(self, model_name):
        super(OriginalModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.preranker = nn.Linear(REP_DIMENSION, REP_DIMENSION, bias=True)
        self.ranking_layer = nn.Linear(REP_DIMENSION, 1, bias=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        vector_representation = self.encoder(**x, return_dict=True)
        vector_representation = vector_representation.last_hidden_state
        # vector_representation = torch.mean(vector_representation, dim=1)
        vector_representation = vector_representation[:, 0, :]
        # vector_representation = vector_representation[:, :5]
        vector_representation = self.preranker(vector_representation)
        vector_representation = nn.ReLU()(vector_representation)
        vector_representation = self.dropout(vector_representation)
        ranking_logits = self.ranking_layer(vector_representation)
        return ranking_logits
