
from transformers import AutoModel
from torch import nn


class Transformers(nn.Module):
    def __init__(self, model):
        super(Transformers, self).__init__()

        self.backbone = AutoModel.from_pretrained(model.name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, model.num_labels)

    def forward(
        self,
        inputs,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.backbone(
            inputs,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        if labels is not None:
            if self.classifier.out_features == 1:
                # regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
        else:
            loss = None

        return logits, loss
