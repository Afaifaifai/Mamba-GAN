from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    AdamW,
    BertForSequenceClassification
)

def create_bert_model(self, model_name_or_path, loss_type, model_type=None, random_weights=False):

    config_class = BertConfig
    config = config_class.from_pretrained(model_name_or_path, cache_dir=None)

    if model_type == "bert_lm":
        if random_weights:
            print("Starting from random")
            model = BertForSequenceClassification(config=config)
        else:
            model_class = BertForMaskedLM
            model_lm = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
                cache_dir=None,
            )
            model = BertForSequenceClassification(config=config)
            model.bert = model_lm.bert

    else:
        if random_weights:
            raise NotImplementedError
        model_class = BertForSequenceClassification
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=None,
        )

    return model.bert if loss_type == "mmd" else model


def main(cfg, vocab):
    ntokens = len(vocab)
        # Create discriminator
    if cfg.DISCRIMINATOR.type == "bert":
        # Can change d_embed
        discriminator = create_bert_model(
            cfg.DISCRIMINATOR.BERT.model_path, cfg.DISCRIMINATOR.BERT.loss_type, cfg.DISCRIMINATOR.BERT.model_type,
            cfg.DISCRIMINATOR.BERT.random_weights
        )
        discriminator.unfreeze_idx = self.calculate_unfreeze_idx(cfg)

    temperature = 1
    vocab = vocab
    vec_len = vocab.vec_len



main()