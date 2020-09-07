import itertools
import tempfile
from typing import Dict, Iterable, List, Tuple

import torch

from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.models import Model
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import GradientDescentTrainer, Trainer

# Moved import
# from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp_models.generation.dataset_readers.seq2seq import Seq2SeqDatasetReader

# Deprecated iterator
# from allennlp.data.iterators import BucketIterator
from allennlp.data.samplers import BucketBatchSampler

from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer

# Deprecated tokenizer, 2850579831f392467276f1ab6d5cda3fdb45c3ba
# from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers import SpacyTokenizer

from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation

# Moved import, 61ec73b3ad9d8c4524789814bfa687888a71b996
# from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp_models.generation.models import SimpleSeq2Seq

from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention

# Moved import, 5c022069dcd8af2ae219a09462cb069102505660
# from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp_models.rc.modules.seq2seq_encoders import StackedSelfAttentionEncoder

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding

# Deprecated and also moved,
# - 61ec73b3ad9d8c4524789814bfa687888a71b996
# - b0e2956677e3191ffd51eef25373f0bf4504c05b
# from allennlp.predictors import SimpleSeq2SeqPredictor
from allennlp_models.generation.predictors import Seq2SeqPredictor

from allennlp.training.trainer import GradientDescentTrainer

EN_EMBEDDING_DIM = 256
ZH_EMBEDDING_DIM = 256
HIDDEN_DIM = 256

if torch.cuda.device_count():
    CUDA_DEVICE = torch.cuda.current_device()
else:
    CUDA_DEVICE = -1


def read_data(reader: DatasetReader) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    train_data = reader.read('data/tatoeba/tatoeba.eng_cmn.train.tsv')
    validation_data = reader.read('data/tatoeba/tatoeba.eng_cmn.dev.tsv')
    return train_data, validation_data


def build_dataset_reader() -> DatasetReader:
    return Seq2SeqDatasetReader(
        source_tokenizer=SpacyTokenizer(),
        target_tokenizer=CharacterTokenizer(),
        source_token_indexers={'tokens': SingleIdTokenIndexer()},
        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_data_loaders(
        train_data: torch.utils.data.Dataset,
        dev_data: torch.utils.data.Dataset) -> Tuple[DataLoader, DataLoader]:
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=8, shuffle=False)
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader
) -> Trainer:
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = AdamOptimizer(parameters)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
        cuda_device=CUDA_DEVICE,
    )
    return trainer


def run_training_loop():
    dataset_reader = build_dataset_reader()

    # These are a subclass of pytorch Datasets, with some allennlp-specific
    # functionality added.
    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)

    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    # These are again a subclass of pytorch DataLoaders, with an
    # allennlp-specific collate function, that runs our indexing and
    # batching code.
    train_loader, dev_loader = build_data_loaders(train_data, dev_data)

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this.
    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(
            model,
            serialization_dir,
            train_loader,
            dev_loader
        )
        print("Starting training")
        trainer.train()
        print("Finished training")


def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    # vocab_size = vocab.get_vocab_size("tokens")
    # embedder = BasicTextFieldEmbedder(
    #     {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)})
    # encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
    # return SimpleClassifier(vocab, embedder, encoder)

    en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=EN_EMBEDDING_DIM)
    # encoder = PytorchSeq2SeqWrapper(
    #     torch.nn.LSTM(EN_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
    encoder = StackedSelfAttentionEncoder(input_dim=EN_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, projection_dim=128, feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)

    source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})

    # attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
    # attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM)
    attention = DotProductAttention()

    max_decoding_steps = 20   # TODO: make this variable
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=ZH_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          attention=attention,
                          beam_size=8,
                          use_bleu=True)
    return model


def main():
    run_training_loop()

    # optimizer = optim.Adam(model.parameters())
    # # iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])
    # iterator = BucketBatchSampler(train_dataset, batch_size=32,
    #                               sorting_keys=[("source_tokens", "num_tokens")])

    # train_dataset.index_with(vocab)
    # validation_dataset.index_with(vocab)

    # trainer = GradientDescentTrainer(
    #     model=model,
    #     optimizer=optimizer,
    #     # iterator=iterator,
    #     data_loader=train_dataset,
    #     validation_data_loader=validation_dataset,
    #     num_epochs=1,
    #     # cuda_device=CUDA_DEVICE,
    # )

    # for i in range(50):
    #     print('Epoch: {}'.format(i))
    #     trainer.train()

    #     predictor = SimpleSeq2SeqPredictor(model, reader)

    #     for instance in itertools.islice(validation_dataset, 10):
    #         print('SOURCE:', instance.fields['source_tokens'].tokens)
    #         print('GOLD:', instance.fields['target_tokens'].tokens)
    #         print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])


if __name__ == '__main__':
    # main()
    run_training_loop()
