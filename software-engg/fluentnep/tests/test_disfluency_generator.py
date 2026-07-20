import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluentnep.synth.disfluency_generator import DisfluencyGenerator
from fluentnep.synth.text_generator import generate_clean_corpus


def test_labels_align_with_tokens():
    gen = DisfluencyGenerator(seed=1)
    for sentence in generate_clean_corpus(20, seed=1):
        sample = gen.inject(sentence)
        assert len(sample.tokens) == len(sample.tags)
        assert sample.text == " ".join(sample.tokens)


def test_fluent_sample_all_O():
    gen = DisfluencyGenerator(seed=2)
    sample = gen.make_fluent("ma school jaanchhu")
    assert sample.tags == ["O", "O", "O"]
    assert not sample.is_disfluent


def test_all_five_types_appear_over_many_samples():
    gen = DisfluencyGenerator(seed=3)
    sentences = generate_clean_corpus(300, seed=3)
    seen_types = set()
    for s in sentences:
        sample = gen.inject(s, n_disfluencies=3)
        seen_types.update(sample.disfluency_types)
    assert seen_types == {"FILLER", "REPETITION", "FALSE_START", "REPAIR", "PROLONGATION"}


def test_dataset_ratio_roughly_respected():
    gen = DisfluencyGenerator(seed=4)
    sentences = generate_clean_corpus(200, seed=4)
    dataset = gen.generate_dataset(sentences, disfluent_ratio=0.7)
    disfluent_count = sum(s.is_disfluent for s in dataset)
    ratio = disfluent_count / len(dataset)
    assert 0.5 < ratio < 0.9


def test_ids_are_unique():
    gen = DisfluencyGenerator(seed=5)
    sentences = generate_clean_corpus(50, seed=5)
    dataset = gen.generate_dataset(sentences)
    ids = [s.id for s in dataset]
    assert len(ids) == len(set(ids))
