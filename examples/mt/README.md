# How to create dataset for example

1. Go to [Tatoeba Downloads](https://tatoeba.org/eng/downloads) page
1. Download the sentences file
1. Download the links file
1. Unzip, `tar xvfj ~/Downloads/sentences.tar.bz2 `
1. Unzip, `tar xvfj ~/Downloads/links.tar.bz2`
1. Convert, `python examples/mt/create_bitext.py eng_cmn sentences.csv links.csv > data/tatoeba/tatoeba.eng_cmn.tsv`
1. Shuffle, `shuf data/tatoeba/tatoeba.eng_cmn.tsv > tmp && mv tmp data/tatoeba/tatoeba.eng_cmn.tsv`
1. Sample, `head -10 data/tatoeba/tatoeba.eng_cmn.tsv > data/tatoeba/tatoeba.eng_cmn.sample.tsv`
1. Train, `head -41111 data/tatoeba/tatoeba.eng_cmn.tsv > data/tatoeba/tatoeba.eng_cmn.train.tsv`
1. Dev, `head -46111 data/tatoeba/tatoeba.eng_cmn.tsv | tail -5000 > data/tatoeba/tatoeba.eng_cmn.dev.tsv`
1. Test, `tail -5000 data/tatoeba/tatoeba.eng_cmn.tsv > data/tatoeba/tatoeba.eng_cmn.test.tsv`
