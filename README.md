# Emotion Models as Design Context Factor

## Introduction
The idea here is to show how the performance of a machine learning model, regardless of the percentage of accuracy, can be significately influenced by modifying the Emotion Model, having everything else the same.

These are the datasets used:

1.- Emotion Detection from Text (https://data.world/crowdflower/sentiment-analysis-in-text)
```bibtex
@misc{crowdflower2016emotion,
  title        = {Emotion Detection from Text},
  author       = {CrowdFlower},
  year         = {2016},
  howpublished = {Data.world},
  url          = {https://data.world/crowdflower/sentiment-analysis-in-text}
}
```


2.- Emotions dataset for NLP (https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp/data)

3.- Emotion (https://huggingface.co/datasets/dair-ai/emotion)

```bibtex
@inproceedings{saravia-etal-2018-carer,
    title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
    author = "Saravia, Elvis  and
      Liu, Hsien-Chi Toby  and
      Huang, Yen-Hao  and
      Wu, Junlin  and
      Chen, Yi-Shin",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1404",
    doi = "10.18653/v1/D18-1404",
    pages = "3687--3697",
    abstract = "Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.",
}
```

4.- ISEAR (https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)

```bibtex
@article{dan2012difficulties,
  title={The difficulties in emotion regulation scale (DERS)},
  author={Dan-Glauser, Elise S and Scherer, Klaus R},
  journal={Swiss Journal of Psychology},
  year={2012},
  publisher={Verlag Hans Huber}
}
```

5.- Google GoEmotions (https://github.com/google-research/google-research/tree/master/goemotions)

```bibtex
@inproceedings{demszky2020goemotions,
 author = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
 booktitle = {58th Annual Meeting of the Association for Computational Linguistics (ACL)},
 title = {{GoEmotions: A Dataset of Fine-Grained Emotions}},
 year = {2020}
}
```

6.- Affective Text Dataset (SemEval-2007 competition) (https://web.eecs.umich.edu/~mihalcea/affectivetext/)

```bibtex
@inproceedings{strapparava2007semeval,
  title={Semeval-2007 task 14: Affective text},
  author={Strapparava, Carlo and Mihalcea, Rada},
  booktitle={Proceedings of the fourth international workshop on semantic evaluations (SemEval-2007)},
  pages={70--74},
  year={2007}
}
```
7.- Emotion-Stimulus dataset in NLP (https://www.site.uottawa.ca/~diana/resources/emotion_stimulus_data/)
```bibtex
@inproceedings{ghazi2015detecting,
  title={Detecting emotion stimuli in emotion-bearing sentences},
  author={Ghazi, Diman and Inkpen, Diana and Szpakowicz, Stan},
  booktitle={Computational Linguistics and Intelligent Text Processing: 16th International Conference, CICLing 2015, Cairo, Egypt, April 14-20, 2015, Proceedings, Part II 16},
  pages={152--165},
  year={2015},
  organization={Springer}
}
```

8.- WASSA-2017 Shared Task on Emotion Intensity (EmoInt) (https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html)
```bibtex
@InProceedings{MohammadB17wassa,
	Title={{WASSA-2017} Shared Task on Emotion Intensity},
	author={Mohammad, Saif M. and Bravo-Marquez, Felipe},
	booktitle={Proceedings of the Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis (WASSA)}, 
	address = {Copenhagen, Denmark},
	year={2017}
}
```