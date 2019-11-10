# NLP paper implementation with PyTorch
The papers were implemented in using korean corpus 

### Classification
+ Using the [Naver sentiment movie corpus v1.0](https://github.com/e9t/nsmc)
+ Hyper-parameter was arbitrarily selected. (defined by `experiments/base_model/config.json`)

| Model \ Accuracy | Train (120,000) | Validation (30,000) | Test (50,000) | Date |
| :--------------- | :-------: | :------------: | :------: | :--------------: |
| SenCNN           |  90.80%  |     86.48%     |  85.90%  | 191027 |
| CharCNN          | 86.20% | 82.21% | 81.60% | 191027 |
| ConvRec          | 86.48% | 82.81% | 82.45% | 191027 |
| VDCNN            | 87.32% | 84.46% | 84.35% | 191027 |
| SAN | 90.86% | 86.76% | 86.47% | 191027 |
| ETRIBERT | 91.13% | 89.18% | 88.88% | 191027 |
| SKTBERT | 92.39% | 88.98% | 88.98% | 191110 |

* [x] [Convolutional Neural Networks for Sentence Classification](https://github.com/aisolab/nlp_implementation/tree/master/Convolutional_Neural_Networks_for_Sentence_Classification) (as SenCNN)
  + https://arxiv.org/abs/1408.5882
* [x] [Character-level Convolutional Networks for Text Classification](https://github.com/aisolab/nlp_implementation/tree/master/Character-level_Convolutional_Networks_for_Text_Classification) (as CharCNN)
  + https://arxiv.org/abs/1509.01626
* [x] [Efficient Character-level Document Classification by Combining Convolution and Recurrent Layers](https://github.com/aisolab/nlp_implementation/tree/master/Efficient_Character-level_Document_Classification_by_Combining_Convolution_and_Recurrent_Layers) (as ConvRec)
  + https://arxiv.org/abs/1602.00367
* [x] [Very Deep Convolutional Networks for Text Classification](https://github.com/aisolab/nlp_implementation/tree/master/Very_Deep_Convolutional_Networks_for_Text_Classification) (as VDCNN)
  + https://arxiv.org/abs/1606.01781
* [x] [A Structured Self-attentive Sentence Embedding](https://github.com/aisolab/nlp_implementation/tree/master/A_Structured_Self-attentive_Sentence_Embedding_cls) (as SAN)
  + https://arxiv.org/abs/1703.03130
* [x] [BERT_single_sentence_classification](https://github.com/aisolab/nlp_implementation/tree/master/BERT_single_sentence_classification) (as ETRIBERT, SKTBERT)
  + https://arxiv.org/abs/1810.04805

### Paraphrase detection
+ Creating dataset from https://github.com/songys/Question_pair 
+ Hyper-parameter was arbitrarily selected. (defined by `experiments/base_model/config.json`)

| Model \ Accuracy | Train (6,136) | Validation (682) | Test (758) | Date |
| :--------------- | :-------: | :------------: | :------------: | -------------- |
| Siam     |  93.30%  |     83.57%     |     84.16%     | 191028       |
| SAN | 94.86% | 83.13% | 84.96% | 191028 |
| Stochastic | 88.70% | 81.67% | 81.92% | 191106 |
| ETRIBERT | 95.04% | 93.69% | 93.93% | 191004 |
| SKTBERT | 93.64% | 91.34% | 91.16% | 191110 |


* [x] [A Structured Self-attentive Sentence Embedding](https://github.com/aisolab/nlp_implementation/tree/master/A_Structured_Self-attentive_Sentence_Embedding_ptc) (as SAN)
  + https://arxiv.org/abs/1703.03130
* [x] [Siamese recurrent architectures for learning sentence similarity](https://github.com/aisolab/nlp_implementation/tree/master/Siamese_recurrent_architectures_for_learning_sentence_similarity) (as Siam)
  + https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12195
* [x] [Stochastic Answer Networks for Natural Language Inference](https://github.com/aisolab/nlp_implementation/tree/master/Stochastic_Answer_Networks_for_Natural_Language_Inference) (as Stochastic)
  + https://arxiv.org/abs/1804.07888
* [x] [BERT_pairwise_text_classification](https://github.com/aisolab/nlp_implementation/tree/master/BERT_pairwise_text_classification) (as ETRIBERT, SKTBERT)
  + https://arxiv.org/abs/1810.04805

### Language model
* [ ] Character-Aware Neural Language Models
  + https://arxiv.org/abs/1508.06615


### Named entity recognition
| Model \ f1 | Train (81,000) | Validation (9,000) | Date |
| :--------------- | :-------: | :------------: | -------------- |
| BiLSTM-CRF |  79.88%  |     76.45%     | 191004         |
+ Using the [Naver nlp-challange corpus for NER](https://github.com/naver/nlp-challenge/tree/master/missions/ner)
+ Hyper-parameter was arbitrarily selected.
* [x] [Bidirectional LSTM-CRF Models for Sequence Tagging](https://github.com/aisolab/nlp_implementation/tree/master/Bidirectional_LSTM-CRF_Models_for_Sequence_Tagging) (BiLSTM-CRF)
	+ https://arxiv.org/abs/1508.01991
* [ ] End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
	+ https://arxiv.org/abs/1603.01354
* [ ] Neural Architectures for Named Entity Recognition
	+ https://arxiv.org/abs/1603.01360
* [ ] BERT_single_sentence_tagging
	+ https://arxiv.org/abs/1810.04805


### Neural machine translation

| Model \ Perplexity | Train () | Validation  () | Test () | Date |
| ------------------ | -------- | -------------- | ------- | ---- |
| LuongAttn          |          |                |         |      |
| Transformer        |          |                |         |      |

* [x] Effective Approaches to Attention-based Neural Machine Translation (as LuongAttn)
	+ https://arxiv.org/abs/1608.07905
* [ ] Attention Is All You Need (as Transformer)
	+ https://arxiv.org/abs/1706.03762


### Machine reading comprension
* [ ] Machine Comprehension Using Match-LSTM and Answer Pointer
	+ https://arxiv.org/abs/1611.01603
* [ ] Bi-directional attention flow for machine comprehension
	+ https://arxiv.org/abs/1611.01603
* [ ] BERT_question_answering
	+ https://arxiv.org/abs/1810.04805


torchtext, spacy 등을 이용하지않고 최대한 python과 pytorch만을 사용하고, 특히 한국어 corpus를 활용하여, 논문을 구현한 구현체 모음을 공개합니다 (pretrained word vector가 필요한 경우 gluonnlp에서 word vector를 활용합니다.) 특히 자연어처리 논문을 구현할 때, 필요한 glue code들이 무엇인 지 궁금하신 분들은 Vocab, Tokenizer 등의 코드들을 보시면 좋을 것 같습니다.

아래의 repo에는 주로 sentence classification, pairwise-text classfication의 논문들이 구현되어있으며, 현재 추가적으로 named entity recognition, machine reading comprehension, neural machine translation 등을 구현 중입니다. 한국어 데이터에 대해서 현재 개발중이신 모델이 어느 정도로 구현이 잘 된 것인지 확인하실 때, 참고해주시면 좋을 것 같습니다.
[sentence classification]
bert pretrained 활용한 경우 etri와 skt pretrained를 사용한 두 가지 버전이 있습니다. 사용한 데이터셋은 이전에 박은정님이 공개하신 "naver movie review corpus"입니다.
link : https://github.com/e9t/nsmc

1. Convolutional Neural Networks for Sentence Classification (https://arxiv.org/abs/1408.5882)
2. Character-level Convolutional Networks for Text Classification (https://arxiv.org/abs/1509.01626)
3. Efficient Character-level Document Classification by Combining Convolution and Recurrent Layers (https://arxiv.org/abs/1602.00367)
4. Very Deep Convolutional Networks for Text Classification (https://arxiv.org/abs/1606.01781)
5. A Structured Self-attentive Sentence Embedding
(https://arxiv.org/abs/1703.03130)
6. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (https://arxiv.org/abs/1810.04805)

[pairwise-text classification]
bert pretrained 활용한 경우 etri와 skt pretrained를 사용한 두 가지 버전이 있습니다. 사용한 데이터셋은 송영숙님이 공개하신 "Question pair" 데이터입니다.
link : https://github.com/songys/Question_pair

1. Siamese recurrent architectures for learning sentence similarity (https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12195)
2. A Structured Self-attentive Sentence Embedding (https://arxiv.org/abs/1703.03130)
3. Stochastic Answer Networks for Natural Language Inference
(https://arxiv.org/abs/1804.07888)
4. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
(https://arxiv.org/abs/1810.04805)
