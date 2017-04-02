# SQuAD-QA-System
This is the adapted implementation (the model has been adapted) of [BI-DIRECTIONAL ATTENTION FLOW FOR MACHINE COMPREHENSION](https://arxiv.org/abs/1611.01603) (Seo et al., 2016).

## Usage
### 0. Requirements
- Python (Verified on 2.7)
- tensorflow (Verified on r1.1)
- numpy
- nltk
- tqdm
- pyprind
- json

### 1. Preprocessing
Run the following command line in the 'code' folder to preprocess the data:

```
./get_started.sh
```

In this script, the following preprocessing steps will be implemented:
- The SQuAD dataset is downloaded and placed in the **_data/squad_** folder. Afterwards, it is tokenized by the tokenizer tool nltk and distributed into four files including contest, question, answer and answer span. Lines in these files are aligned. Each line in the answer span file contains two numbers: the first number refers to the index of the first word of the answer in the context paragraph. The second number is the index of the last word of the answer in the context paragraph.
- [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/) of dimensionality d = 50, 100, 200, and 300 and vocab size of 400k that have been pretrained on Wikipedia 2014 and Gigaword 5 are downloaded and stored in the **_data/dwr_** subfolder. By default, embedding dimensionality of 100 is used.
- The vocabulary of the SQuAD dataset is extracted and then the GloVe embedding is accordingly trimmed into a much smaller file.
- For SQuAD dataset, the train dataset is splitted into two parts: a 95% slice for training, and the rest 5% for validation purposes. The dev dataset will be used as the test set in this project.

### 2. Training
Run the following command line to start training the model:

```
python train.py
```

This model has around 1.8 million trainable parameters. For faster training, computing environment with a GPU is preferred. During training, the model parameters will be stored for each epoch.

### 3. Evaluation
First run the following command line to output a JSON file with the prediction result on the dev dataset (named dev-prediction.json and stored in the **_code_** folder):

```
python qa_answer.py
```

Then run the following command line to calculate the ExactMatch (EM) and F1 score to evaluate the performance of the model:

```
python evaluate.py data/squad/dev-v1.1.json dev-prediction.json
```

## Modeling
In brief, the adapted model mainly consists of four layers, as discussed in details here:

### Word Embedding Layer
Based on the pre-trained word vectors GloVe ([Pennington et al., 2014](https://nlp.stanford.edu/pubs/glove.pdf)), each token in the context and question is mapped into a high dimensional vector space to obtain word embeddings.

### Contextual Embedding Layer
The context word vectors are first filtered based on the relevance between the context and question so that only the portion of context that is most related to question would be transported into the following layers and the unrelevant noise can be suppressed. The revevance is measured by calculating the cosine similarity between the context and question word vectors, as brought forward by [Wang et al. (2016)](https://arxiv.org/abs/1612.04211).
Then the filtered context and question word vectors are input into a bi-directional Long Short-Term Memory Network (BiLSTM) and the hidden states in both directions are recorded and concatenated to extract the contextual vector representations of context and question.

### Attention Flow Layer
In this layer, attentions in two directions including the one from context to query and the other from query to context are computed to link and fuse information from the context and the query words. Details can be found in [Seo et al. (2016)](https://arxiv.org/abs/1611.01603).

### Decoding Layer
This layer first takes as input the query-aware representations of context words obtained in the last layer and further captures the interaction among the context words conditioned on the query by using another BiLSTM. Then the input and output of this BiLSTM layer are concatenated and used to calculate the probability for each position in the context as the start and the end of the answer with SoftMax. In some situations, though very rare, we would flip over the start and ending labels if the ending predicted maybe even appears earlier than the start in the context. 

## Training Details
The training loss (to be minimized) is defined as the sum of the negative log probabilities of the true start and end indices by the predicted distributions, averaged over all examples.
The model is trained for 10 epochs with a batch size of 128 and an initial learning rate 0f 0.001 that decays at a rate of 0.9 per epoch. The hidden state dimension of LSTM is 100.

## Results
| F1 | EM |
|----|----|
| 65.8 | 52.0 |

## Directions for improvement
- The word embeddings used here is pre-trained on 6B word corpora (Wikipedia + Gi-gaword) and the dimension is chosen as 100. A larger dimension can be selected and other kinds of word embeddings training on larger corpora such as the [GloVe trained on 840B comman crawl texts](https://nlp.stanford.edu/projects/glove/) can be used.
- Here NLTK is used as the tokenizing tool. As a more powerful alternative, the [Stanford CoreNLP tokenizer](https://nlp.stanford.edu/software/tokenizer.html) can be used.
- In the decoding layer, when calculating the probability of start word and end word position, a more advanced pointer network ([Wang and Jiang, 2016](https://arxiv.org/abs/1608.07905)) can be utilized to improve the output accuracy.
- In the decoding layer, the single layer BiLSTM can be replaced by the multi-layer BiLSTM, which should be better.
- The hidden state dimensionality should be tuned based on the validation set.

## Acknowledgement
This project is designed for [Stanford cs224n course](http://web.stanford.edu/class/cs224n/). Thanks for the great guidance of this course!
