### NLP with Deep Learning (https://youtu.be/8rXD5-xhemo) with CS284
#### Lecture:01
- [Course Website](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1204/)
- Language passes information in comprssed form.
- Meaning: The idea represented by a words
- WordNet: Synonym words (fine grained )
- Problems in wordNet: Missing new words, No word similarity
- Traditional NLP(before 2012)- words are denoted as discrete sympols ( 1 or 0)--> one hot encoded vector
- Problem with one hot -> 
		- very large vector for vocab
		- does not represent relationship b/w words
	
- Representation of words with their Context: Distributional symentics
	  -  word's meaning is given by the words that frequently appear close by:
	  -  context = set of words that appear near by
- Word vector representation: Distributed vector
		- Vector representation also shows word similarity information
		- Non-zero number vector
		- also called word embeddings
		- not standard vectors
		- dimension of vector is not fixed
- **Word2Vec:**
	- A framework for learning word vector
	- Word Embedding is a language modeling technique used for mapping words to vectors of real numbers.
	-  Corpus(corpora) to Word vectors(embedding)
	-  work on probability model (predicting words around a center word)
	-  start with random vectors
	-  Distribution vector means: word vector know which word occur in context of itself(so it wil give high probablity)
	-  Like if word is bank the  words like, branch, withdrawal, credit, debit , will be given high probability
	-  Shallow 2 layer NN
	-  Two architectures of Word2Vec:
	-  Word2vec algorithm learns word associations from large text corpora
	1. CBOW(continious Bag of words)--> predict current word given conext words
	2. Skip gram  --> predict context words given current word
	-  **The basic idea of word embedding is words that occur in similar context tend to be closer to each other in vector space.**
	-  king-men+women= Queen
	-  Python Library for word vector: Gensim
	-  CBOW is faster while skip-gram does a better job for infrequent words.
	-  Invented at Google
	-  Milokov ----et al 2013 paper introduced word2vec


### Lecture:02
- **The idea of word2vec is to maximise the similarity (dot product) between the vectors for words which appear close together (in the context of each other) in text, and minimise the similarity of words that do not**
-  Word2vec put similar words together in vector space.
-  word2vec: two model variants
-  1. Continious skip gram model (SG) --> Predicts context words(position independent) given center words
-  2. Continious Bag of words(CBOG)  --> predict center words given bag of context words
-  Naive softmax : expensive training skip gram model 
-  skip-gram with negative sampling:
-  original word2vec model was computationally expensive: 
- Author did two modification to deal with:
- 1. Subsampling frequent words to decrease the number of training examples.
- 2. Modifying the optimization objective with a technique they called “Negative Sampling”, which causes each training sample to update only a small percentage of the model’s weights.
- It’s worth noting that subsampling frequent words and applying Negative Sampling not only reduced the compute burden of the training process, but also improved the quality of their resulting word vectors as well.
-  **the size of our word vocabulary means that our skip-gram neural network has a tremendous number of weights, all of which would be updated slightly by every one of our billions of training samples!**
-  **Negative sampling addresses this by having each training sample only modify a small percentage of the weights, rather than all of them. Here’s how it works**
-  when the number of output classes is very large , computing softmax is very expensive. for this various approximations to make the computation more efficient.
-  various approximations to the softmax layer that have been proposed over the last years, some of which have so far only been employed in the context of language modelling or MT.

### Softmax approximation methods:
1. Softmax based : softmax is intact
2. sampling based: some approximation of softmax

- Softmax based Approahes:
	1. Hierarchical Softmax
	2. Differentiated Softmax
	3. CNN Softmax
-  Sampling bases approaches:
    -   They do this by approximating the normalization in the denominator of the softmax with some other loss that is cheap to compute.
    -   However, sampling-based approaches are only useful at training time -- during inference, the full softmax still needs to be computed to obtain a normalised probability.
	1. Importance Sampling
	2. Adaptive Importance Sampling
	3. Target Sampling
	4. Noise Contrastive Estimation
	5. Negative Sampling

- **GloVe:** (global vector)
	- GloVe is an unsupervised learning algorithm for obtaining vector representations for words
	-  word vectors put words to a nice vector space, where similar words cluster together and different words repel
	-  The advantage of GloVe is that, unlike Word2vec, GloVe does not rely just on local statistics (local context information of words), but incorporates global statistics (word co-occurrence) to obtain word vectors.
	-  , Word2vec relies only on local information of language. That is, the semantics learnt for a given word, is only affected by the surrounding words.
	-  Word2vec which captures local statistics do very well in analogy tasks.
	-  Idea of GloVe: You can derive semantic relationships between words from the co-occurrence matrix
	-  
	




		
		
		
		
		
- 
- 
- 






























