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
	-  Corpus(corpora) to Word vectors
	-  work on probability model (predicting words around a center word)
	-  start with random vectors
	-  Distribution vector means: word vector know which word occur in context of itself(so it wil give high probablity)
	-  Like if word is bank the  words like, branch, withdrawal, credit, debit , will be given high probability
	-  Shallow 2 layer NN
	-  Two architectures of Word2Vec:
	1. CBOW(continious Bag of words)--> predict current word given conext words
	2. Skip gram  --> predict context words given current word
	-  **The basic idea of word embedding is words that occur in similar context tend to be closer to each other in vector space.**
	-  king-men+women= Queen


### Lecture:02
-  
		
		
		
		
		
- 
- 
- 






























