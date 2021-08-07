### Yotube channel : From Language to information
- [link](https://www.youtube.com/channel/UC_48v322owNVtORXuMeRmpA)
### Week 1-3
1. **Regular Expressions Regex**
  - disjunction []
  - [A-Z] : all capital alphabet included
  - [a-z] : all small letters included
  - [abc] : only a, b, c included
  - [a-d] : a,b,c,d letters included
  - [0-9] : 0 to 9 all digit included
  - [^a-z] : a to z excluded
  - [^char] : character excluded
  - pip | disjunction  --> or 
  - a|b|c : a or b or c --> similar to [abc] (can be combined)
  - **special characters**
  1. ? : optional previous character : colou?r -> color, colour
  2. * : 0 or more previous character: oo*h! -> oh!, ooh!, oooh!, 
  3. + : 1 or moreprevios char : o+h! -> oh!, ooh!, oooh!, ooooh!
  4. ba++ -> ba, baa baaaa, (one or more a)
  5. . : any character i.e beg.n -> begin, began   
  6. ^ : begining of the line
  7. $ : end of line
  8. \.$ -> full stop  at end of line
  
  - **Error in string matching**
  1. False Positive (matching strings we should not have matched)->type I
  2. False Negative (Not matching things that we should have matched)-> type II
- Ex: find all the in the string 
- [^A-Za-z][Tt]he[^A-Za-z]

- **Substitutions**
  - s/ : substitute 
  - () : capture a pattern
  - 
2. **Words and Corpora**
- Lemma : same stem , part of speech i.e cat and cats = same lemma
- wordform : the full inflected surface form
- cat and cats : two wordform but same lemma
- In corpora: N= no of tokens and V= vocabulary size (type)
- token : count each instance of word
- type: count single instance of same word
- V = kN^b (.67<b<.75) Heap's Law or Herdan law
- Example: Google n-gram: (1 trillion token and 13 million type(vocab)
- total 7097 languages in the word
3. **Word Tokebization**
  - Every NLP task require text normalization
  - i.e Tokenizing, Normalizing , 
  - space-based tokenization
  - Issue in Tokenization
  - Can not blindly remove punctuations(Ph.d, Urls, email adresss, dates)
  - Clitic : word that doesn't stand on its own i.e are in We're
  - Multi word (New York)
  - many languages don't have spaces(chinese, japanese) each char as token

3. **Byte pair encoding**
  - Other than space based and single character segementation
  - Use data to tell us how to tokenize
  - **subword tokenization** Three common algos
  - 1. Byte Pair Encoding (BPE)
  - 2. Unigram language modeling tokenization
  - 3. WordPience
  - all have two parts that takes a raw training corpora and induces a vocab (tokens)
  - A token Segenter that takes a raw test sentence and tokenize it acc to vocab
 4. **Word Normalization*
  - putting words/tokens in a standard format
  - i.e U.S.A = USA
  - use all letters to lower case
  - Lemmatization (root word)-> dictionary headform 
  - am, are, is-> be
  - cars, car, car's = car
  - Morphemes: The small meaningful units that make up words
  - Stems: The core meaninf -bearing units
  - Affixes: Parts that adhere to stems 
  - Morphological parsers
  - cats = cat+s
  - steming = chooping off affixes
  - Portar stemmer:  based on a series of rewrite rules run in series
  - Sentence Segementation (end of sentence)-> ? or . or ! (sentence boundary)
5. **Minimum edit distance**
  - how similar are two strings?
  - The minimum edit distance b/w two srings needed to transform one into other
  - i.e Insertion, deketion, substitution
  - use in machine translation and speech recognition
  
 6. Dynamic programming for minimum distance:
  - tabular computation D(n, m)
  - n and m are string lengths
  - Backtracing
  - Weighted Edit Distance
 
 
 ### Week 2
 - **Text Classification and Naive Bayes**
  - It is assigning subject categories , topics or genres to text
  - example 1: email spam /non-spam,  arthor of paper
  - example 2: Positive pr negative movie review (or product review)
  - Example 3: subject of article (topic)
  - Classification Methods: 
  - 1. Hand-Coded rules (i.e rule based spam detection)i.e blacklist words
      - accuracy can be high
      - if rules are defined by expert
      - expensive
  - 2. Supervised Machine Learning
      - ML classifier
      - training data is used (document , class)--> (d1,c1) (d2,c3)
      - Naive Bayes, Logistic, SVM, KNN
  - ** The Naive Bayes Classifier**
      - based or bayes rule
      - used bag or words (BOG)- document --> words --> mixed
      - word --> count in document 
      - classifier take BOG and predict class
      - Y(words,count)--> class
      - goal is to find p(c/d) = P(d/c)*P(c)/P(d)
      - Cout = argmax(P(c/d) of all vaues  ---> maximizing posteriori
      - P(d) is same for all classes
      - class  = P(x1,x2,x3....xn/c)P(c)
      - x1,x2,x3.....xn = words
      - BOG Assumptions: position doesn't matter
      - featue probabilities P(x/c) is independent given class
      - so P(x1,x2...xn/c) = P(x1/c)*P(x2/c).......P(xn/c)
      - P(x/c) = prob of word given class 
      - i.e how ofter word is present in given class
      - Problem: multiplying lots of probabilities can be floating point underflow. so log is used
      - Linear classifier
    
  - **Learning the Multinomial naive bayes model**
      - 1. Maximum likelihood estimates (MLE)
      -  simply use frequencies in the data (P= document(C= cj)/total doc)
      -  parameter estimate:
      - Problem with MLE: if perticular word was not in the training (gives zero probability) 
      - solution : add 1 smoothing ( add 1 to all word count and each word count)
      - unknown words which appear in test data(not in training) : we ignore
      - stop words: frequent words: the, an , a : removed(top frequency words)


  - **sentiment analysis** using binary naive bayes
      - prior = class probability in document = total no of negative/positive doc / total doc 
      - for sentiment analysis word occurnce seems to be more important that frequency
      - Multinomial naive bayes: binary NB
      - 


      - 
  - 






















 - Name entity recongition[NER]: proper name
 - 4 tags : PER(person), LOC(location), ORG(organization), GPE(geo political entity)
 - also tags inclued : dates, price
 - NER is hard due to 
 - 1. Segementation
 - 2. Type ambiguity (person name can be same as organization name)
 - BIO Tagging: one label per word
 - 
  - 
  - 
  - 
  - 
  - 






