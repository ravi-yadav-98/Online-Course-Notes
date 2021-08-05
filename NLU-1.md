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






