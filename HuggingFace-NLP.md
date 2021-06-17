## HuggingFace  : NLP based startup
## [Course : NLP Link](https://huggingface.co/course/chapter1?fw=tf)
### Course Content:
  - Transformers
  - Datasets
  - Tokenizers
  - Accelerate
------------------------------------------------------------------------------------------------------------------
### Chapter:01
### What is NLP ?
- NLP is a field of linguistics and machine learning focused on understanding everything related to human language
- The aim of NLP tasks is not only to understand single words individually, but to be able to understand the context of those words.

### Chapter:01: Transformers
-  pip install "transformers[dev]"

-  **Pipeline tasks available:**
    - Text Classification
    - Zero- shot Learning
    - Text Generation
    - Text Completion
    - Token Classification
    - Question Answering
    - Summarization
    - Translation
   
   ### Text --> Preprocessing --> Model --> Post Processing  ---> Intelligible answer
   ### Zero-shot-claffication:--> label the string (i.e education, polictics) for which the sentence related to
   
### **How Transformers work?**
   #### **History**
    - The Transformer architecture was introduced in June 2017.
    -  June 2018, GPT, the first pretrained Transformer model, used for fine-tuning on various NLP tasks and obtained state-of-the-art results
    -  October 2018: BERT, another large pretrained model, this one designed to produce better summaries of sentences
    -  February 2019: GPT-2, an improved (and bigger) version of GPT that was not immediately publicly released due to ethical concerns
    -  October 2019: DistilBERT, a distilled version of BERT that is 60% faster, 40% lighter in memory, and still retains 97% of BERT’s performance
    -  October 2019: BART and T5, two large pretrained models using the same architecture as the original Transformer model (the first to do so)
    -  May 2020, GPT-3, an even bigger version of GPT-2 that is able to perform well on a variety of tasks without the need for fine-tuning (called zero-shot learning)
  
####This list is far from comprehensive, and is just meant to highlight a few of the different kinds of Transformer models. Broadly, they can be grouped into three categories:

1. GPT-like (also called auto-regressive Transformer models)
2. BERT-like (also called auto-encoding Transformer models)
3. BART/T5-like (also called sequence-to-sequence Transformer models)

 - **Transformers are language models which are trained on large text datasets in a self supervised fashion.**
 - **Self-supervised learning is a type of training in which the objective is automatically computed from the inputs of the model.
  That means that humans are not needed to label the data!**
 
 
 - **Transformers are big models**
 - **Pretraining is the act of training a model from scratch: the weights are randomly initialized, and the training starts without any prior knowledge.**
 - **Fine-tuning, on the other hand, is the training done after a model has been pretrained. To perform fine-tuning, you first acquire a pretrained language model,
 then perform additional training with a dataset specific to your task.**
 -  Since the pretrained model was already trained on lots of data, the fine-tuning requires way less data to get decent results.
 -  For the same reason, the amount of time and resources needed to get good results are much lower.
 -  The fine-tuning will only require a limited amount of data: the knowledge the pretrained model has acquired is “transferred,” hence the term transfer learning.
 
 ### **Transformer Architecture**
 - Composed of two blocks
 1. Encoder: The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
 2. Decoder: The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. 
 This means that the model is optimized for generating outputs.
 - ***Each of these parts can be used independently, depending on the task:***
 1. Encoder-only models: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
2. Decoder-only models: Good for generative tasks such as text generation.
3. Encoder-decoder models or sequence-to-sequence models: Good for generative tasks that require an input, such as translation or summarization.

### Attention Layer:
- A key feature of Transformer models is that they are built with special layers called attention layers.
-  this layer will tell the model to pay specific attention to certain words in the sentence you passed it (and more or less ignore the others) when dealing 
with the representation of each word.
### Keywords
- **Architecture**: This is the skeleton of the model — the definition of each layer and each operation that happens within the model.
- **Checkpoints**: These are the weights that will be loaded in a given architecture.
- Model: This is an umbrella term that isn’t as precise as “architecture” or “checkpoint”: it can mean both. This course will specify architecture or
checkpoint when it matters to reduce ambiguity.
 
## Encoder Models:
- Encoder models use only the encoder of a Transformer model. At each stage, the attention layers can access all the words in the initial sentence. 
These models are often characterized as having “bi-directional” attention, and are often called auto-encoding models.
- Encoder outputs numerical representation of each word in input (Feature Vector)
- **Each word affects the representation of other words in intial input** (CONTEXT)---self attention mechanism
- Bi-Directional model
- Good at extracting usefull information, NLU
- Good at MLM( mask language modelling-- guessing missing words)
- Encoder models are best suited for tasks requiring an understanding of the full sentence, such as sentence classification, named entity recognition 
(and more generally word classification), and extractive question answering
-  Example: BERT, ALBERT, DistilBERT, RoBERTa, ELECTRA

## Decoder Models:
-  It also creates a features vector of words from initial input
-  **Words can see the words of their left side --right side words are hidden!!**
-  These models are often called auto-regressive models
-  The pretraining of decoder models usually revolves around predicting the next word in the sentence.
-  Unidirectional , access to left or (right) context
-  Great at casual tasks : generating sequences
-  Natural Language Generation
-  Example: GPT-2, GPT neo, Transformer XL, GPT, CTRL

## Sequence to Sequence Models (Encoder-Decoder model:)
- Encoder generates feature vectors of each input word which is used as input for decoder
- Decoder takes some additional input(start input) with feature vector and outputs words
- Once encoder generates feature vector --- decoder start genetating word sequences on auto-regressive manner
- it means--> words generated by decoder are used as input to generate next words
-  encoder is discarded once it completes its works and decoder is used many times
-  Encoder - decoder do not share weights
-  Ecoder takes care understanding the sequence
-  Decoder takes care generating a sequence according to the understanding of the encoder.
-  input distribution is different to the output sequence
-  Use cases: Many to many , Translation, Summarization 
  #### Key Points:
  -  Encoder-decoder models (also called sequence-to-sequence models) use both parts of the Transformer architecture. At each stage,
the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the decoder can only access 
the words positioned before a given word in the input.
  - Examples : T5, BART, Marian, mBART
 
 
 ### Limitations of Pre-trained Lanuage models:
 -  Bias that are associated with the training Data
 
 
 -----------------------------------------------------------------------------------------------------------------------------
 ### Chapter:02
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
