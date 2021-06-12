# Online Courses Notes
## Advance Deep Learning For Computer Vision
### [YouTube Link:](https://youtu.be/utfM_XK7n_M)
### [Course Website :](https://dvl.in.tum.de/teaching/adl4cv-ss20/)
-----------------------------------------------------------------
### Date : 05 June 2020

## **Lecture-01 ADL4CV - Visualization and Interpretability**
 - Visualization is a great way of debugging CNN (weights, gradients)
 - visualize feature maps
 - lower layers: detect low level features (edges, lines,shape)
 - Deeper layers : detect high level features
 - The Occlusion experiment: Occlusion of image changes classification score
 - 1.  **DeconvNet**: visualize activations by maping activations back to image space
 
### ConvNet: Image ----> Feature representation
### DeconvNet: Feature representation ---> Image
- Unpooling: Lower to higher dimension(2x2 to 4x4)
- Transposed ConvNet kernals(filters) are used for deconv and unpoorling
- Inverting ReLu for Deconv 
(Paper: striving for simplicity: the all convolution net)
- 2. **Gradient Ascent** :Generates a synthetic image that maximally activates filters
-  GA: want to find an image that maximizes the score of perticular classification
- [Online CNN visualize](http://yosinski.com/deepvis)
-  ###  **Deep Dream**: Amplify the feature activations at some layer in network
       - Over-processed images
       - Initially it was invented to help scientists and engineers to see what a deep neural network is seeing when it is        looking in a given image
       - 
#### steps
. Feed an image to CNN network
. Choose a layer and ask the network to enhance whatever was detected: if you see 
dogs show me more dogs.
. change the image little bit

### **t-SNE** - Visualization technique for high dimension data into low dimension:
-  map high-dimension embedding to 2D map which preserve the pairwise distance of points
-  Example: t-SNE visualize MNSIT data, ImageNet
-  similar data points are clustered together
-  shapeNet -  checkout

_______________________________________________________________________
## **Lecture:02 Siamese NN andd Similarity Learning**
### Decreasing Order of paremeters in Following :
- VGG-19 ---155M
- VGG-16
- ResNet152
- Inception-v4
- ResNet101
- Inception-v3
- ResNet34
- ResNet18
- GoogleNet
- ENet
- BN-Net
- BN-AlexNet
- AlexNet

--------------------------
Neural Network:
Image --> Pretrained model (Feature extracter)---> Embedding ---> Fully connected layer -->> Predictions
. regression Problem: --> Bounding Boxx Regression
. Paper: Learning to track 100 FPS with Deep Regression Network
. Siamese Network: for comparing two images 
### Comparison Problems with deep Learning:
-  **Similarity Learning Application**: face Unloacking  of Phone,Face recognition based attendence
-  Issues with the NN classification problem: 
      - fixed no of input/output neurons
      - Can't remore/add class on trained models
      -  will require to re-train the model if any new class is added.---Scalability Issues
      - So here Similarity learning comes into picture:

### Similarity Learning : find the distance between two images d(imageA, imageB)
-  if d is less ---same person
-  if d is large : Not same person

#### **Question: How to train neural network to learn the similarity 

- Proess the two images A and B using same way using CNN  --- compressed images to d-length encoded vector
-  Siamese Network - shared weights
-  compare the encoded images vecA and VecB 
- if the encodings are same that mean d(A,B) is small---same person 
- Distance function : used L2 norm , hinge loss (less computation)
- **constrative loss**: combine positive and negative losses (L2 and hinge loss) y=1 for positive , y=0 for negative
    - l(A,B) = y*(||F(A)-F(B)||^2)+ (1-y)*max(0, m^2- (||F(A)-F(B)||^2))
    -  loss function helps us to train to bring positive pairs closer and negative pairs apart
-  Triplet loss: allow us to Learn ranking - how much similarity is there between images 
    -  three image : Anchor(A), Positve(P), Negative(N)
    -  positive and anchor images are same images
    -  we want d(A,P) << d(A,N)  
    - add margin m to make it little flexible
    - Hard negative mining: Training NN with hard cases i.e. distance between d(A,P) ~= d(A,N)
    - test : Do nearest Neibhour Search : give Query images and retrive results which are similar in ranking order
    - Challenges of triplet loss:
                   - network needs to be trained for long time O(n^3)
                   -  stuck in local minima even after hard negative Training

### Approaches to improve similarity learning:
-  1.Sampling: Choose Best triplets for trainig +  hard samples + diversity
-  2. Ensembles: using several networks on different triplets
-  3. Improve the loss function further
 #### Sampling:
- **Hierarchial triplet loss** : made a Hierarchial tree with leaves of tree are image classes
- merge them untill reach the root based on the similarity

### **Application of Siamese Network for similarity Learning**
  - Clustering
  - Image Coresepondence (Obect detect, 3D construction, Tracking, Image retrival, Image Aligment)
  - Unsupervised Learning : (Track object in video)
  - Optical flow: Apparant flow of object motion (pattern)- Displacement of pixels in consecutive frames
  - 
--------------------------------------------------------------------------------------------------
## **Lecture :03 Autoencoders , VAE, and style transfer**

### **Autoencoders**: 
   - Unsupervised Approach for learning a lower dimension feature representation frpm unlabelled training data
   - Encoder and Decoder 
   - Encoder: Higher dimension to lower Dimension( Bottleneck Layer)- feature vector
   - Decoder: use low dimension feature vector to reconstruct the input image
   -  Goal of Autoencoder learning to minimize the reconstruction loss
   - Applications of Autoencoders:
        1. Pre-training: i.e Medical CT images as these images are very different from imagenet data, we can't use
        pre-trained model on imagenet dataset.
        2. Train it on large amount of unlabelled images and then supervised training 
        3. get pixel wise prediction : Segementation 
        ### **Sementic Segementation**
      -**SegNet**: for  Semantic Segementation 
        - Encoder-decoder based segmentation method
        - 1. Encoder: Convolution and max polling performed. (2x2- max pooling on Vgg16)
        - 2. Decoder: Upsampling and convolution is performed : max pooling is recalled (Unpooling)- Transposed Convolution
        - last layer: K-Class softmax to predict class of pixel
        - Types of Upsampling: Instead of filling zero while unpooling
           -  Interpolation i.e nearest neibhour interpolation (fill value of nearest), few artifact
           -  Fix unpooling : unpooling + Convolution
           -  Unpooling : Deconvolution: Keep the locations where max came from : convolve with transposed filters
    #### U-Net: skip connections- a CNN for biomedical image segmentation:
     - Bolttlenect layer: contains high level information of image in compressed manner which is used by decoder to reconstruct the image again
     -  Goal is to resonstruct the image with max possible imformation
     - Low level information is not provided to the decoder that's why quality of constructed image is not good
     -  Low level information is lost while convoution and only high level features are carried by bottleneck layer
     - **Decoder need both high and low level information to reconstruct good quality image**: 
     -  In U-Net skip connections combine both low and high level information feature map and : pass low level information to decoder
     - i.e ResNet:  similar intuition to pass information directly through residual connections:
     - ![image](https://user-images.githubusercontent.com/85448160/121581189-31b16780-ca4b-11eb-9f37-ec772008751b.png)
     - **skip connections: skip some layer in the neural network and feeds the output of one layer as the input to the next layers**
     #### **Ways to do skip connections:
              - 1. Additions : as in ResNet : for avoiding gradient vanishiing and passing features from lower layers to upper layers 
              - 2. Concatenation : DenseNet: Feature resusability by concatenating it to high level features 
        - Long and Short skip connections
        - Short skip connection: (Resnet)
        - Long Skip Connections: i.e Autoencoders
        - **U-Net: Encoder-Decoder + Skip Connections :
        -  **Using Skip Connections fine-gained informations/details  can be recovered in the decoder prediction: 
        -  Used mainly where output spatial dimension is same as input ans symmetrical architecture
        - **long skip connections are used to pass features from the encoder path to the decoder path in order to recover spatial information lost during downsampling**
        - Short skip connections appear to stabilize gradient updates in deep architectures. Finally, skip connections enable feature reusability and stabilize training and convergence.
 #### **Autoincoders in Vision**:
 -  SegNet: semantic Segmentation
 -  Monocular Depth Prediction: (Generally we need two cameras for estimating depths of object(two eyes to trianglate the depth)
 -  Image Super Resolution:->

### **Generative Models**:  Given training data-> Generate new samples from same distribution
- Types of Geneative Models: 1. Explicit density (i.e VAE) 2. Implicit Density (GAN)
### ** Variational Autoencoder**: when we want to sample from bottleneck layer
- Sample from latent distribution to gerenate new sample output
- in VAE : latent distribution is gaussian
- Encoder: convert images to a bottleneck latent gaussian space
- Bottleneck space: Gaussian -> a sample z is passed to autoecoder for geneating image

### Image Synthesis ( Without GANs)
- Semantic Segmentation Image --->> Real Image
- Use Perceptual loss for high quality results:
- Perceptual loss/content loss: measure content of image
- compare the ground truth and constructed image  1. Feature map of generated image  2. feature map of ground truth image  --L2 loss
- Neural Network sees both black car and white car same way as they are same simentaically : so feature maps will be silimar
- **So Do not compare pixels but compare feature maps**
- so Use VGG pretrained model for feature map representation and compare the maps for content loss
- Content loss was originally not used for image synthesis but for Style transfer

### ** Style Transfer**:
 - 1. Content Image   2. Style Image
 - create a new image with help of both of above
 - Content loss: Feature Representation similarity
 - Style Loss: Compare gram metrices : Preserves the stylic features not content
 - slow 

#### Fast style transfer : use neural network


-----------------------------------------------------------------------------------------------------------------
## **Lecture: 04- Graph Neural Networks and Attention**
- Regularity on the domain: i.e in images order of pixel is important
- New Domain: Graph i.e Point Colud (3D): Irregularity
- Permutation Invariance
- Transformation Invariance
- Example: A citation Network: where each node is a paper and connection(edge) is citation
- Other example: Social Network, Recommendation Systems
- we can not apply convolutions to graphs
### Deep learning on Graphs: 
-Challenges:
         - Variable size input (irregular shape)

-  Node and edge embeddings are needed.
-  Goal is to encode contexual graph information in node embeddings by iteratively combining neighboring features
- Embeddings are updated in hidden layers
### Neural message passing:
   - at each iteration,every node receives features from its neighboring nodes
   - these featres are then aggregated with an order invariant operation and combined with the current feature with learnable function
   - to apply convolution in graphs ---use permutation invariance
 - mostly node embeddings are updated.
 - 
Applications:
  - 1. Multi-Object tracking with Graphs
  - 2. 
### Sequence to sequence (seq2Seq) Learning:
### Transformers : use attention only (No RNN, NO CNN, Only Attention)
- Graphs that show relationship b/w words
- Transformers as Graphs NN
- Transformer are based on Graph attention networks
- Attention for Vision: -->  We use whole image to make classifictaion
             - Not all pixels are equally important
             - only object pixels are important
             - so it will be efficient if use patches for classification
             
 #### Image Captioning : Input : Image, Output: A sentence describing image
 -  Encoder: computes the feature maps,- VGG Net, Alex Net
 -  Decoder: An attention based RNN
 -  In each step  the decoder computes an attention map over the entire image, effectly decide which region to focus on.
 -  It receives a context vector, which is the weighted average of the conv net feature
 -  Convention Captioning: Encoder: CNN, Decoder: LSTM(sees image once)
 -  Attention based captioning: Decoder uses attention- look only those regions of image which are important for captioning.
 -  Type of attention:
 -  soft attention: determine processes which can be backprop- deterministic-- focuses on object
 -  hard attention: stochastic process, gradient is estimated through Monte carlo sampling
 -   soft attention is widely used in optimization
 -   
 -   


-  

 
