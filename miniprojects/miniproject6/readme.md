Mini Project 6 - Due on 3/24/2017
---------------------------------

This is an independent project. This project is in three parts. In the first part you will simply 
run the [Autoencoder Tutorial](http://yann.readthedocs.io/en/master/pantry/tutorials/autoencoder.html).
Run this tutorial with varying length of codewords and varying number of layers. Summarize your 
findings with appropriate figures of the filters you learnt.

In the second part you will simply 
run the [Generative Adversarial Network Tutorial](http://yann.readthedocs.io/en/master/pantry/tutorials/gan.html).
Run this tutorial with varying number of layers and types of layers in both the discriminator and 
the generator. Summarize your 
findings with appropriate figures of the filters you learnt. Be sure to simulate a situation where
your GAN will mode collapse.

In the yann toolbox, you can create a variety of layers. One type of layer is the 
[``random`` layer](http://yann.readthedocs.io/en/master/yann/layers/random.html) and another is the 
[``merge`` layer](http://yann.readthedocs.io/en/master/yann/layers/merge.html). Refer to the 
[``add_layer``](http://yann.readthedocs.io/en/master/yann/network.html#yann.network.network.add_layer)
method for parameters and how to add these layers to the network module. ``merge`` layer can take 
an argument ``layer_type``. If you provide this argument with ``sum``, the 
output for the ``merge`` layer would be the sum of the two input layers supplied. 

Using these layers and the MNIST dataset cooked during the autoencoder tutorial, create a layer 
that will produce a noisy version of the image. De-noise the image using a denoising autoencoder 
setup like we studied in the class. Attempt this for varying over-complete encoders and 
depth of network. Summarize your findings with appropriate figures of the filters you learnt. 

The submission for this project is a three-page report. The three-page report will
be typeset in the [camera-ready](https://www.computer.org/web/tpami/author)
style of IEEE TPAMI. The report should contain detailed analysis of your reporting along with a 
table of all generalities (Essentially 3X3). Using these values determine which is the most 
general of all datasets.


Installation
------------

For the yann toolbox installation and other setup details refer the 
[yann toolbox documentation](http://www.yann.network).

Academic Integrity
------------------

You are expected to maintain the utmost level of academic integrity in the project. Any violation 
of the code of academic integrity will be reported to the dean for official actions. It is an 
academic violation to copy, to include text from other sources, including online sources for both
material and code, without proper citation and licensing. To get a better idea of what constitutes 
plagiarism, consult the 
[ASU policy on student obligations](https://provost.asu.edu/academic-integrity). 
This is a serious violation and evidence of plagiarism or academic dishonesty, will likely result
in failing the course and at worse can lead to disqualification from your degree program. Please 
contact the instructor before borrowing material when unsure. Your code might be run through a 
plagiarism checker.

Even when using material from sources with appropriate citations and licensing, be aware of 
reasonableness and of the areas of the project for which you are borrowing the code or materials. 
The \emph{core} of the project is expected to be implemented by you, the student and borrowing 
code or material for the central objective of the project, even though explicitly allowed for 
supporting and auxiliary purposes will be considered unreasonable and dishonest.  Please consult 
the instructor before borrowing material when unsure. Note that your code might be run through a 
plagiarism checker for inspection. 