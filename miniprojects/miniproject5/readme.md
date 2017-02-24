Mini Project 5 - Due on 3/17/2017
---------------------------------

This is an independent project.  

In the yann toolbox, you can supply initial parameters to the ``add_layer`` method using the 
argument ``input_params``. It can also take the argument ``learnable`` which is ``True`` by default.
If ``learnable`` was ``False``, those parameters would be considered as frozen and will not be 
updated during backprop. Using these two options, and the three datasets provided through the 
generators in the ``datasets.py`` you will study the generality of these datasets on the softmax
layer.

To save a network's parameters down, you may use the following commands:

    from yann.utils.pickle import pickle
    pickle(net, 'network.pkl)

To load the network parameters as a dictionary you can run the following command:

    from yann.utils.pickle import load
    params = load('network.pkl')

The ``params`` is now a dictionary with the keys being the layer ids and each value being the list 
of parameters that could be supplied to the ``add_layer`` method. 

    
Using all these tools you will perform the following generality experiments:

1. Train a network (on a base dataset (one of the three).
2. Save the network down and note its performance down as ![Base Performance](https://latex.codecogs.com/gif.latex?%24%5CPsi%28D_i%7Cr%29%24)

This can be done for all the three datasets independently and the networks saved down.
For some new dataset,
3. Load the parameters of a base and create a new network which uses the parameters of a network trained 
    on the base. 
4. Setit up with all the layers but the softmax layer frozen (``learnable = False``).
4. Train only the softmax layer for this network on a re-train dataset (one of the remaining 
    two). 
5. Note the performance down ![Re train performance](https://latex.codecogs.com/gif.latex?%24%5CPsi%28D_j%7CD_i%29%24).

The generality of the dataset the first dataset with respect to the second is,

![Generality](https://latex.codecogs.com/gif.latex?g%28D_i%20%5Cvert%20D_j%29%20%3D%20%5Cfrac%7B%5Cpsi%28D_j%20%5Cvert%20D_i%29%7D%7B%5Cpsi%28D_j%20%5Cvert%20r%29%7D)

Be weary of the notations in the above equation.

 
Using this technique measure dataset generality of all the dataset with respect to all the other datasets
among those provided in the ``datasets.py`` file. The network you will use to train them is a two
convolutional, two dense layer network. The first layer of the network has 20 neurons, of 5X5 with a 
pooling of 2X2. The second is a 50 neurons of 3X3 with a pooling of 2X2. The third and fourth are 
dot product layers with 800 nodes each and ``dropout_rate = 0.5``. Apply a $L1$ and $L2$ 
co-efficient on all layers of $0.0001$. Use RMSPROP with Nesterov momentum. The other properties
are upto you. 

The submission for this project is an one-page report. The one-page report will
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