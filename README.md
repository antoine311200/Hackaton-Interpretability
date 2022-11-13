<center><h1>Alignment Jam : The Interpretability Hackathon</h1></center>

### Description 

Here is my github repository containing all codes that I used.
The two main files are :
- standard_neural_network.ipynb
- tensorised_neural_network.ipynb.ipynb

To run them, simply create a venv with the all requirements.txt packages installed and run them linearly!

I provided most of the papers that I used in the folder Papers and the images of the report as well as the report in itself.

For the curious reviewer, the misc folder contains what I did during the first 24h that did not prove to be doable in 40h as it is a massive load of works to make Riemannian optmisation work with a custom made library of Tensor Networks.

<center>

![Neural Network GradCAM](https://raw.githubusercontent.com/antoine311200/Hackaton-Interpretability/master/Images/nn_grad.png "Neural Network GradCAM")
![Tensorised Neural Network GradCAM](https://raw.githubusercontent.com/antoine311200/Hackaton-Interpretability/master/Images/tn_grad.png "Tensorised Neural Network GradCAM")

</center>

<center>

![Neural Network SmoothGrad](https://raw.githubusercontent.com/antoine311200/Hackaton-Interpretability/master/Images/nn_sal.png "Neural Network SmoothGrad")
![Tensorised Neural Network SmoothGrad](https://raw.githubusercontent.com/antoine311200/Hackaton-Interpretability/master/Images/tn_sal.png "Tensorised Neural Network SmoothGrad")

</center>

---

### Introduction of the papers

In the thriving world of physics and quantum computing, researchers have elaborated a multitude
of methods to simulate complex quantum systems since the 90s. Among these techniques, tensor
networks have an important role in the way they make sense of the compression of data to fit
complex quantum systems in the memory of classical computers. The rapid expansion of tensor
networks is due to the need to visualize and store physical structures.
In this paper, a tensor train decomposition of the linear layers of a simple Convolutional Neural
Network has been implemented and trained on the dataset Cifar10. The observations show that
the various attention images inferred on both a neural network and its tensor network equivalent
has radically different and the models focus on different parts. Secondly, I proposed some considerations on miscellaneous gradient descent methods that can be used to specifically optimise tensor
networks. Tensor networks evolve in a smooth Riemannian manifold, using Riemannian optimisation (RO) techniques to perform gradient descent geometrically. Indeed, the projections implied by
the RO allow the traceability of the gradients and thus, easily reverse engineer a backpropagation.

### Table of contents

    1 Tensor train decomposition & Matrix Product States 2


    2 Safety & Alignment 3


    3 Gradient-based Visualisation methods 4

        3.1 Interpretability visualisation Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

        3.2 Results - GradCAM . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

        3.3 Results - SmoothGrad . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

        3.4 Results - Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6


    4 Future perspectives 6

        4.1 DMRG-like optimisation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

        4.2 Tangent Space Gradient optimisation . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

        4.3 Riemanian optimisation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7


    5 Acknowledgements 8