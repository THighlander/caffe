
#OaA
This repo is devoted to adding the overlap and add technique to GPU convolutions in CAFFE. This was accomplished by adding a new layer to CAFFE. “ConvolutionOaA” will use fbFFT to compute convolutions using the overlap-and-add technique in CAFFE. The layer uses the same parameters as the traditional Convolution layer. Limited testing has been done, but OaA seems to benefit larger layers with mid sized (8-16) kernels. For a single convolution layer with 100 inputs of size 256 x 256 and a kernel of size 16 x 16 traditional CAFFE takes 63.4 seconds on a NVIDIA TITAN X to compute 100 forward propagations while ConvolutionOaA takes 11.6 seconds. The current implementation requires a computation complexity at or above 3.5, and a max kernel size of 16 x 16. Future work would include polishing the code for better performance and allowing for larger than 16x16 kernels. If this code is used for your research please cite our [BMVC paper] (http://bmvc2015.swansea.ac.uk/proceedings/papers/paper160/paper160.pdf) , and [fbFFT] (http://arxiv.org/pdf/1412.7580v3.pdf).

# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
