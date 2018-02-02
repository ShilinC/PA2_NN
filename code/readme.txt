Environment:
    Python 2.7

Configure network:
    nn = Network([784, 64, 10]) 

    # You can set the parameters of the network in initilization. THe parameters are as follows:
    layers
    init_method_weights 
    init_method_bias 
    init_method_delta_w 
    activation_fn
    learning_rate 
    momentum 
    epoches
    batch_size 
    nesterov_momentum 


Start training:
    nn.train(training_images, one_hot_train_labels, training_labels, test_images, one_hot_test_labels, test_labels, validation_images, validation_labels, one_hot_validation_labels)