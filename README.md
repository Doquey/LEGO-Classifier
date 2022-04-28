# LEGO minifigures-Classifier

I took the dataset of LEGO images from this kaggle post: https://www.kaggle.com/datasets/ihelon/lego-minifigures-classification if you open the dataset you will see that it has one folder for harry-potter images, one folder for star-wars, one folder for jurassic-world and one folder for the marvel. In each of these folders there are other folders separating the images for characters of each movie/company. What I then did was: I went trough each 'superclass', marvel/harry-potter/star-wars/jurassic-world, then through all the characters folders inside each superclass and I took all of the images and sent them into a new random folder inside 'train'. 




# Train_gen and test_gen

Those are objects of the class keras.preprocessing.image.Imagedatagenerator(). I applied some data augmentation to the training images. Therefore the train_gen does augmentation and scaling to the images, and also it separates a part of the data to validation. The test_gen only scales the data.


#Trainig

I first set the train_batches and val_batches as subsets of the train_gen, both the train_batches and val_batches will receive data from our directory.

I used the Adam optmizer and the Sparse cross entropy loss, both from the keras API. I used a callback for earlystopping tracking the val_accuracy of the model.



