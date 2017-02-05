from datasets import FER2013
from models import FCN12, ResNet20
import os

def main():
    dataset = FER2013(data_directory="data/fer2013", target_size=(48, 48))
    if not os.path.exists(dataset.train_directory):
        dataset.split()

    train_generator = dataset.train_generator(batch_size=32, shear_range=0.2, zoom_range=0.2, width_shift_range=.1, height_shift_range=.1, horizontal_flip=True, fill_mode="wrap")
    test_generator = dataset.test_generator(batch_size=32)
    valid_generator = dataset.valid_generator(batch_size=32)

    classifier = ResNet20(input_shape=dataset.target_size)
    if os.path.exists("models/resnet20.h5"):
        classifier.load("models/resnet20.h5")
    
    #while True:
    #	classifier.fit_generator(train_generator, test_generator=None, nb_epoch=1)
    #	classifier.save("models/resnet20.h5")
    print classifier.model.evaluate_generator(valid_generator, val_samples=3589)

if __name__ == "__main__":
    main()

