# Author: Vahid Abrishami
from optparse import OptionParser
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os.path
from scipy import misc

# Keras Libraries for CNN
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint


class CytoToxAssay:
    def __init__(self):
        pass

    # To read parameters from the command line
    def define_params(self):
        desc = 'Please clean your root directory if you do not want to use previous data'
        self.parser = OptionParser(description=desc)
        self.parser.add_option("-i", "--input_csv",
                               action="store", dest="input_fname", metavar="FILE", type="string",
                               help="CSV file name with cropped cells information")
        self.parser.add_option("-o", "--out_root",
                               action="store", dest="output_directory", metavar="FILE", type="string",
                               help="Output directory to save intermediate and final results")
        self.parser.add_option("--model_fname",
                               action="store", dest="model_fname",
                               metavar="FILE", type="string",
                               help="Model file name on disk")
        self.parser.add_option("--image_size", action="store", nargs=2, type="int",
                               dest="image_size", help="Size of the image [height width]")
        self.parser.add_option("--batch_size", action="store", type="int", default=256,
                               dest="batch_size", help="Size of the batch for CNN training")
        self.parser.add_option("--epochs", action="store", type="int",
                               default=120, dest="epochs",
                               help="Number of times all of the training vectors are used to update the weights")
        self.parser.add_option("--kernel_size", action="store", type="int",
                               default=3, dest="kernel_size",
                               help="Size of the kernel for convolution")
        self.parser.add_option("--droput", action="store", type="float",
                               default=0.25, dest="dropout",
                               help="Randomly drop out a fraction  of input units to prevent overfitting")
        self.parser.add_option("--chunk_size", action="store", type="int",
                               default=50000, dest="chunk_size",
                               help="Number of samples to bring to memory RAM")
        self.parser.add_option('--save_int_results', action="store_true",
                               dest="save_results", default=False,
                               help='Set if you want to save intermediate results')
        self.parser.add_option('--net_tune', action="store_true",
                               dest="net_tune", default=False,
                               help='Set to true to fine tune the current network')
        self.parser.add_option("--checkp_interval",
                               action="store", dest="checkp_interval",
                               type="int", default=5,
                               help="Save the model after a number of epochs")
        self.parser.add_option('--plt_history', action="store_true",
                               dest="plt_history", default=False,
                               help='Set this flag to save accuracy and loss plots in the '
                                    'root directory')
        self.parser.add_option("-v", '--verbose', action="store_true", dest="verbose", default=False,
                               help='Set this flag to show more information for the classification')
        self.parser.add_option('--in_memory', action="store_true", dest="in_memory", default=False,
                               help='Set this flag to bring all the data into RAM memory')
        #self.parser.add_option('--layer_name', default='', dest="layer_name", type="string",
        #                       help='Name of first layer in the network that will not be freezed for fine tunning (only if net_tune==True)')

        self.parser.add_option('--path_column', default='PATH_Crop_Nuc', dest="path_column", type="string", help='Name of column in input table that stores the full paths for images')
        self.parser.add_option('--label_column', default='CLASS', dest="label_column", type="string", help='Name of column in input table that stores the label/annotation for images (HEALTHY or TOXICITY_AFFECTED only)')
    # To assign command line parameter to class variables
    def read_params(self):
        (options, args) = self.parser.parse_args()
        self.input_fname = options.input_fname
        self.out_dir = options.output_directory
        self.img_height = options.image_size[0]
        self.img_width = options.image_size[0]
        self.batch_size = options.batch_size
        self.epochs = options.epochs
        self.kernel_size = options.kernel_size
        self.dropout = options.dropout
        self.chunk_size = options.chunk_size
        self.checkp_interval = options.checkp_interval
        self.net_tune = options.net_tune
        self.save_results = options.save_results
        self.verbose = options.verbose
        self.in_memory = options.in_memory
        self.plt_history = options.plt_history
        self.model_fname = options.model_fname
        #self.layer_name = options.layer_name
        self.path_column = options.path_column
        self.label_column = options.label_column

    # To initialize some variables required by the class
    def produce_side_info(self):
        # To read the CSV file relate to assay
        if self.input_fname:
            self.dataFrame = pd.read_csv(self.input_fname)
            if not self.in_memory:  # If not in memory, then shuffle the data
                tmp_dataFrame = pd.read_csv(self.input_fname)
                tmp_dataFrame.drop('Unnamed: 0', axis=1, inplace=True)
                # To shuffle the rows of the data frame (This is useful when we train in disk)
                self.dataFrame = tmp_dataFrame.reindex(np.random.permutation(tmp_dataFrame.index)).reset_index(
                    drop=True)
                del tmp_dataFrame  # To free the memory of temporal data frame
        else:
            print('CSV file does not exist!!!')
            exit()
        self.trainX, self.trainY, self.dataX = [], [], []
        self.predict, self.bad_index = [], []
        self.train_flag = True

        # Parameters to save the intermediate results
        self.trainset_fname_inter = [os.path.join(self.out_dir, "trainx.npy"), os.path.join(self.out_dir, "trainy.npy")]
        self.dataset_fname_inter = [os.path.join(self.out_dir, "dataset.npy"), '']
        self.model_fname_inter = os.path.join(self.out_dir, "model.hdf5")
        self.prediction_fname_inter = os.path.join(self.out_dir, "prediction.npy")
        self.cache_dir = os.path.join(self.out_dir, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.plt_acc_fname = os.path.join(self.out_dir, "acc_val.png")
        self.plt_loss_fname = os.path.join(self.out_dir, "loss_val.png")
        base = os.path.basename(self.input_fname)
        self.output_csv = os.path.join(self.out_dir, base.rsplit('.', 1)[0]) + '_classified.csv'
        #self.output_xlsx = os.path.join(self.out_dir, base.rsplit('.', 1)[0]) + '_classified_perwell.xlsx'

    ''' Method to generate dataset from a CSV (XLSX) file
        This method can generate a dataset matrix in
        memory to train a dataset. Instead of the whole dataset
        one can bring data in memory in batches to avoid
        overflow.
    '''

    # To create a dataset from a CSV file
    def dataset_from_csv(self, phase='train', path_label_columns=None,
                         in_memory=True, chunk_size=1000):

        if path_label_columns is None:
            path_label_columns=[self.path_column, self.label_column]
        
        dataX, dataY, zero_image_index = [], [], []
        chunk_num = 0

        images_fnames = self.dataFrame[path_label_columns[0]]
        labels = self.dataFrame[path_label_columns[1]]
        
        labels = np.array((labels == 'HEALTHY').astype(int))  # To have 0/1 classes instead of string
        labels = np_utils.to_categorical(labels, 2)  # Convert to binary classes
        pbar = tqdm(total=len(images_fnames), ascii=True, desc="Adding images", dynamic_ncols=True)

        for idx, image_fname in enumerate(images_fnames, 1):
            pbar.update(1)
            if image_fname[0] == '\\':  # If image path begins with \ then add additional \
                image_fname = '\\' + image_fname
            # Check if image size is zero or doesn't exist
            if (not os.path.isabs(image_fname)):
                image_fname = os.path.abspath(image_fname)

            if  (not os.path.isfile(image_fname)) or (os.stat(image_fname).st_size == 0):
                zero_image_index.append(idx)
                print('Image '+image_fname+' does not exist')
                continue
            img = misc.imread(image_fname)
            img = img.astype(float)  # Convert image to float
            # Data standardization
            mean = np.mean(img)
            if mean == 0:  # If the mean is zero, remove it from dataset
                zero_image_index.append(idx)
                continue
            img -= mean
            img /= np.std(img)
            img = np.expand_dims(img, axis=0)
            dataX.append(img)
            dataY.append(labels[idx - 1])

            if not in_memory:  # Save chunks on disk instead of memory
                chunk_aux = int(idx / chunk_size)
                if (idx % chunk_size) == 0:
                    np.save(os.path.join(self.cache_dir, '{}x{:02d}.npy'.format(phase, chunk_aux)), np.array(dataX))
                    np.save(os.path.join(self.cache_dir, '{}y{:02d}.npy'.format(phase, chunk_aux)), np.array(dataY))
                    chunk_num += 1
                    dataX = []
                    dataY = []
                elif idx == len(images_fnames):
                    np.save(os.path.join(self.cache_dir, '{}x{:02d}.npy'.format(phase, chunk_aux + 1)), np.array(dataX))
                    np.save(os.path.join(self.cache_dir, '{}y{:02d}.npy'.format(phase, chunk_aux + 1)), np.array(dataY))
                    chunk_num += 1
                    pbar.close()
                    return chunk_num, np.array(zero_image_index)
        pbar.close()
        return np.array(dataX), np.array(dataY), np.array(zero_image_index)

    # Generate or load train/test dataset
    def load_generate_data_matrix(self, dataset_matrix_fname, phase=''):
        if os.path.isfile(dataset_matrix_fname[0]):  # Check if we have train set on disk
            if phase == 'train':
                print("Train set found in the root directory. Loading ...")
                self.trainX = np.load(dataset_matrix_fname[0])
                self.trainY = np.load(dataset_matrix_fname[1])
                if self.verbose:
                    print('\nShape of the training matrix:', np.shape(self.trainX))
            else:
                print("Test set found in the root directory. Loading ...")
                self.dataX = np.load(dataset_matrix_fname[0])
                if self.verbose:
                    print('\nShape of the test matrix:', np.shape(self.dataX))
            return True
        else:
            if phase == 'train':
                print("Generating the train dataset ...")
                self.trainX, self.trainY, self.bad_index = self.dataset_from_csv()
            else:
                print("Generating the test dataset ...")
                self.dataX, _, self.bad_index = self.dataset_from_csv(phase='test')
            return False

    # Initialize a classifier or load a classifier from disk
    def _initialize_classifier(self):

        # Define Keras structure for data augmentation
        self.dataGen = ImageDataGenerator(
            rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images
        # Configure check pointer to save the model each checkp_interval epoch/s
        self.checkPointer = ModelCheckpoint(os.path.join(self.out_dir, 'model-{epoch:02d}.hdf5'),
                                            monitor='val_loss', verbose=int(self.verbose),
                                            save_best_only=False, save_weights_only=False,
                                            mode='auto', period=self.checkp_interval)
        # Load the trained classifier if it is specified
        if self.model_fname:
            if os.path.isfile(self.model_fname):
                print('Loading the specified classifier ...')
                self.train_flag = self.net_tune
                self.model = load_model(self.model_fname)
                # self.model.summary()
                # print(self.model.layers[5].name)
                # if not self.layer_name=='' and self.net_tune:
                #     from vis.utils import utils
                #     print('Freezing layers before {} layer'.format(self.layer_name))
                #     layer_idx = utils.find_layer_idx(self.model, self.layer_name)
                #     for layer in self.model.layers[:layer_idx]:
                #         layer.trainable = False
                if self.net_tune:
                    print('Freezing first convolutional layers')
                    layer_idx = 6
                    for layer in self.model.layers[:layer_idx]:
                        layer.trainable = False

                if self.verbose:
                    self.model.summary()
                return
            else:
                print('The specified model does not exist on the disk!!!')
                exit()
        # Check if we have a model in root dir to continue
        if os.path.isfile(self.model_fname_inter):
            print('Loading the classifier from the root directory ...')
            self.train_flag = False
            self.model = load_model(self.model_fname_inter)  # Load the model
            if self.verbose:
                self.model.summary()
            return

        # Define different layers of the network
        self.model = Sequential()
        self.model.add(Conv2D(32, 3, padding='same',
                              input_shape=(1, self.img_height, self.img_height)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(self.dropout))
        self.model.add(Conv2D(64, 3, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(self.dropout))
        # To define fully connected layer
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2))
        self.model.add(Activation('softmax'))
        # To configure learning process
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta',
                           metrics=['accuracy'])

    # Method to train CNN classifier for data doesn't fit in memory
    def _train_cnn_in_disk(self):

        chunk_num, _ = self.dataset_from_csv(in_memory=False, phase='train', chunk_size=self.chunk_size)
        print('chunk_num is equal to:', chunk_num)
        for e in range(self.epochs):
            print('Epoch number: %d' % e)
            # This is to save model each checkp_interval epochs
            if (e + 1) % self.checkp_interval == 0:
                self.model.save(os.path.join(self.out_dir, 'model-{:03d}.hdf5'.format(e)))
            # To use different orders of chunks for each epoch
            index = np.random.choice(chunk_num, chunk_num, replace=False)
            print(index)
            for chunk in index:
                self.trainX = np.load(os.path.join(self.cache_dir, 'trainx{:02d}.npy'.format(chunk+1)))
                self.trainY = np.load(os.path.join(self.cache_dir, 'trainy{:02d}.npy'.format(chunk+1)))
                self.trainX = self.trainX.reshape(self.trainX.shape[0], 1, self.img_height, self.img_width)
                self.dataGen.fit(self.trainX)
                history = self.model.fit_generator(
                    self.dataGen.flow(self.trainX, self.trainY, batch_size=self.batch_size),
                    validation_data=self.dataGen.flow(self.trainX, self.trainY,
                                                      batch_size=self.batch_size),
                    validation_steps=len(self.trainX) / self.batch_size,
                    steps_per_epoch=len(self.trainX) / self.batch_size,
                    epochs=1, verbose=self.verbose)
        return history

    # To train the classifier in disk or memory
    def train_classifier(self, in_memory=True):

        if in_memory:  # If in_memory then load all the data into the RAM
            print('Loading data to memory ...')
            if not self.load_generate_data_matrix(self.trainset_fname_inter,
                                                  'train'):  # To load/generate train data matrix
                self.trainX = self.trainX.reshape(self.trainX.shape[0], 1, self.img_height, self.img_width)
                if self.save_results:
                    print('\nSaving train dataset ...')
                    np.save(self.trainset_fname_inter[0], self.trainX)  # save train data to disk
                    np.save(self.trainset_fname_inter[1], self.trainY)  # save train labels to disk
            self.dataGen.fit(self.trainX)
            history = self.model.fit_generator(
                self.dataGen.flow(self.trainX, self.trainY, batch_size=self.batch_size),
                validation_data=self.dataGen.flow(self.trainX, self.trainY,
                                                  batch_size=self.batch_size),
                validation_steps=len(self.trainX) / self.batch_size,
                steps_per_epoch=len(self.trainX) / self.batch_size,
                epochs=self.epochs, callbacks=[self.checkPointer],
                verbose=self.verbose)
        else:
            history = self._train_cnn_in_disk()
        if self.save_results:
            print('Saving model to disk ...')
            self.model.save(self.model_fname_inter)  # Save the model
            del self.trainX  # Free memory of trainX
            del self.trainY  # Free memory of trainY
        return history

    # Method to predict classes for a dataset matrix
    def class_prediction(self, in_memory=True):
        # First to check if we have prediction in root directory
        if os.path.isfile(self.prediction_fname_inter):
            print('Predictions found in the root directory. Loading ...')
            self.predict = np.load(self.prediction_fname_inter)
        else:
            if in_memory:
                # To generate test dataset for prediction
                if not self.load_generate_data_matrix(self.dataset_fname_inter, 'test'):
                    self.dataX = self.dataX.reshape(self.dataX.shape[0], 1, self.img_height, self.img_width)
                    if self.save_results:
                        print('\nSaving test dataset to the disk ...')
                        np.save(self.dataset_fname_inter[0], self.dataX)  # save test dataset to the disk
                self.predict = self.model.predict(self.dataX,
                                                  verbose=self.verbose)  # Predict the class for each row
            else:
                self.predict = np.empty((0, 2))
                chunk_num, self.bad_index = self.dataset_from_csv(self, in_memory=False, phase='test',
                                                                  chunk_size=self.chunk_size)
                for chunk in range(chunk_num):
                    self.dataX = np.load(os.path.join(self.cache_dir, 'testx{:02d}.npy'.format(chunk + 1)))
                    self.dataX = self.dataX.reshape(self.dataX.shape[0], 1, self.img_height, self.img_width)
                    self.predict = np.concatenate((self.predict, self.model.predict(self.dataX, verbose=self.verbose)),
                                                  axis=0)
                self.predict = np.array(self.predict)
            if self.save_results:
                print('Saving predictions to the disk ...')
                np.save(self.prediction_fname_inter, self.predict)  # save predictions to the disk
        if len(self.bad_index) > 0:
            print('We have some zero images')
            self.dataFrame.drop(self.dataFrame.index[self.bad_index], inplace=True)
        self.dataFrame['Probability_HEALTHY_py'] = pd.Series(self.predict[:, 1], index=self.dataFrame.index)
        self.dataFrame['Prediction_Probability_HEALTHY_py'] = pd.Series(1*(self.predict[:, 1]>=0.5), index=self.dataFrame.index)
        self.dataFrame.to_csv(self.output_csv)

    # Method to sort and group by well cells after classification
    # def gen_perwell_table(self):
        # self.dataFrame = pd.read_csv(self.output_csv)
        # test = self.dataFrame.groupby(['WellID', 'Row', 'Column', 'Line', 'Treatment', 'TreatmentConcentration',
                                       # 'Set', 'CLASS', 'RealSet', 'Plate']).agg({'Probability_HEALTHY_py':
            # {
                # 'Group_Count': 'count',
                # 'mean_Probability_HEALTHY_py': 'mean',
                # 'mean_Prediction_Probability_HEALTHY_py':
                    # lambda
                        # x: sum(
                        # x >= 0.5) / x.count()}})
        # test.columns = test.columns.droplevel(0)
        # test = test.reset_index()
        # writer = pd.ExcelWriter(self.output_xlsx, engine='xlsxwriter')
        # test.to_excel(writer, sheet_name='Sheet1')
        # writer.save()

    # Method to save figures for accuracy and loss values of the classifier
    def save_acc_loss_figures(self):
        # summarize history for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.plt_acc_fname)
        plt.close()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.plt_loss_fname)

    # Method to run the program
    def try_run(self):
        self.define_params()
        self.read_params()
        self.produce_side_info()
        self._initialize_classifier()
        if (self.train_flag):
            self.history = self.train_classifier(self.in_memory)
            if self.plt_history:
                self.save_acc_loss_figures()
        else:
            self.class_prediction(self.in_memory)
            #self.gen_perwell_table()


def main():
    cyto_obj = CytoToxAssay()
    cyto_obj.try_run()
    print('Program finished successfully.')

if __name__ == "__main__":
    main()
