from __future__ import print_function
import Model_3Dunet
from datetime import datetime
from DS import DS

from keras import backend as Keras
# initialization and prepare data set#########################################################
patch_size = [40, 40, 16]
batch_size = 16
epochs = 100
channel = 1
K = 5
angles = []
scales = []
g_path = r'C:\result\\' + datetime.now().strftime('%Y-%m-%d--%H-%M')




######################################
ds = DS('.\data\\', patch_size, channel, K, angles, scales,flip=True)
logger = ds.create_files(K,1,g_path)

####################################################################################################
#9###################################################################################################
scales = []
angles = []
logger.write("***************************************flipp*********************************************\n")
for fold in range(0, K):
    x_train, y_train, x_test, y_test, x_train2, y_train2, x_test2, y_test2 = ds.get_data(fold)
    logger.write('===================fold: ' + str(fold) + '===================\n')
    Keras.clear_session()
    model = Model_3Dunet.get_model(logger, 1, input_shape=(patch_size[0], patch_size[1], patch_size[2], channel))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                  validation_data=(x_test, y_test), verbose=2)
    pred = model.predict(x_test)
    pred2 = model.predict(x_test2)
    micro, macro, macro2 = DS.post_process(logger, y_test, pred, y_test2, pred2)
    logger.write('my method: ' + str(micro) + '  ' + str(macro) + '  ' + str(macro2) + '\n')
    logger.flush()
    del model
####################################################################################################
#10###################################################################################################
scales = [2,4,0.25,0.5]
logger.write("***************************************flip and 2,4,0.25,0.5*********************************************\n")
for fold in range(0, K):
    x_train, y_train, x_test, y_test, x_train2, y_train2, x_test2, y_test2 = ds.get_data(fold)
    logger.write('===================fold: ' + str(fold) + '===================\n')
    Keras.clear_session()
    model = Model_3Dunet.get_model(logger, 1, input_shape=(patch_size[0], patch_size[1], patch_size[2], channel))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                  validation_data=(x_test, y_test), verbose=2)
    pred = model.predict(x_test)
    pred2 = model.predict(x_test2)
    micro, macro, macro2 = DS.post_process(logger, y_test, pred, y_test2, pred2)
    logger.write('my method: ' + str(micro) + '  ' + str(macro) + '  ' + str(macro2) + '\n')
    logger.flush()
    del model
####################################################################################################
#11###################################################################################################
scales = [2,0.5]
angles = []
logger.write("***************************************flip and 2,0.5*********************************************\n")
for fold in range(0, K):
    x_train, y_train, x_test, y_test, x_train2, y_train2, x_test2, y_test2 = ds.get_data(fold)
    logger.write('===================fold: ' + str(fold) + '===================\n')
    Keras.clear_session()
    model = Model_3Dunet.get_model(logger, 1, input_shape=(patch_size[0], patch_size[1], patch_size[2], channel))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                  validation_data=(x_test, y_test), verbose=2)
    pred = model.predict(x_test)
    pred2 = model.predict(x_test2)
    micro, macro, macro2 = DS.post_process(logger, y_test, pred, y_test2, pred2)
    logger.write('my method: ' + str(micro) + '  ' + str(macro) + '  ' + str(macro2) + '\n')
    logger.flush()
    del model
####################################################################################################
#12###################################################################################################
scales = [4,0.25]
angles = []
logger.write("***************************************flip and 4,0.25*********************************************\n")
for fold in range(0, K):
    x_train, y_train, x_test, y_test, x_train2, y_train2, x_test2, y_test2 = ds.get_data(fold)
    logger.write('===================fold: ' + str(fold) + '===================\n')
    Keras.clear_session()
    model = Model_3Dunet.get_model(logger, 1, input_shape=(patch_size[0], patch_size[1], patch_size[2], channel))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                  validation_data=(x_test, y_test), verbose=2)
    pred = model.predict(x_test)
    pred2 = model.predict(x_test2)
    micro, macro, macro2 = DS.post_process(logger, y_test, pred, y_test2, pred2)
    logger.write('my method: ' + str(micro) + '  ' + str(macro) + '  ' + str(macro2) + '\n')
    logger.flush()
    del model
####################################################################################################