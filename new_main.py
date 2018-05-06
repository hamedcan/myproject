from __future__ import print_function
import Model_rnd_norm
import Model_rnd_uni
import Model_glr_norm
import Model_glr_uniform
from Init_data_gabor import InitData
import Model_gabor
import Model_CLAHE
import Model_histEQU
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
scales = [0.5]
g_path = r'C:\result\\' + datetime.now().strftime('%Y-%m-%d--%H-%M')

ds = DS('.\data\\', patch_size, channel, K, angles, scales)
logger = ds.create_files(K, 1, g_path)


logger.write("****************************************Model_gabor*********************************************\n")
for fold in range(0, K):
    x_train, y_train, x_test, y_test, x_train2, y_train2, x_test2, y_test2 = ds.get_data(fold)
    logger.write('===================fold: ' + str(fold) + '===================\n')
    Keras.clear_session()
    model = Model_gabor.get_model(logger, 1, input_shape=(patch_size[0], patch_size[1], patch_size[2], channel))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                  validation_data=(x_test, y_test), verbose=2)
    pred = model.predict(x_test)
    pred2 = model.predict(x_test2)
    micro, macro, macro2 = DS.post_process(logger, y_test, pred, y_test2, pred2)
    logger.write('my method: ' + str(micro) + '  ' + str(macro) + '  ' + str(macro2) + '\n')
    logger.flush()
    del model
####################################################################################################
#6###################################################################################################
logger.write("****************************************Model_CLAHE*********************************************\n")
for fold in range(0, K):
    x_train, y_train, x_test, y_test, x_train2, y_train2, x_test2, y_test2 = ds.get_data(fold)
    logger.write('===================fold: ' + str(fold) + '===================\n')
    Keras.clear_session()
    model = Model_CLAHE.get_model(logger, 1, input_shape=(patch_size[0], patch_size[1], patch_size[2], channel))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                  validation_data=(x_test, y_test), verbose=2)
    pred = model.predict(x_test)
    pred2 = model.predict(x_test2)
    micro, macro, macro2 = DS.post_process(logger, y_test, pred, y_test2, pred2)
    logger.write('my method: ' + str(micro) + '  ' + str(macro) + '  ' + str(macro2) + '\n')
    logger.flush()
    del model
####################################################################################################
#7###################################################################################################
logger.write("****************************************Model_histEQU*********************************************\n")
for fold in range(0, K):
    x_train, y_train, x_test, y_test, x_train2, y_train2, x_test2, y_test2 = ds.get_data(fold)
    logger.write('===================fold: ' + str(fold) + '===================\n')
    Keras.clear_session()
    model = Model_histEQU.get_model(logger, 1, input_shape=(patch_size[0], patch_size[1], patch_size[2], channel))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                  validation_data=(x_test, y_test), verbose=2)
    pred = model.predict(x_test)
    pred2 = model.predict(x_test2)
    micro, macro, macro2 = DS.post_process(logger, y_test, pred, y_test2, pred2)
    logger.write('my method: ' + str(micro) + '  ' + str(macro) + '  ' + str(macro2) + '\n')
    logger.flush()
    del model
####################################################################################################

logger.close()
