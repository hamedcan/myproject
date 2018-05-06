from __future__ import print_function
import Model_3Dunet as Model
from datetime import datetime
from DS import DS

patch_size = [24, 24, 8]
batch_size = 16
epochs = 150
K = 5
g_path = r'C:\result\\' + datetime.now().strftime('%Y-%m-%d--%H-%M')

ds = DS('.\data\\', patch_size, K)
model = Model.get_model(input_shape=(patch_size[0], patch_size[1], patch_size[2], 1))

for fold in range(0, K):
    x_train, y_train, x_test, y_test, x_train2, y_train2, x_test2, y_test2, x_train3, y_train3, x_test3, y_test3 = ds.get_data(fold)
    model = Model.get_model(input_shape=(patch_size[0], patch_size[1], patch_size[2], 1))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                  validation_data=(x_test, y_test))
    pred = model.predict(x_test)
    pred2 = model.predict(x_test2)
    pred3 = model.predict(x_test3)
    micro, macro, macro2 = DS.post_process(y_test, pred, y_test2, pred2, y_test3, pred3)
    x = round(x_test.shape[1] / 4)
    y = round(x_test.shape[2] / 4)
    z = round(x_test.shape[3] / 4)
