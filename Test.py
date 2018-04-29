from PreTrain import PreTrain
import Model_3Dunet as Model

patch_size = [40, 40, 16]
batch_size = 16
epochs = 10

width = [20, 20, 8]
margin = 20
channel = 1

model = Model.get_model(open('D:\pretrain.txt', "w"), 1, input_shape=(patch_size[0], patch_size[1], patch_size[2], channel))

data = PreTrain.getdata(width, margin)

model.fit(data, data, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)
model.save('D:\model.h5')
