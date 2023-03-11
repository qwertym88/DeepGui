const fs = require('fs').promises;
const path = require('path');

generate_code = async (options, dimensions, file_path) => {
    //creating the string to write (stw)
    stw = "#import statements\n";;
    if (options.framework === "TensorFlow") {
        stw += "import keras\n";
        stw += "import tensorflow as tf\n";
        stw += "from keras.layers import *\n";
        stw += "from keras.callbacks import TensorBoard\n";
        stw += `from keras.datasets import ${options.dataset}\n`;
        stw += "from datetime import datetime\n\n";

        stw += "TIMESTAMP = '{0:%H-%M-%S %m-%d-%Y/}'.format(datetime.now())\n";
        stw += "train_log_dir = 'logs/train/' + TIMESTAMP\n\n";

        stw += "#specify x_train and y_train here:\n";
        stw += `(x_train,y_train),(x_test,y_test)=${options.dataset}.load_data()\n\n`;
        stw += "#creating the model\n"
        stw += "model = keras.Sequential()\n"
        stw += "#adding layers\n"

        let input_shape = "(";
        for (let i = 0; i < dimensions.length; i++) {
            input_shape += `${dimensions[i]},`;
        }
        if (dimensions.length !== 1) {
            input_shape = input_shape.slice(0, -1);
        }

        input_shape += ")";
        if (options.layers[0].name !== "Embedding") {
            stw += `model.add(InputLayer(input_shape = ${input_shape}))\n`;
        }

        for (layer of options.layers) {
            let ret_seq = "False";
            switch (layer.name) {
                //linear case
                case "Dense":
                    stw += `model.add(Dense(${layer.unit_num}, activation = '${layer.activation.toLowerCase()}'))`
                    break;

                //convolution 1D case
                case "Convolution 1D":
                    stw += `model.add(Conv1D(${layer.filter_num}, ${layer.filter_size}, strides = ${layer.stride}, activation = '${layer.activation.toLowerCase()}', padding = '${layer.padding}'))`
                    break;

                //convolution 2D case
                case "Convolution 2D":
                    stw += `model.add(Conv2D(${layer.filter_num}, (${layer.filter_size[0]}, ${layer.filter_size[1]}), strides = ${layer.stride}, activation = '${layer.activation.toLowerCase()}', padding = '${layer.padding}'))`
                    break;

                //convolution 3D case
                case "Convolution 3D":
                    stw += `model.add(Conv3D(${layer.filter_num}, (${layer.filter_size[0]}, ${layer.filter_size[1]}, ${layer.filter_size[2]}), strides = ${layer.stride}, activation = '${layer.activation.toLowerCase()}', padding = '${layer.padding}'))`
                    break;

                //max pool 1D case
                case "Max Pool 1D":
                    stw += `model.add(MaxPooling1D(${layer.filter_size}, strides = ${layer.stride}))`
                    break;

                //max pool 2D case
                case "Max Pool 2D":
                    stw += `model.add(MaxPooling2D((${layer.filter_size[0]}, ${layer.filter_size[1]}), strides = ${layer.stride}))`
                    break;

                //max pool 3D case
                case "Max Pool 3D":
                    stw += `model.add(MaxPooling3D((${layer.filter_size[0]}, ${layer.filter_size[1]}, ${layer.filter_size[2]}), strides = ${layer.stride}))`
                    break;

                //max pool 3D case
                case "Activation":
                    if (['ELU', 'LeakyReLU', 'PReLU', 'ReLU', 'Softmax', 'ThresholdedReLU'].indexOf(layer.type) >= 0) {
                        stw += `model.add(${layer.type}())`;
                    }
                    else {
                        stw += `model.add(Activation(activations.${layer.type}))`
                    }
                    break;

                //avg pool 1D case
                case "Avg Pool 1D":
                    stw += `model.add(AveragePooling1D(${layer.filter_size}, strides = ${layer.stride}))`;
                    break;

                //avg pool 2D case
                case "Avg Pool 2D":
                    stw += `model.add(AveragePooling2D((${layer.filter_size[0]}, ${layer.filter_size[1]}), strides = ${layer.stride}))`;
                    break;

                //avg pool 3D case
                case "Avg Pool 3D":
                    stw += `model.add(AveragePooling3D((${layer.filter_size[0]}, ${layer.filter_size[1]}, ${layer.filter_size[2]}), strides = ${layer.stride}))`;
                    break;

                //batch normalization case
                case "Batch Normalization":
                    stw += `model.add(BatchNormalization())`;
                    break;

                //dropout case
                case "Dropout":
                    stw += `model.add(Dropout(rate = ${layer.prob}))`;
                    break;

                //embedding case
                case "Embedding":
                    stw += `model.add(Embedding(input_dim = ${layer.input_dim}, output_dim = ${layer.output_dim}, input_length = ${layer.input_length}))`;
                    break;

                //flatten case
                case "Flatten":
                    stw += `model.add(Flatten())`;
                    break;

                //LSTM and GRU case
                case "GRU":
                case "LSTM":
                    if (layer.ret_seq) {
                        ret_seq = "True";
                    }
                    stw += `model.add(${layer.name}(units = ${layer.units}, activations = '${layer.activation}', recurrent_activation = '${layer.re_activation}', return_sequences = ${ret_seq}))`;
                    break;

                //RNN case
                case "RNN":
                    if (layer.ret_seq) {
                        ret_seq = "True";
                    }
                    stw += `model.add(SimpleRNN(units = ${layer.units}, activations = '${layer.activation}', return_sequences = ${ret_seq}))`;
                    break;
            }
            stw += "\n"
        }
        stw += "model.summary()\n\n";

        stw += "#compiling the model\n";
        stw += `model.compile(optimizer = tf.keras.optimizers.${options.optimizer}(learning_rate = ${options.lr}), loss = '${options.loss}', metrics=['acc'])\n\n`;

        stw += "#training the model\n";
        stw += "tbCallBack = TensorBoard(log_dir=train_log_dir)\n";
        stw += `history = model.fit(x= x_train, y= y_train, batch_size = ${options.batch}, epochs = ${options.epoch},callbacks=[tbCallBack])\n\n\n`;

        stw += "#testing the model\n";
        stw += "print('\\n\\nevaluating model performance...\\n')\n";
        stw += "model.evaluate(x_test,y_test)";

        try {
            await fs.writeFile(file_path, stw);
            return true;
        }
        catch (err) {
            return false;
        }
    }
}

module.exports = generate_code;