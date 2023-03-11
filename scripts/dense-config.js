const { ipcRenderer } = require('electron');
let layer;

document.getElementById('cancle').addEventListener('click', () => {
    ipcRenderer.send('close-small');
});

document.getElementById('done').addEventListener('click', () => {
    let complete = true;
    if (document.getElementById("number-of-units").value == "") { complete = false }
    if (complete === false) {
        return;
    }
    layer.unit_num = document.getElementById("number-of-units").value;
    layer.activation = document.getElementById("activation-selector").value;
    ipcRenderer.send("layer-config-finish", layer);
});

ipcRenderer.on("layer-config", (event, arg) => {
    layer_config = arg.layer;
    document.getElementById("number-of-units").value = layer_config.unit_num;
    layer = layer_config;
    document.getElementById("activation-selector").value = layer_config.activation;
});

(function () {
    ipcRenderer.send('ready-layer-config');
})();