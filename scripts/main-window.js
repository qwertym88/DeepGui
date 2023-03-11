const electron = require('electron');
const { ipcRenderer } = electron;
const add_layer_to_window = require("../scripts/utils/new-layer.js");
const change_desc = require("../scripts/utils/change-desc.js");

let grBrxZdqwDVxusulvh = true;

const canvas = document.getElementById('bgCanvas') // 背景效果
const ctx = canvas.getContext('2d')
let width = window.innerWidth
let height = window.innerHeight

let dotsNum = 80
let radius = 1
let fillStyle = 'rgba(255,255,255,0.5)'
let lineWidth = radius * 2
let connection = 120
let followLength = 80

let dots = [];
let animationFrame = null;
let mouseX = null;
let mouseY = null;
let mouseOn = false;

function addCanvasSize() {
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);
    dots = [];
    if (animationFrame) window.cancelAnimationFrame(animationFrame);
    initDots(dotsNum);
    moveDots();
}

function mouseMove(e) {
    if (mouseOn) {
        mouseX = e.clientX;
        mouseY = e.clientY;
    }
}

function mouseOut(e) {
    mouseX = null;
    mouseY = null;
}

function mouseDown() {
    mouseOn = true;
}

function mouseUp() {
    mouseOn = false;
    for (const dot of dots) dot.elastic();
}

class Dot {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.speedX = Math.random() * 2 - 1;
        this.speedY = Math.random() * 2 - 1;
        this.follow = false;
    }
    draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, radius, 0, 2 * Math.PI);
        ctx.fill();
        ctx.closePath();
    }
    move() {
        if (this.x >= width || this.x <= 0) this.speedX = -this.speedX;
        if (this.y >= height || this.y <= 0) this.speedY = -this.speedY;
        this.x += this.speedX;
        this.y += this.speedY;
        if (this.speedX >= 1) this.speedX--;
        if (this.speedX <= -1) this.speedX++;
        if (this.speedY >= 1) this.speedY--;
        if (this.speedY <= -1) this.speedY++;
        this.correct();
        let connected = this.connectMouse();
        this.draw();
        return connected;
    }
    correct() {
        if (!mouseX || !mouseY) return;
        let lengthX = mouseX - this.x;
        let lengthY = mouseY - this.y;
        const distance = Math.sqrt(lengthX ** 2 + lengthY ** 2)
        if (distance <= followLength) this.follow = true;
        else if (this.follow === true && distance > followLength && distance <= followLength + 8) {
            let proportion = followLength / distance;
            lengthX *= proportion;
            lengthY *= proportion;
            this.x = mouseX - lengthX;
            this.y = mouseY - lengthY;
        } else this.follow = false;
    }
    connectMouse() {
        if (mouseX && mouseY) {
            let lengthX = mouseX - this.x;
            let lengthY = mouseY - this.y;
            const distance = Math.sqrt(lengthX ** 2 + lengthY ** 2);
            if (distance <= connection) {
                opacity = (1 - distance / connection) * 0.5;
                ctx.strokeStyle = `rgba(0,0,0,${opacity})`;
                ctx.beginPath();
                ctx.moveTo(this.x, this.y);
                ctx.lineTo(mouseX, mouseY);
                ctx.stroke();
                ctx.closePath();
                return 1;
            }
        }
        return 0;
    }
    elastic() {
        let lengthX = mouseX - this.x;
        let lengthY = mouseY - this.y;
        const distance = Math.sqrt(lengthX ** 2 + lengthY ** 2);
        if (distance >= connection) return;
        const rate = 1 - distance / connection;
        this.speedX = 40 * rate * -lengthX / distance;
        this.speedY = 40 * rate * -lengthY / distance;
    }
}

function initDots(num) { // 初始化粒子
    ctx.fillStyle = fillStyle;
    ctx.lineWidth = lineWidth;
    for (let i = 0; i < num; i++) {
        const x = Math.floor(Math.random() * width);
        const y = Math.floor(Math.random() * height);
        const dot = new Dot(x, y);
        dot.draw();
        dots.push(dot);
    }
}

function moveDots() {
    let dotsNum = 0;
    ctx.clearRect(0, 0, width, height)
    for (const dot of dots) {
        dotsNum += dot.move();
        if (dotsNum >= 40 && grBrxZdqwDVxusulvh) {
            grBrxZdqwDVxusulvh = false;
            display_layer();
        }
    }
    for (let i = 0; i < dots.length; i++) {
        for (let j = i; j < dots.length; j++) {
            const distance = Math.sqrt((dots[i].x - dots[j].x) ** 2 + (dots[i].y - dots[j].y) ** 2);
            if (distance <= connection) {
                opacity = (1 - distance / connection) * 0.5;
                ctx.strokeStyle = `rgba(0,0,0,${opacity})`;
                ctx.beginPath();
                ctx.moveTo(dots[i].x, dots[i].y);
                ctx.lineTo(dots[j].x, dots[j].y);
                ctx.stroke();
                ctx.closePath();
            }
        }
    }
    animationFrame = window.requestAnimationFrame(moveDots);
}

addCanvasSize();

initDots(dotsNum);
moveDots();

document.onmousemove = mouseMove;
document.onmouseout = mouseOut;
document.onmousedown = mouseDown;
document.onmouseup = mouseUp;
window.onresize = addCanvasSize;

let layers = [];
let layers_count = 0;
let framework = "TensorFlow";

const diagram = () => {
    return {
        framework: document.getElementById('framework-selector').value,
        dataset: document.getElementById('dataset-selector').value,
        optimizer: document.getElementById('optimizer-selector').value,
        lr: parseFloat(document.getElementById('optimizer-lr').value),
        loss: document.getElementById('loss-function-selector').value,
        epoch: parseInt(document.getElementById('epoch').value),
        batch: parseInt(document.getElementById('batch').value),
        layers: layers
    }
}

const delete_layer = (element) => {
    for (let i = 0; i < layers.length; i++) {
        if (layers[i].id === element.id.slice(0, -6)) {
            layers.splice(i, 1);
            layer_to_remove = document.getElementById(element.id.slice(0, -6));
            layer_to_remove.parentNode.removeChild(layer_to_remove);
            break;
        }
    }
}

const layer_config = element => {
    for (let i = 0; i < layers.length; i++) {
        if (layers[i].id === element.id.slice(0, -4)) {
            ipcRenderer.send("config-layer", layers[i])
            break;
        }
    }
}

const add_new_layer_buttons = element => {
    ipcRenderer.send('new-layer-request', element.id.slice(0, -4));
}

const remove_attention = element => {
    if (element.id === "optimizer-lr") {
        document.getElementById('lr-attention').setAttribute('style', 'opacity: 0');
    }
    else if (element.id === "batch") {
        document.getElementById('batch-attention').setAttribute('style', 'opacity: 0');
    }
    else if (element.id === "epoch") {
        document.getElementById('epoch-attention').setAttribute('style', 'opacity: 0');
    }
}

const check_and_generate = () => {
    if (document.getElementById('optimizer-lr').value <= 0) {
        document.getElementById('lr-attention').setAttribute('style', 'opacity: 1');
    }
    else if (document.getElementById('epoch').value < 1) {
        document.getElementById('lr-attention').setAttribute('style', 'opacity: 0');
        document.getElementById('epoch-attention').setAttribute('style', 'opacity: 1');
    }
    else if (document.getElementById('batch').value < 1) {
        document.getElementById('lr-attention').setAttribute('style', 'opacity: 0');
        document.getElementById('epoch-attention').setAttribute('style', 'opacity: 0');
        document.getElementById('batch-attention').setAttribute('style', 'opacity: 1');
    }
    else if (layers.length === 0) {
        document.getElementById('lr-attention').setAttribute('style', 'opacity: 0');
        document.getElementById('epoch-attention').setAttribute('style', 'opacity: 0');
        document.getElementById('batch-attention').setAttribute('style', 'opacity: 0');
        document.getElementById('layer-attention').setAttribute('style', 'opacity: 1');
    }
    else {
        document.getElementById('lr-attention').setAttribute('style', 'opacity: 0');
        document.getElementById('epoch-attention').setAttribute('style', 'opacity: 0');
        document.getElementById('batch-attention').setAttribute('style', 'opacity: 0');
        document.getElementById('layer-attention').setAttribute('style', 'opacity: 0');
        ipcRenderer.send('generate-code', diagram());
    }
}

document.getElementById('close-btn').addEventListener('click', () => {
    ipcRenderer.send('exit-app');
});

document.getElementById('min-btn').addEventListener('click', () => {
    ipcRenderer.send('min-app');
});

document.getElementById('max-btn').addEventListener('click', () => {
    ipcRenderer.send('max-app');
});

document.getElementById('new-layer-button').addEventListener('click', () => {
    add_new_layer_buttons({ id: 'new-layer-button-parent-add' })
});

document.getElementById('generate-button').addEventListener('click', () => {
    check_and_generate();
});

document.getElementById('save-button').addEventListener('click', () => {
    ipcRenderer.send('save-diagram', diagram());
});

document.getElementById('load-button').addEventListener('click', () => {
    ipcRenderer.send('load-diagram');
});


document.getElementById('input-shape-cog').addEventListener('click', () => {
    ipcRenderer.send('input-shape-cog');
});

ipcRenderer.on('add-new-layer', (event, args) => {
    add_layer_to_window(args, layers_count, framework);
    layers_count += 1;
    document.getElementById('layer-attention').setAttribute('style', 'opacity: 0');
});

//setting input shape
ipcRenderer.on('set-input-shape', (event, arg) => {
    let shape = "";
    for (dim of arg) {
        shape += `${dim}, `;
    }
    shape = shape.slice(0, -2);
    document.getElementById('input-shape-text').innerHTML = shape;
})

//getting configurations
ipcRenderer.on('set-config', (event, layer) => {
    for (let i = 0; i < layers.length; i++) {
        if (layers[i].id === layer.id) {
            layers[i] = layer;
            change_desc(layer, framework);
            break;
        }
    }
});

//cleaning the diagram
ipcRenderer.on("load-new-diagram", (event, arg) => {
    layers_count = 0;
    layers = [];
    const diagram_layers = document.getElementsByClassName('layer-class');
    while (diagram_layers.length > 0) {
        diagram_layers[0].parentNode.removeChild(diagram_layers[0]);
    }
    document.getElementById("framework-selector").value = arg.framework;
    document.getElementById('optimizer-lr').value = arg.lr;
    document.getElementById('epoch').value = arg.epoch;
    document.getElementById('batch').value = arg.batch;
    document.getElementById('optimizer-selector').value = arg.optimizer;
    document.getElementById('loss-function-selector').value = arg.loss;
})

window.onload = function () {
    document.getElementsByClassName('heading-4-parnet')[0].style.display = 'none';
}