import {vec2, vec3} from 'gl-matrix';
import * as Stats from 'stats-js';
import * as DAT from 'dat-gui';
import Square from './geometry/Square';
import OpenGLRenderer from './rendering/gl/OpenGLRenderer';
import Camera from './Camera';
import {setGL} from './globals';
import ShaderProgram, {Shader} from './rendering/gl/ShaderProgram';

// Define an object with application parameters and button callbacks
// This will be referred to by dat.GUI's functions that add GUI elements.
const controls = {
  blob_effect: 1.2,
  ring_size: 2.5,
  mountainCount: 40.0,
  //'Load Scene': loadScene, // A function pointer, essentially
  //'Color': [255, 0, 0, 1],
  //'Shaders': 'Lambert',
  'R': 250,
  'G': 230,
  'B':178
};

let square: Square;
let time: number = 0;
let prevRingSize: number = 2.5;
let prevR = 250.0;
let prevG = 230.0;
let prevB = 178.0;
let prevBlob = 1.2;


function loadScene() {
  square = new Square(vec3.fromValues(0, 0, 0));
  square.create();
  // time = 0;
}

function main() {
  window.addEventListener('keypress', function (e) {
    // console.log(e.key);
    switch(e.key) {
        // Use this if you wish
    }
  }, false);

  window.addEventListener('keyup', function (e) {
    switch(e.key) {
        // Use this if you wish
    }
  }, false);

  // Initial display for framerate
  const stats = Stats();
  stats.setMode(0);
  stats.domElement.style.position = 'absolute';
  stats.domElement.style.left = '0px';
  stats.domElement.style.top = '0px';
  document.body.appendChild(stats.domElement);

  // Add controls to the gui
  const gui = new DAT.GUI();
  gui.add(controls, 'ring_size', 1.0, 6.0).step(0.1);
  gui.add(controls, 'blob_effect', 0.5, 2.5).step(0.1);
  var colorFolder = gui.addFolder('Change Color');
  colorFolder.add(controls, "R", 0, 255).step(1);
  colorFolder.add(controls, "G", 0, 255).step(1);
  colorFolder.add(controls, "B", 0, 255).step(1);


  // get canvas and webgl context
  const canvas = <HTMLCanvasElement> document.getElementById('canvas');
  const gl = <WebGL2RenderingContext> canvas.getContext('webgl2');
  if (!gl) {
    alert('WebGL 2 not supported!');
  }
  // `setGL` is a function imported above which sets the value of `gl` in the `globals.ts` module.
  // Later, we can import `gl` from `globals.ts` to access it
  setGL(gl);

  // Initial call to load scene
  loadScene();

  const camera = new Camera(vec3.fromValues(-2, 3, -8), vec3.fromValues(-1.5, 1.2, 0));
  //const camera = new Camera(vec3.fromValues(-10, 7, -10), vec3.fromValues(-3, 5, 0));

  const renderer = new OpenGLRenderer(canvas);
  renderer.setClearColor(164.0 / 255.0, 233.0 / 255.0, 1.0, 1);
  gl.enable(gl.DEPTH_TEST);

  const flat = new ShaderProgram([
    new Shader(gl.VERTEX_SHADER, require('./shaders/flat-vert.glsl')),
    new Shader(gl.FRAGMENT_SHADER, require('./shaders/flat-frag.glsl')),
  ]);
  flat.setRingSize(prevRingSize);
  flat.setColor(vec3.fromValues(prevR / 255.0, prevG / 255.0, prevB / 255.0));
  flat.setBlob(prevBlob);

  function processKeyPresses() {
    // Use this if you wish
  }

  let r = 1;
  let g = 0;
  let b = 0;

  // This function will be called every frame
  function tick() {
    camera.update();
    stats.begin();
    gl.viewport(0, 0, window.innerWidth, window.innerHeight);
    renderer.clear();

    //Gui Controls
    if(controls.ring_size != prevRingSize)
    {
      prevRingSize = controls.ring_size;
      flat.setRingSize(prevRingSize);
    }

    if(controls.blob_effect != prevBlob)
    {
      prevBlob = controls.blob_effect;
      flat.setBlob(prevBlob);
    }


    if(controls.R != prevR) {
      prevR = controls.R;
      r = prevR / 255.0;
      flat.setColor(vec3.fromValues(r, g, b));

    }

    if(controls.G != prevG) {
      prevG = controls.G;
      g = prevG / 255.0;
      flat.setColor(vec3.fromValues(r, g, b));

    }

    if(controls.B != prevB) {
      prevB = controls.B;
      b = prevB / 255.0;
      flat.setColor(vec3.fromValues(r, b, b));

    }

    processKeyPresses();
    renderer.render(camera, flat, [
      square,
    ], time);
    time++;
    stats.end();

    // Tell the browser to call `tick` again whenever it renders a new frame
    requestAnimationFrame(tick);
  }

  window.addEventListener('resize', function() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.setAspectRatio(window.innerWidth / window.innerHeight);
    camera.updateProjectionMatrix();
    flat.setDimensions(window.innerWidth, window.innerHeight);
  }, false);

  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.setAspectRatio(window.innerWidth / window.innerHeight);
  camera.updateProjectionMatrix();
  flat.setDimensions(window.innerWidth, window.innerHeight);

  // Start the render loop
  tick();
}

main();
