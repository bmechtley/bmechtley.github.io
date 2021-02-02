import * as THREE from './three.module.js';
import * as DAT from './dat.gui.module.js';

export const presets = {
  preset: 'flat',
  closed: false,
  remembered: {
    flat: {
      '0': {
        visualization: 'density',
        simulation: {
          origin: [0, 1], periodic: [true, true], closed: [false, false, false, false],
          pressure_iterations: 20, viscosity_iterations: 0
        },
        dynamics: {
          dx: 30, dt: 0.9, viscosity: 0, vorticity_confinement: 10,
          dissipation_velocity: [1, 1, 1, 1], dissipation_density: [0.99, 0.99, 0.95, 1],
        },
        environment: {
          gravity: [0, -0.01], lapse: 6.5, Lv: 2501000, external_viscosity_gain: 0,
          p0: 70, T0: 150, z0: -37981, dry_cv: 717, dry_cp: 1004, dry_cv: 1850, vapor_cv: 1388.5,
        },
        additional: { wind: [0, 0] },
        external: { viscosity_gain: 0, mouse_velocity: 2, mouse_radius: 0.1, mouse_sharpness: 1.0, mouse_density: [0,0.1,0,0] },
        vis_velocity: { contrast: [1,1,1,1], high: [.1,.1,.1,.1], low: [-.1,-.1,-.1,-.1], source_vars: [0,1,2,3] },
        vis_density: { contrast: [.231, .0889, 1, 1], high: [0.0048, 0.002, 785, 1], low: [0, 0, 230, 0], source_vars: [0, 1, 2, 3] }
      }
    },
    vertical: {
      '0': {
        visualization: 'density',
        simulation: {
          origin: [0, 1], periodic: [true, false], closed: [false, false, false, false],
          pressure_iterations: 20, viscosity_iterations: 0
        },
        dynamics: {
          dx: 30, dt: 0.9, viscosity: 0, vorticity_confinement: 10,
          dissipation_velocity: [.98, .98, 0], dissipation_density: [1, 1, 1],
        },
        environment: {
          gravity: [0, -0.001], lapse: 6.5, Lv: 2501000, external_viscosity_gain: 0,
          p0: 70, T0: 150, z0: 0, dry_cv: 717, dry_cp: 1004, dry_cv: 1850, vapor_cv: 1388.5
        },
        additional: { wind: [0, 0] },
        external: { viscosity_gain: 0, mouse_velocity: 2, mouse_radius: 0.1, mouse_sharpness: 1.0, mouse_density: [0,0.1,0,0] },
        vis_velocity: { contrast: [1,1,1,1], high: [.1,.1,.1,.1], low: [-.1,-.1,-.1,-.1], source_vars: [0,1,2,3] },
        vis_density: { contrast: [.231, .0889, 1, 1], high: [0.0048, 0.002, 785, 1], low: [0, 0, 230, 0], source_vars: [0, 1, 2, 3] }
      }
    }
  }
}

export const parameters = {
  render: {value: false},
  //presets: {value: 'flat', options: Object.keys(presets)},
  visualization: {value: 'density', options: ['density', 'velocity']},
  simulation: {
    pressure_iterations: {value: 10, min: 0, max: 50, step: 1},
    viscosity_iterations: {value: 0, min: 0, max: 50, step: 1},
    dim: {value: [512, 512]},
    periodic: {value: [false, false]},
    closed: {value: [false, false, false, false]},
    origin: {value: [0, 1], min: [0, 0], max: [0, 0], step: 0.01},
  },
  dynamics: {
    dissipation_density: {value: [1, 1, 1], min: [0, 0, 0], max: [1, 1, 1], step: [0.01, 0.01, 0.01]},
    dissipation_velocity: {value: [.98, .98, .98], min: [0, 0, 0], max: [1, 1, 1], step: [0.01, 0.01, 0.01]},
    viscosity: {value: 0, min: 0, max: 10, step: 0.01},
    dx: {value: 30, min: 0, max: 100, step: 0.01},
    dt: {value: 0.9, min: 0, max: 2, step: 0.01},
    vorticity_confinement: {value: 10, min: 0, max: 100, step: 0.01}
  },
  environment: {
    gravity: {value: [0, -0.001]},//, min: [-20, -20], max: [20, 20], step: [0.01, 0.01]},
    T0: {value: 150, min: 0, max: 1000, step: 0.01},
    dry_cp: {value: 1004, min: 0, max: 3000, step: 0.01},
    dry_cv: {value: 717, min: 0, max: 3000, step: 0.01},
    vapor_cp: {value: 1850, min: 0, max: 3000, step: 0.01},
    vapor_cv: {value: 1388.5, min: 0, max: 3000, step: 0.01},
    Lv: {value: 2501000, min: 0, max: 5000000, step: 1},
    lapse: {value: 6.5, min: -100, max: 100, step: 0.01},
    z0: {value: 0},//, min: -1000, max: 1000, step: 0.01},
    p0: {value: 70, min: 0, max: 1000, step: 0.01}
  },
  additional: {
    wind: {value: [0, 0], min: [-100, -100], max: [100, 100], step: [0.01, 0.01]},
  },
  external: {
    viscosity_gain: {value: 0, min: -10, max: 10, step: 0.01},
    mouse_velocity: {value: 2, min: -10, max: 10, step: 0.01},
    mouse_radius: {value: 0.1, min: 0, max: 2, step: 0.01},
    mouse_sharpness: {value: 1.0, min: 0, max: 10, step: 0.01},
    mouse_density: {value: [0, 0.01, 0, 0], min: [0, 0, 0, 0], max: [.1, .1, .1, .1]}
  },
  vis_velocity: {
    contrast: {value: [1, 1, 1, 1], min: [-10, -10, -10, -10], max: [10, 10, 10, 10], step: [0.01, 0.01, 0.01, 0.01]},
    high: {value: [1, 1, 1, 1], min: [-100, -100, -100, -100], max: [100, 100, 100, 100], step: [0.01, 0.01, 0.01, 0.01]},
    low: {value: [-1, -1, -1, -1], min: [-100, -100, -100, -100], max: [100, 100, 100, 100], step: [0.01, 0.01, 0.01, 0.01]},
    source_vars: {value: [0, 1, 2, 3], min: [0, 0, 0, 0], max: [3, 3, 3, 3], step: [1, 1, 1, 1]}
  },
  vis_density: {
    contrast: {value: [.231, .0889, 1, 1], min: [-10, -10, -10, -10], max: [10, 10, 10, 10], step: [0.01, 0.01, 0.01, 0.01]},
    high: {value: [0.0048, 0.002, 785, 1], min: [-100, -100, -100, -100], max: [100, 100, 100, 100], step: [0.01, 0.01, 0.01, 0.01]},
    low: {value: [0, 0, 230, 0], min: [-100, -100, -100, -100], max: [100, 100, 100, 100], step: [0.01, 0.01, 0.01, 0.01]},
    source_vars: {value: [0, 1, 2, 3], min: [0, 0, 0, 0], max: [3, 3, 3, 3], step: [1, 1, 1, 1]}
  }
};

export const uniforms = {}, guiparams = {}, controllers = {};
export const gui = new DAT.GUI({ load: presets });

function build_gui_params_and_uniforms(paramset = parameters, uniformset = uniforms, guiparamset = guiparams) {
  Object.keys(paramset).map(key => {
    const param = paramset[key];

    if (param.hasOwnProperty('value')) { // It's a value.
      let value = param.value, min = param.min, max = param.max, step = param.step;

      if (Array.isArray(value)) { // It's a vector.
        // Make a shader parameter, a GUI parameter binding, and a controller.
        let vector = (value.length == 2) ? new THREE.Vector2() : (value.length == 3 ? new THREE.Vector3() : new THREE.Vector4());
        vector.fromArray(value);

        uniformset[key] = {value: vector};
        guiparamset[key] = vector;
      } else { // It's a number or boolean.
        guiparamset[key] = value;
        uniformset[key] = {value: value};

      }
    } else { // It's a folder.
      guiparamset[key] = {};
      uniformset[key] = {};
      build_gui_params_and_uniforms(paramset[key], uniformset[key], guiparamset[key]);
    }
  });
}

function build_gui_controllers(paramset = parameters, guiparamset = guiparams, controllerset = controllers, folder = gui) {
  gui.remember(guiparamset);

  Object.keys(paramset).map(key => {
    const gui_param = guiparamset[key];
    const param = paramset[key];

    if (param.hasOwnProperty('value')) {
      const is_slider = ['min', 'max', 'step'].every(v => param.hasOwnProperty(v));
      gui.remember(guiparamset[key]);

      if (Array.isArray(param.value)) {
        const vfolder = folder.addFolder(key);
        controllerset[key] = {};
        param.value.map((v, i) => {
          var d = ['x', 'y', 'z', 'w'][i];
          controllerset[key][d] = is_slider ?
            vfolder.add(gui_param, d, param.min[i], param.max[i], param.step[i]) :
            vfolder.add(gui_param, d);
        });
      } else if (is_slider) controllerset[key] = folder.add(guiparamset, key, param.min, param.max, param.step);
      else if (param.hasOwnProperty('options')) controllerset[key] = folder.add(guiparamset, key, param.options);
      else controllerset[key] = folder.add(guiparamset, key);
    } else if (typeof param === 'object') {
      controllerset[key] = {};
      build_gui_controllers(param, gui_param, controllerset[key], folder.addFolder(key));
    }
  });
}

build_gui_params_and_uniforms();
gui.remember(guiparams);
build_gui_controllers();

uniforms.simulation.time = {value: 0};
export const stats = new Stats();
stats.setMode(0);
stats.domElement.style.position = 'absolute';
stats.domElement.style.left = '0';
stats.domElement.style.top = '0';
document.body.appendChild(stats.domElement);
